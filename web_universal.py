# -*- coding: utf-8 -*-
import os
import re
import time
import hashlib
import json
import requests
import zipfile
import shutil
import importlib.util
import sys
import tempfile
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, send_from_directory
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
import psutil

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---------- Disable Playwright on Railway (avoid crash) ----------
PLAYWRIGHT_AVAILABLE = False

# ---------- OpenRouter settings (free Gemini models) ----------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TEXT_MODEL = "google/gemini-2.0-flash-001"
IMAGE_MODEL = "google/gemini-2.5-flash-image"

def ask_openrouter(prompt, model=None):
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not set.", None
    if model is None:
        model = DEFAULT_TEXT_MODEL
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"]
            return reply, None
        else:
            return f"OpenRouter error: {resp.status_code}", None
    except Exception as e:
        return f"Error: {str(e)}", None

def generate_image_with_openrouter(prompt, image_base64=None):
    if not OPENROUTER_API_KEY:
        return None, "No API key"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    content_parts = [{"type": "text", "text": prompt}]
    if image_base64:
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })
    payload = {
        "model": IMAGE_MODEL,
        "messages": [{"role": "user", "content": content_parts}],
        "temperature": 0.7
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        msg = resp.json()["choices"][0]["message"]["content"]
        url_match = re.search(r'(https?://[^\s]+\.(png|jpg|jpeg|gif|webp))', msg, re.IGNORECASE)
        if url_match:
            img_url = url_match.group(1)
            img_resp = requests.get(img_url, timeout=30)
            if img_resp.status_code == 200:
                os.makedirs("data", exist_ok=True)
                fname = f"generated_image_{int(time.time())}.png"
                path = os.path.join("data", fname)
                with open(path, "wb") as f:
                    f.write(img_resp.content)
                return path, None
        return None, "No image URL found"
    except Exception as e:
        return None, str(e)

# ---------- Memory (ChromaDB with TFIDF – no downloads, safe for Railway) ----------
USER_HOME = os.environ.get("USERPROFILE", os.path.expanduser("~")) if os.name == 'nt' else os.environ.get("HOME", ".")
CHROMA_PATH = os.path.join(USER_HOME, ".studio_memory")
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = embedding_functions.TFIDFEmbeddingFunction()
memory_collection = chroma_client.get_or_create_collection(
    name="studio_memory",
    embedding_function=embedding_fn
)

def store_memory(user_input, action, output, metadata=None):
    doc_id = f"{int(time.time())}_{hashlib.md5(user_input.encode()).hexdigest()[:8]}"
    doc = f"User: {user_input}\nAction: {action}\nOutput: {output[:500]}"
    meta = {"user_input": user_input, "action": action, "timestamp": time.time()}
    if metadata:
        meta.update(metadata)
    memory_collection.upsert(documents=[doc], metadatas=[meta], ids=[doc_id])

def recall_memory(query, n_results=3):
    results = memory_collection.query(query_texts=[query], n_results=n_results)
    if results and results["documents"] and results["documents"][0]:
        return results["documents"][0]
    return []

# ---------- Plugin system ----------
PLUGINS_DIR = os.path.join(os.path.dirname(__file__), "plugins")
os.makedirs(PLUGINS_DIR, exist_ok=True)
init_path = os.path.join(PLUGINS_DIR, "__init__.py")
if not os.path.exists(init_path):
    with open(init_path, "w") as f:
        f.write("")

_plugins = {}

def load_plugins():
    global _plugins
    _plugins = {}
    for filename in os.listdir(PLUGINS_DIR):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            file_path = os.path.join(PLUGINS_DIR, filename)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                if hasattr(module, "run") and callable(module.run):
                    info = {"run": module.run}
                    if hasattr(module, "get_info") and callable(module.get_info):
                        info.update(module.get_info())
                    else:
                        info["name"] = module_name.replace("_", " ").title()
                        info["description"] = f"Plugin {module_name}"
                    _plugins[info["name"].lower().replace(" ", "_")] = info
                    print(f"[Plugin] Loaded: {info['name']}")
            except Exception as e:
                print(f"[Plugin] Error loading {filename}: {e}")

load_plugins()

def get_plugins_info():
    return {name: {"name": info.get("name", name), "description": info.get("description", "")} for name, info in _plugins.items()}

def run_plugin(plugin_name, args):
    if plugin_name not in _plugins:
        return {"error": f"Plugin '{plugin_name}' not found"}
    try:
        result = _plugins[plugin_name]["run"](args)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# ---------- Helper functions ----------
def clean_html(raw):
    raw = re.sub(r'^```[^\n]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    raw = raw.replace('```', '')
    raw = re.sub(r'`html`', '', raw)
    return raw.strip()

def save_file(content, filename):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)
    if isinstance(content, bytes):
        with open(path, "wb") as f:
            f.write(content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return path

# ---------- Core tools (without Playwright) ----------
def generate_website(task, filename, style_guide, model=None):
    memories = recall_memory("website " + task, n_results=2)
    memory_context = "\n".join(memories) if memories else ""
    prompt = f"""You are an expert front-end developer. The user asks: {task}
Follow this style guide: {style_guide}
Similar past successful examples:
{memory_context}
Create a complete, standalone HTML page. Output ONLY raw HTML starting with <!DOCTYPE html>. No backticks."""
    response, _ = ask_openrouter(prompt, model=model)
    clean = clean_html(response)
    html_match = re.search(r'<!DOCTYPE\s+html[^>]*>.*?</html>', clean, re.DOTALL | re.IGNORECASE)
    if not html_match:
        html_match = re.search(r'<html[^>]*>.*?</html>', clean, re.DOTALL | re.IGNORECASE)
    if html_match:
        html_content = html_match.group(0)
        safe_title = re.sub(r'[\\/*?:"<>|]', "", filename)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        full_filename = f"{safe_title}_{timestamp}.html"
        saved_path = save_file(html_content, full_filename)
        result = {"html": html_content, "path": saved_path, "error": None}
        store_memory(task, "website_generation", result.get("html", ""))
        return result, None
    return {"error": "Could not extract valid HTML", "raw": response}, None

def refine_html(original_html, instruction, model=None):
    refine_prompt = f"""You are an expert front-end developer. Here is an HTML document:
{original_html}
The user wants: {instruction}
Output the **complete** refined HTML code, starting with <!DOCTYPE html>. No triple backticks."""
    response, _ = ask_openrouter(refine_prompt, model=model)
    clean = clean_html(response)
    html_match = re.search(r'<!DOCTYPE\s+html[^>]*>.*?</html>', clean, re.DOTALL | re.IGNORECASE)
    if not html_match:
        html_match = re.search(r'<html[^>]*>.*?</html>', clean, re.DOTALL | re.IGNORECASE)
    if html_match:
        new_html = html_match.group(0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"refined_{timestamp}.html"
        saved_path = save_file(new_html, filename)
        result = {"new_html": new_html, "path": saved_path, "error": None}
        store_memory(instruction, "html_refinement", result.get("new_html", ""))
        return result, None
    return {"error": "No valid HTML", "raw": response}, None

def convert_code_to_html(code, model=None):
    prompt = f"""You are a helpful assistant. The user provided code:
```{code}```
Create a **complete, standalone HTML page** that displays this code nicely. Output ONLY raw HTML starting with <!DOCTYPE html>. No backticks."""
    response, _ = ask_openrouter(prompt, model=model)
    clean = clean_html(response)
    html_match = re.search(r'<!DOCTYPE\s+html[^>]*>.*?</html>', clean, re.DOTALL | re.IGNORECASE)
    if not html_match:
        html_match = re.search(r'<html[^>]*>.*?</html>', clean, re.DOTALL | re.IGNORECASE)
    if html_match:
        html_content = html_match.group(0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"code_display_{timestamp}.html"
        saved_path = save_file(html_content, filename)
        result = {"html": html_content, "path": saved_path, "error": None}
        store_memory(code[:100], "code_to_html", result.get("html", ""))
        return result, None
    return {"error": "Failed", "raw": response}, None

def refine_code(code, instruction, model=None):
    prompt = f"""You are an expert programmer. The user provided code:
```{code}```
The user wants: {instruction}
Output ONLY the refined code, no explanations, no backticks."""
    response, _ = ask_openrouter(prompt, model=model)
    refined = re.sub(r'^```[^\n]*\n?', '', response)
    refined = re.sub(r'\n?```$', '', refined)
    result = {"refined_code": refined, "error": None}
    store_memory(instruction, "code_refinement", result.get("refined_code", ""))
    return result, None

def web_search(query, model=None):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            formatted = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
            result = {"results": formatted, "error": None}
            store_memory(query, "web_search", result.get("results", ""))
            return result, None
    except Exception as e:
        return {"error": str(e)}, None

def generate_app(description, language, filename_base, model=None):
    lang_ext = {"python": ".py", "powershell": ".ps1", "bash": ".sh", "batch": ".bat"}
    ext = lang_ext.get(language, ".txt")
    prompt = f"You are a senior software engineer. The user wants: {description}\nGenerate complete, ready-to-run code in {language}. Output ONLY raw code, no backticks."
    code, _ = ask_openrouter(prompt, model=model)
    code = re.sub(r'^```[^\n]*\n?', '', code)
    code = re.sub(r'\n?```$', '', code)
    if not filename_base:
        words = re.findall(r'\b\w+\b', description.lower())
        base = "_".join(words[:3]) if words else "app"
    else:
        base = re.sub(r'[\\/*?:"<>|]', "", filename_base)
    filename = f"{base}{ext}"
    saved_path = save_file(code, filename)
    result = {"code": code, "path": saved_path, "error": None}
    store_memory(description, "app_generation", result.get("code", ""))
    return result, None

# ---------- Flask routes (most important endpoints) ----------
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": [DEFAULT_TEXT_MODEL, IMAGE_MODEL]})

@app.route("/list_plugins", methods=["GET"])
def list_plugins():
    return jsonify({"plugins": get_plugins_info()})

@app.route("/run_plugin", methods=["POST"])
def run_plugin_endpoint():
    data = request.get_json()
    plugin_name = data.get("plugin")
    args = data.get("args", {})
    result = run_plugin(plugin_name, args)
    return jsonify(result)

@app.route("/generate", methods=["POST"])
def route_generate():
    data = request.get_json()
    task = data.get("task", "")
    filename = data.get("filename", "website")
    style_guide = data.get("styleGuide", "")
    model = data.get("model", DEFAULT_TEXT_MODEL)
    res, _ = generate_website(task, filename, style_guide, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"html": res["html"], "path": res["path"]})

@app.route("/refine", methods=["POST"])
def route_refine():
    data = request.get_json()
    original_html = data.get("html", "")
    instruction = data.get("instruction", "")
    model = data.get("model", DEFAULT_TEXT_MODEL)
    res, _ = refine_html(original_html, instruction, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"new_html": res["new_html"], "path": res["path"]})

@app.route("/convert_code", methods=["POST"])
def route_convert_code():
    data = request.get_json()
    code = data.get("code", "")
    model = data.get("model", DEFAULT_TEXT_MODEL)
    res, _ = convert_code_to_html(code, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"html": res["html"], "path": res["path"]})

@app.route("/refine_code", methods=["POST"])
def route_refine_code():
    data = request.get_json()
    code = data.get("code", "")
    instruction = data.get("instruction", "")
    model = data.get("model", DEFAULT_TEXT_MODEL)
    res, _ = refine_code(code, instruction, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"refined_code": res["refined_code"]})

@app.route("/search", methods=["POST"])
def route_search():
    data = request.get_json()
    query = data.get("query", "")
    res, _ = web_search(query)
    if res.get("error"):
        return jsonify({"error": res["error"]})
    return jsonify({"results": res["results"]})

@app.route("/generate_app", methods=["POST"])
def route_generate_app():
    data = request.get_json()
    description = data.get("description", "")
    language = data.get("language", "python")
    filename = data.get("filename", "")
    model = data.get("model", DEFAULT_TEXT_MODEL)
    res, _ = generate_app(description, language, filename, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"code": res["code"], "path": res["path"]})

@app.route("/chat", methods=["POST"])
def route_chat():
    data = request.get_json()
    message = data.get("message", "")
    model = data.get("model", DEFAULT_TEXT_MODEL)
    relevant = recall_memory(message, n_results=5)
    memory_context = ""
    if relevant:
        memory_context = "Relevant past memories:\n" + "\n".join(relevant) + "\n\n"
    name_match = re.search(r"(my name is|call me|i am) (\w+)", message, re.IGNORECASE)
    if name_match:
        user_name = name_match.group(2)
        store_memory(message, "personal_info", f"User's name is {user_name}")
        memory_context += f"IMPORTANT: The user's name is {user_name}. Always address them by name.\n"
    if 'conv_history' not in session:
        session['conv_history'] = []
    history = session['conv_history']
    prompt = "You are a helpful AI assistant that remembers past conversations.\n"
    prompt += memory_context
    for msg in history[-10:]:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"User: {message}\nAssistant:"
    reply, _ = ask_openrouter(prompt, model=model, use_cache=False)
    store_memory(message, "chat_interaction", reply)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    session['conv_history'] = history[-20:]
    return jsonify({"reply": reply})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop('conv_history', None)
    return jsonify({"status": "cleared"})

@app.route("/gemini_image", methods=["POST"])
def route_gemini_image():
    data = request.get_json()
    prompt = data.get("prompt", "")
    image_base64 = data.get("image", None)
    if not prompt:
        return jsonify({"error": "No prompt"})
    save_path, error = generate_image_with_openrouter(prompt, image_base64)
    if error:
        return jsonify({"error": error})
    return jsonify({"path": save_path, "status": "success"})

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    prefix = request.args.get("prefix", "").strip()
    if not prefix or len(prefix) < 2:
        return jsonify({"suggestions": []})
    suggestions = set()
    memories = recall_memory(prefix, n_results=5)
    for mem in memories:
        match = re.search(r"User: (.*?)(?:\n|$)", mem)
        if match:
            user_input = match.group(1).strip()
            if user_input:
                suggestions.add(user_input[:100])
    plugins = get_plugins_info()
    for pname, info in plugins.items():
        suggestions.add(info["name"])
    tool_commands = ["generate website", "refine html", "convert code", "refine code", "search web", "generate app"]
    for cmd in tool_commands:
        if cmd.startswith(prefix.lower()) or prefix.lower() in cmd:
            suggestions.add(cmd)
    suggestions = sorted(suggestions)[:10]
    return jsonify({"suggestions": suggestions})

@app.route("/system_stats", methods=["GET"])
def system_stats():
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    ram_percent = memory.percent
    ram_used_gb = memory.used / (1024**3)
    ram_total_gb = memory.total / (1024**3)
    return jsonify({
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "ram_used_gb": round(ram_used_gb, 1),
        "ram_total_gb": round(ram_total_gb, 1)
    })

@app.route("/stream_smart_agent", methods=["GET"])
def stream_smart_agent():
    request_text = request.args.get("request", "")
    style_guide = request.args.get("styleGuide", "")
    model = request.args.get("model", DEFAULT_TEXT_MODEL)
    if not request_text:
        return Response("data: {}\n\n".format(json.dumps({"error": "No request"})), mimetype="text/event-stream")
    def generate():
        # Image generation keywords
        image_keywords = ["draw", "generate image", "create an image", "generate a picture", "make an image", "image of", "picture of"]
        if any(kw in request_text.lower() for kw in image_keywords):
            save_path, error = generate_image_with_openrouter(request_text, None)
            if error:
                yield f"data: {json.dumps({'error': error})}\n\n"
            else:
                yield f"data: {json.dumps({'result': f'Image saved to {save_path}', 'path': save_path, 'tool': 'image_generation'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
        # Email plugin
        email_keywords = ["send an email", "send email", "email to", "mail to"]
        if any(kw in request_text.lower() for kw in email_keywords):
            args = {"to": "recipient@example.com"}
            email_match = re.search(r'[\w\.-]+@[\w\.-]+', request_text)
            if email_match:
                args["to"] = email_match.group(0)
            res = run_plugin("email_sender", args)
            if "error" in res:
                yield f"data: {json.dumps({'error': res['error']})}\n\n"
            else:
                yield f"data: {json.dumps({'result': res.get('result', ''), 'tool': 'email_sender'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
        # For all other requests, simple chat (no complex tool chaining to avoid errors)
        memories = recall_memory(request_text, n_results=3)
        memory_context = "\n".join(memories) if memories else ""
        prompt = f"Memory:\n{memory_context}\n\nUser: {request_text}\nAssistant:"
        reply, _ = ask_openrouter(prompt, model=model)
        yield f"data: {json.dumps({'token': reply})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        store_memory(request_text, "smart_agent", reply)
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route('/favicon.ico')
def favicon():
    return '', 204

# ---------- Backup & memory management ----------
@app.route("/backup_memory", methods=["POST"])
def backup_memory():
    try:
        backup_dir = os.path.join(os.environ["USERPROFILE"], "Desktop", "studio_backups")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"memory_backup_{timestamp}.zip"
        backup_path = os.path.join(backup_dir, backup_name)
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(CHROMA_PATH):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(CHROMA_PATH))
                    zipf.write(file_path, arcname)
        return jsonify({"status": "success", "path": backup_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/restore_memory", methods=["POST"])
def restore_memory():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not file.filename.endswith('.zip'):
        return jsonify({"error": "Need .zip"}), 400
    temp_dir = tempfile.mkdtemp()
    try:
        zip_path = os.path.join(temp_dir, "backup.zip")
        file.save(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        backup_folder = None
        for root, dirs, files in os.walk(temp_dir):
            if '.studio_memory' in dirs:
                backup_folder = os.path.join(root, '.studio_memory')
                break
        if not backup_folder:
            return jsonify({"error": "Invalid backup"}), 400
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        shutil.copytree(backup_folder, CHROMA_PATH)
        return jsonify({"status": "success", "message": "Memory restored. Restart to apply."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route("/list_memories", methods=["GET"])
def list_memories():
    search = request.args.get("search", "")
    limit = int(request.args.get("limit", 50))
    if search:
        results = memory_collection.query(query_texts=[search], n_results=limit)
    else:
        results = memory_collection.query(query_texts=["a"], n_results=limit)
    memories = []
    if results and results['ids'] and results['ids'][0]:
        for i, doc_id in enumerate(results['ids'][0]):
            mem = {
                "id": doc_id,
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
            }
            memories.append(mem)
    return jsonify({"memories": memories})

@app.route("/delete_memory", methods=["POST"])
def delete_memory():
    data = request.get_json()
    mem_id = data.get("id")
    if not mem_id:
        return jsonify({"error": "No ID"}), 400
    try:
        memory_collection.delete(ids=[mem_id])
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compress_memory", methods=["POST"])
def compress_memory():
    # Simple version: just delete memories older than 30 days (no extra LLM call to avoid cost)
    data = request.get_json() or {}
    days_old = data.get("days_old", 30)
    try:
        results = memory_collection.get(limit=1000)
        if not results or not results['ids']:
            return jsonify({"error": "No memories"}), 400
        now = time.time()
        to_delete = []
        for i, doc_id in enumerate(results['ids']):
            meta = results['metadatas'][i] if results['metadatas'] else {}
            ts = meta.get("timestamp", 0)
            if ts and (now - ts) > days_old * 86400:
                to_delete.append(doc_id)
        if to_delete:
            memory_collection.delete(ids=to_delete)
        return jsonify({"message": f"Deleted {len(to_delete)} old memories."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 AI Studio starting on port {port}")
    print(f"☁️ Playwright disabled (safe for Railway)")
    app.run(debug=False, host="0.0.0.0", port=port)