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
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, send_from_directory
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
from playwright.sync_api import sync_playwright
import psutil

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ----------------------------------------------------------------------
# Portable home directory (works on Windows and Linux)
# ----------------------------------------------------------------------
def get_user_home():
    if os.name == 'nt':
        return os.environ.get("USERPROFILE", os.path.expanduser("~"))
    else:
        return os.environ.get("HOME", os.path.expanduser("~"))

USER_HOME = get_user_home()

# ----------------------------------------------------------------------
# OpenRouter API configuration (free, no billing)
# ----------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"  # current free Gemini model on OpenRouter

def is_valid_openrouter_model(model):
    """Check if model looks like a valid OpenRouter model ID."""
    valid_prefixes = ("google/", "openai/", "meta-llama/", "anthropic/", "microsoft/", "cohere/", "mistralai/", "deepseek/")
    return model.startswith(valid_prefixes)

def ask_openrouter(prompt, model=DEFAULT_MODEL, use_cache=True):
    if not OPENROUTER_API_KEY:
        return "Error: OpenRouter API key not set. Please set OPENROUTER_API_KEY environment variable.", None

    if not is_valid_openrouter_model(model):
        print(f"Warning: Invalid model '{model}' -> using default {DEFAULT_MODEL}")
        model = DEFAULT_MODEL

    if use_cache:
        cache_key = hashlib.md5((model + prompt).encode()).hexdigest()
        if cache_key in _openrouter_cache:
            return _openrouter_cache[cache_key], None
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        resp = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"]
            if use_cache:
                if len(_openrouter_cache) >= 50:
                    _openrouter_cache.pop(next(iter(_openrouter_cache)))
                _openrouter_cache[cache_key] = reply
            token_info = {"duration": 0, "total_tokens": 0}
            return reply, token_info
        else:
            return f"OpenRouter error: {resp.status_code} - {resp.text}", None
    except Exception as e:
        return f"OpenRouter error: {str(e)}", None

def ask_openrouter_stream(prompt, model=DEFAULT_MODEL):
    if not OPENROUTER_API_KEY:
        yield f"data: {json.dumps({'error': 'OpenRouter API key not set'})}\n\n"
        return

    if not is_valid_openrouter_model(model):
        print(f"Warning: Invalid model '{model}' -> using default {DEFAULT_MODEL}")
        model = DEFAULT_MODEL

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        with requests.post(OPENROUTER_URL, headers=headers, json=data, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'OpenRouter error {resp.status_code}'})}\n\n"
                return
            for line in resp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        line = line[6:]
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield f"data: {json.dumps({'token': delta['content']})}\n\n"
                        except:
                            continue
            yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

_openrouter_cache = {}

# ----------------------------------------------------------------------
# ChromaDB persistent memory
# ----------------------------------------------------------------------
CHROMA_PATH = os.path.join(USER_HOME, ".qwen_studio_memory")
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
memory_collection = chroma_client.get_or_create_collection(
    name="studio_memory",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

def store_memory(user_input, action, output, metadata=None):
    doc_id = "{}_{}".format(int(time.time()), hashlib.md5(user_input.encode()).hexdigest()[:8])
    doc = "User: {}\nAction: {}\nOutput: {}".format(user_input, action, output[:500])
    meta = {"user_input": user_input, "action": action, "timestamp": time.time()}
    if metadata:
        meta.update(metadata)
    memory_collection.upsert(documents=[doc], metadatas=[meta], ids=[doc_id])

def recall_memory(query, n_results=3):
    results = memory_collection.query(query_texts=[query], n_results=n_results)
    if results and results['documents'] and results['documents'][0]:
        return results['documents'][0]
    return []

# ----------------------------------------------------------------------
# Plugin system (unchanged)
# ----------------------------------------------------------------------
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
                else:
                    print(f"[Plugin] Skipped {filename}: missing run() function")
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

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def clean_html(raw):
    raw = re.sub(r'^```[^\n]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    raw = raw.replace('```', '')
    raw = re.sub(r'`html`', '', raw)
    return raw.strip()

def save_file(content, filename, directory="Desktop"):
    if not os.path.exists("data"):
        os.makedirs("data")
    path = os.path.join("data", filename)
    if isinstance(content, bytes):
        with open(path, "wb") as f:
            f.write(content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return path

# ----------------------------------------------------------------------
# Core tools (using OpenRouter)
# ----------------------------------------------------------------------
def generate_website(task, filename, style_guide, model=DEFAULT_MODEL):
    memories = recall_memory("website " + task, n_results=2)
    memory_context = "\n".join(memories) if memories else ""
    prompt = f"""You are an expert front-end developer. The user asks: {task}
Follow this style guide: {style_guide}
Similar past successful examples:
{memory_context}
Create a complete, standalone HTML page that implements a fully functional, beautiful website.
Requirements: semantic HTML5, modern dark theme with neon accents, rounded corners, CSS Grid/Flexbox, vanilla JavaScript.
Output ONLY the raw HTML code, starting with <!DOCTYPE html>. No triple backticks."""
    response, token_info = ask_openrouter(prompt, model=model)
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
        return result, token_info
    return {"error": "Could not extract valid HTML", "raw": response}, token_info

def refine_html(original_html, instruction, model=DEFAULT_MODEL):
    refine_prompt = f"""You are an expert front-end developer. Here is an HTML document:
{original_html}
The user wants: {instruction}
Output the **complete** refined HTML code, starting with <!DOCTYPE html>. No triple backticks."""
    response, token_info = ask_openrouter(refine_prompt, model=model)
    clean_refined = clean_html(response)
    html_match = re.search(r'<!DOCTYPE\s+html[^>]*>.*?</html>', clean_refined, re.DOTALL | re.IGNORECASE)
    if not html_match:
        html_match = re.search(r'<html[^>]*>.*?</html>', clean_refined, re.DOTALL | re.IGNORECASE)
    if html_match:
        new_html = html_match.group(0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"refined_{timestamp}.html"
        saved_path = save_file(new_html, filename)
        result = {"new_html": new_html, "path": saved_path, "error": None}
        store_memory(instruction, "html_refinement", result.get("new_html", ""))
        return result, token_info
    return {"error": "OpenRouter did not return valid HTML", "raw": refined}, token_info

def convert_code_to_html(code, model=DEFAULT_MODEL):
    prompt = f"""You are a helpful assistant. The user provided code:
```{code}```
Create a **complete, standalone HTML page** that displays this code nicely (syntax-highlighted) and explains what it does. Output ONLY raw HTML starting with <!DOCTYPE html>. No triple backticks."""
    response, token_info = ask_openrouter(prompt, model=model)
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
        return result, token_info
    return {"error": "Failed to generate valid HTML", "raw": response}, token_info

def refine_code(code, instruction, model=DEFAULT_MODEL):
    prompt = f"""You are an expert programmer. The user provided code:
```{code}```
The user wants: {instruction}
Output ONLY the refined code, no explanations, no markdown, no triple backticks."""
    response, token_info = ask_openrouter(prompt, model=model)
    refined = re.sub(r'^```[^\n]*\n?', '', response)
    refined = re.sub(r'\n?```$', '', refined)
    result = {"refined_code": refined, "error": None}
    store_memory(instruction, "code_refinement", result.get("refined_code", ""))
    return result, token_info

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

def generate_app(description, language, filename_base, model=DEFAULT_MODEL):
    lang_ext = {"python": ".py", "powershell": ".ps1", "bash": ".sh", "batch": ".bat"}
    ext = lang_ext.get(language, ".txt")
    prompt = f"You are a senior software engineer. The user wants: {description}\nGenerate complete, ready-to-run code in {language}. Output ONLY the raw code, no backticks."
    code, token_info = ask_openrouter(prompt, model=model)
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
    return result, token_info

def clone_website(url, output_dir, progress_callback=None):
    clone_root = os.path.join("data", output_dir)
    os.makedirs(clone_root, exist_ok=True)
    url_to_local = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        if progress_callback:
            progress_callback("Loading page...")
        page.goto(url, wait_until="networkidle", timeout=180000)
        page.wait_for_timeout(2000)
        html = page.content()
        if progress_callback:
            progress_callback("Page loaded, collecting assets...")
        asset_urls = page.evaluate('''() => {
            const urls = new Set();
            document.querySelectorAll('link[rel="stylesheet"]').forEach(link => { if (link.href) urls.add(link.href); });
            document.querySelectorAll('script[src]').forEach(script => { if (script.src) urls.add(script.src); });
            document.querySelectorAll('img[src]').forEach(img => { if (img.src) urls.add(img.src); });
            document.querySelectorAll('link[rel="icon"]').forEach(icon => { if (icon.href) urls.add(icon.href); });
            document.querySelectorAll('link[rel="preload"]').forEach(link => { if (link.href) urls.add(link.href); });
            return Array.from(urls);
        }''')
        total = len(asset_urls)
        for idx, asset_url in enumerate(asset_urls):
            if progress_callback:
                progress_callback(f"Downloading asset {idx+1}/{total}: {os.path.basename(asset_url)[:40]}...")
            parsed = urlparse(asset_url)
            full_url = asset_url if parsed.netloc else urljoin(url, asset_url)
            path = parsed.path if parsed.path else "/"
            if path.startswith("/"):
                path = path[1:]
            if not path or path.endswith('/'):
                path = path + "index.html"
            local_path = os.path.join(clone_root, path)
            try:
                response = requests.get(full_url, timeout=30)
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    content_type = response.headers.get('content-type', '')
                    if 'text/' in content_type or content_type == 'application/javascript' or content_type == 'text/css':
                        with open(local_path, "w", encoding="utf-8") as f:
                            f.write(response.text)
                    else:
                        with open(local_path, "wb") as f:
                            f.write(response.content)
                    rel_path = os.path.relpath(local_path, clone_root).replace('\\', '/')
                    url_to_local[asset_url] = rel_path
            except Exception as e:
                print(f"Failed to download {full_url}: {e}")
        if progress_callback:
            progress_callback("Rewriting asset paths...")
        for orig_url, local_rel in url_to_local.items():
            escaped = re.escape(orig_url)
            html = re.sub(escaped, local_rel, html)
        index_path = os.path.join(clone_root, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html)
        browser.close()
    store_memory(url, "website_clone", f"Cloned to {clone_root}")
    return index_path, clone_root

# ----------------------------------------------------------------------
# Flask routes (using OpenRouter)
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": ["google/gemini-2.0-flash-001", "google/gemini-2.5-flash-preview", "mistralai/mistral-7b-instruct"]})

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
    model = data.get("model", DEFAULT_MODEL)
    res, token_info = generate_website(task, filename, style_guide, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"html": res["html"], "path": res["path"], "token_info": token_info})

@app.route("/refine", methods=["POST"])
def route_refine():
    data = request.get_json()
    original_html = data.get("html", "")
    instruction = data.get("instruction", "")
    model = data.get("model", DEFAULT_MODEL)
    res, token_info = refine_html(original_html, instruction, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"new_html": res["new_html"], "path": res["path"], "token_info": token_info})

@app.route("/convert_code", methods=["POST"])
def route_convert_code():
    data = request.get_json()
    code = data.get("code", "")
    model = data.get("model", DEFAULT_MODEL)
    res, token_info = convert_code_to_html(code, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"html": res["html"], "path": res["path"], "token_info": token_info})

@app.route("/refine_code", methods=["POST"])
def route_refine_code():
    data = request.get_json()
    code = data.get("code", "")
    instruction = data.get("instruction", "")
    model = data.get("model", DEFAULT_MODEL)
    res, token_info = refine_code(code, instruction, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"refined_code": res["refined_code"], "token_info": token_info})

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
    model = data.get("model", DEFAULT_MODEL)
    res, token_info = generate_app(description, language, filename, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"code": res["code"], "path": res["path"], "token_info": token_info})

@app.route("/chat", methods=["POST"])
def route_chat():
    data = request.get_json()
    message = data.get("message", "")
    model = data.get("model", DEFAULT_MODEL)
    relevant = recall_memory(message, n_results=5)
    memory_context = ""
    if relevant:
        memory_context = "Relevant past memories:\n" + "\n".join(relevant) + "\n\n"
    name_match = re.search(r"(my name is|call me|i am) (\w+)", message, re.IGNORECASE)
    if name_match:
        user_name = name_match.group(2)
        store_memory(message, "personal_info", f"User's name is {user_name}")
        memory_context += f"IMPORTANT: The user's name is {user_name}. Always address them by name.\n"
    if re.search(r"(i like|i prefer|my favorite|i love)", message, re.IGNORECASE):
        store_memory(message, "preference", message)
    if 'conv_history' not in session:
        session['conv_history'] = []
    history = session['conv_history']
    prompt = "You are a helpful AI assistant that remembers past conversations.\n"
    prompt += memory_context
    for msg in history[-10:]:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"User: {message}\nAssistant:"
    reply, token_info = ask_openrouter(prompt, model=model, use_cache=False)
    store_memory(message, "chat_interaction", reply)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    session['conv_history'] = history[-20:]
    return jsonify({"reply": reply, "token_info": token_info})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop('conv_history', None)
    return jsonify({"status": "cleared"})

@app.route("/clone", methods=["POST"])
def route_clone():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"})
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"cloned_site_{timestamp}"
    try:
        index_path, clone_dir = clone_website(url, output_dir)
        return jsonify({"path": index_path, "directory": clone_dir, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/preview/<clone_folder>/<path:filename>")
def serve_clone(clone_folder, filename):
    clone_path = os.path.join("data", clone_folder)
    if not os.path.exists(clone_path):
        return "Clone folder not found", 404
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    return send_from_directory(clone_path, filename)

@app.route("/open_file_cleaner", methods=["POST"])
def open_file_cleaner():
    return jsonify({"error": "File cleaner not available in cloud version"})

@app.route("/backup_memory", methods=["POST"])
def backup_memory():
    return jsonify({"error": "Backup not available in cloud version"}), 501

@app.route("/restore_memory", methods=["POST"])
def restore_memory():
    return jsonify({"error": "Restore not available in cloud version"}), 501

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
        return jsonify({"error": "No ID provided"}), 400
    try:
        memory_collection.delete(ids=[mem_id])
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compress_memory", methods=["POST"])
def compress_memory():
    data = request.get_json() or {}
    days_old = data.get("days_old", 30)
    max_memories = data.get("max_memories", 20)
    try:
        results = memory_collection.get(limit=1000)
        if not results or not results['ids']:
            return jsonify({"error": "No memories found"}), 400
        memories = []
        for i, doc_id in enumerate(results['ids']):
            doc = results['documents'][i]
            meta = results['metadatas'][i] if results['metadatas'] else {}
            timestamp = meta.get("timestamp", 0)
            memories.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "timestamp": timestamp
            })
        now = time.time()
        old_memories = [m for m in memories if (now - m["timestamp"]) > days_old * 86400]
        if not old_memories:
            return jsonify({"message": f"No memories older than {days_old} days found."})
        old_memories.sort(key=lambda x: x["timestamp"])
        to_compress = old_memories[:max_memories]
        docs_text = "\n\n---\n\n".join([f"Memory {i+1}:\n{m['document']}" for i, m in enumerate(to_compress)])
        summary_prompt = f"""You are a memory compression assistant. Below are several past memory entries (user requests and AI actions). Please summarise the key facts, preferences, and important information from these memories into a short paragraph (max 300 characters). Focus on what the user likes, their name, important tasks, and recurring themes.

Memories:
{docs_text}

Summary:"""
        summary, _ = ask_openrouter(summary_prompt, model=DEFAULT_MODEL, use_cache=False)
        summary = summary.strip()[:500]
        store_memory("memory_compression", "memory_summary", summary, metadata={"compressed": True, "original_count": len(to_compress)})
        ids_to_delete = [m["id"] for m in to_compress]
        memory_collection.delete(ids=ids_to_delete)
        return jsonify({
            "message": f"Compressed {len(to_compress)} old memories into a summary.",
            "summary": summary,
            "deleted_count": len(to_compress)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------------
# Streaming smart agent (using OpenRouter)
# ----------------------------------------------------------------------
@app.route("/stream_smart_agent", methods=["GET"])
def stream_smart_agent():
    request_text = request.args.get("request", "")
    style_guide = request.args.get("styleGuide", "")
    model = request.args.get("model", DEFAULT_MODEL)
    if not request_text:
        return Response("data: {}\n\n".format(json.dumps({"error": "No request"})), mimetype="text/event-stream")
    def generate():
        # Forced email rule
        email_keywords = ["send an email", "send email", "email to", "mail to", "send a message to"]
        if any(kw in request_text.lower() for kw in email_keywords):
            args = {"to": "recipient@example.com"}
            email_match = re.search(r'[\w\.-]+@[\w\.-]+', request_text)
            if email_match:
                args["to"] = email_match.group(0)
            yield f"data: {json.dumps({'step_start': 'plugin_email_sender', 'step_index': 0})}\n\n"
            res = run_plugin("email_sender", args)
            if "error" in res:
                yield f"data: {json.dumps({'error': res['error']})}\n\n"
            else:
                result_text = res.get('result', '')
                yield f"data: {json.dumps({'result': result_text, 'tool': 'plugin_email_sender'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Forced screenshot rule
        screenshot_keywords = ["screenshot", "take a screenshot", "capture screenshot", "screen capture", "snap a picture"]
        if any(kw in request_text.lower() for kw in screenshot_keywords):
            url_match = re.search(r'(https?://[^\s]+)', request_text)
            if not url_match:
                yield f"data: {json.dumps({'error': 'No URL found in request for screenshot'})}\n\n"
                return
            url = url_match.group(0)
            args = {"url": url, "action": "screenshot"}
            res = run_plugin("agent_browser", args)
            if "error" in res:
                yield f"data: {json.dumps({'error': res['error']})}\n\n"
            else:
                result_text = res.get('result', '')
                yield f"data: {json.dumps({'result': result_text, 'tool': 'plugin_agent_browser'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        memories = recall_memory(request_text, n_results=5)
        memory_context = "\n".join(memories) if memories else "No similar past tasks."
        plugins_info = get_plugins_info()
        plugins_text = "\n".join([f"- plugin_{name}: {info['description']} (args: any JSON object)" for name, info in plugins_info.items()]) if plugins_info else "No plugins available."
        decision_prompt = f"""You are a smart assistant that can plan a sequence of tools to fulfill a complex request.
IMPORTANT: Use the memory context below to personalize your response.

Memory (past similar tasks / user info):
{memory_context}

Available built-in tools:
- website: generate a website. Args: task (string), filename (optional).
- refine_html: improve existing HTML. Args: html (full HTML), instruction.
- code_to_html: convert code to HTML page. Args: code.
- refine_code: modify code. Args: code, instruction.
- search: web search. Args: query.
- app_gen: generate script. Args: description, language (python|powershell|bash|batch), filename (optional).
- clone: clone a website. Args: url (string).
- chat: just talk. Args: message.

Available plugins (custom tools):
{plugins_text}

Now respond with JSON only. No extra text.

User request: "{request_text}"
Style guide: {style_guide}
"""
        decision_text, _ = ask_openrouter(decision_prompt, model=model, use_cache=False)
        try:
            decision = json.loads(decision_text)
        except:
            yield f"data: {json.dumps({'error': f'Routing failed: {decision_text}'})}\n\n"
            return
        if isinstance(decision, dict):
            decisions = [decision]
        else:
            decisions = decision
        step_idx = 0
        while step_idx < len(decisions):
            action = decisions[step_idx]
            tool = action.get("tool")
            args = action.get("arguments", {})
            yield f"data: {json.dumps({'step_start': tool, 'step_index': step_idx})}\n\n"
            error_occurred = False
            try:
                if tool == "chat":
                    message = args.get("message", request_text)
                    mem = recall_memory(message, n_results=3)
                    mem_text = "\n".join(mem) if mem else ""
                    prompt = f"Memory:\n{mem_text}\n\nUser: {message}\nAssistant:"
                    for chunk in ask_openrouter_stream(prompt, model=model):
                        yield chunk
                elif tool == "search":
                    query = args.get("query", request_text)
                    res, _ = web_search(query)
                    if res.get("error"):
                        error_occurred = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        yield f"data: {json.dumps({'result': res['results'], 'tool': 'search'})}\n\n"
                        store_memory(request_text, tool, res.get("results", ""))
                elif tool == "clone":
                    url = args.get("url", request_text)
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_dir = f"cloned_site_{timestamp}"
                    yield f"data: {json.dumps({'result': '🌐 Starting clone...', 'tool': 'clone'})}\n\n"
                    try:
                        yield f"data: {json.dumps({'result': '📄 Loading webpage...', 'tool': 'clone'})}\n\n"
                        index_path, clone_dir = clone_website(url, output_dir)
                        store_memory(request_text, tool, f"Cloned to {clone_dir}")
                        yield f"data: {json.dumps({'result': f'✅ Clone completed! Saved to {clone_dir}', 'path': index_path, 'directory': clone_dir, 'tool': 'clone'})}\n\n"
                        yield f"data: {json.dumps({'clone_preview': clone_dir})}\n\n"
                    except Exception as e:
                        error_occurred = True
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                elif tool.startswith("plugin_"):
                    plugin_name = tool[7:]
                    if plugin_name not in _plugins:
                        error_occurred = True
                        yield f"data: {json.dumps({'error': f'Plugin {plugin_name} not found'})}\n\n"
                    else:
                        res = run_plugin(plugin_name, args)
                        if "error" in res:
                            error_occurred = True
                            yield f"data: {json.dumps({'error': res['error']})}\n\n"
                        else:
                            result_text = res.get('result', '')
                            yield f"data: {json.dumps({'result': result_text, 'tool': tool})}\n\n"
                            store_memory(request_text, tool, str(result_text))
                else:
                    if tool == "website":
                        task = args.get("task", request_text)
                        filename = args.get("filename", "website")
                        res, token_info = generate_website(task, filename, style_guide, model)
                        if res.get("error"):
                            error_occurred = True
                            yield f"data: {json.dumps({'error': res['error'], 'raw': res.get('raw')})}\n\n"
                        else:
                            result_str = f"Website saved to {res['path']}"
                            yield f"data: {json.dumps({'result': result_str, 'tool': 'website', 'html_preview': res['html'][:500], 'path': res['path'], 'token_info': token_info})}\n\n"
                            store_memory(request_text, tool, res.get("html", ""))
                    elif tool == "refine_html":
                        html = args.get("html", "")
                        instruction = args.get("instruction", "")
                        res, token_info = refine_html(html, instruction, model)
                        if res.get("error"):
                            error_occurred = True
                            yield f"data: {json.dumps({'error': res['error']})}\n\n"
                        else:
                            result_str = f"Refined HTML saved to {res['path']}"
                            yield f"data: {json.dumps({'result': result_str, 'tool': 'refine_html', 'new_html_preview': res['new_html'][:500], 'path': res['path'], 'token_info': token_info})}\n\n"
                            store_memory(request_text, tool, res.get("new_html", ""))
                    elif tool == "code_to_html":
                        code = args.get("code", "")
                        res, token_info = convert_code_to_html(code, model)
                        if res.get("error"):
                            error_occurred = True
                            yield f"data: {json.dumps({'error': res['error']})}\n\n"
                        else:
                            result_str = f"HTML saved to {res['path']}"
                            yield f"data: {json.dumps({'result': result_str, 'tool': 'code_to_html', 'html_preview': res['html'][:500], 'path': res['path'], 'token_info': token_info})}\n\n"
                            store_memory(request_text, tool, res.get("html", ""))
                    elif tool == "refine_code":
                        code = args.get("code", "")
                        instruction = args.get("instruction", "")
                        res, token_info = refine_code(code, instruction, model)
                        if res.get("error"):
                            error_occurred = True
                            yield f"data: {json.dumps({'error': res['error']})}\n\n"
                        else:
                            yield f"data: {json.dumps({'result': 'Code refined', 'tool': 'refine_code', 'refined_code': res['refined_code'], 'token_info': token_info})}\n\n"
                            store_memory(request_text, tool, res.get("refined_code", ""))
                    elif tool == "app_gen":
                        description = args.get("description", request_text)
                        language = args.get("language", "python")
                        filename = args.get("filename", "")
                        res, token_info = generate_app(description, language, filename, model)
                        if res.get("error"):
                            error_occurred = True
                            yield f"data: {json.dumps({'error': res['error']})}\n\n"
                        else:
                            result_str = f"App saved to {res['path']}"
                            yield f"data: {json.dumps({'result': result_str, 'tool': 'app_gen', 'code_preview': res['code'][:500], 'path': res['path'], 'token_info': token_info})}\n\n"
                            store_memory(request_text, tool, res.get("code", ""))
                    else:
                        error_occurred = True
                        yield f"data: {json.dumps({'error': f'Unknown tool: {tool}'})}\n\n"
            except Exception as e:
                error_occurred = True
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            if error_occurred:
                break
            step_idx += 1
        yield f"data: {json.dumps({'done': True})}\n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route('/favicon.ico')
def favicon():
    return '', 204

# ----------------------------------------------------------------------
# Gemini image endpoint (disabled in cloud)
# ----------------------------------------------------------------------
@app.route("/gemini_image", methods=["POST"])
def route_gemini_image():
    return jsonify({"error": "Image generation temporarily disabled in cloud version. Use local version for images."})

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
            if user_input and len(user_input) > 2:
                suggestions.add(user_input[:100])
    plugins = get_plugins_info()
    for pname, info in plugins.items():
        suggestions.add(info["name"])
        suggestions.add(f"Run plugin {info['name']}")
    tool_commands = [
        "generate website", "refine html", "convert code to html",
        "refine code", "search web", "generate app", "clone website",
        "send email", "read emails", "organize files", "check weather",
        "review code", "make presentation", "create video"
    ]
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
    gpu_info = {}
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
            gpu_info = {
                "gpu_percent": int(gpu_util),
                "gpu_mem_used_gb": float(mem_used) / 1024,
                "gpu_mem_total_gb": float(mem_total) / 1024
            }
    except:
        pass
    return jsonify({
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "ram_used_gb": round(ram_used_gb, 1),
        "ram_total_gb": round(ram_total_gb, 1),
        "gpu": gpu_info
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting Qwen AI Studio (cloud version with OpenRouter) on port {port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)