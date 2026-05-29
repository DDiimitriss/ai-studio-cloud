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
import base64
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, send_from_directory
from urllib.parse import urljoin, urlparse
from duckduckgo_search import DDGS
import psutil

app = Flask(__name__)
app.secret_key = os.urandom(24)

PLAYWRIGHT_AVAILABLE = False

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TEXT_MODEL = "openai/gpt-3.5-turbo"
IMAGE_MODEL = "google/gemini-2.0-flash-exp-image-generation"   # ✅ working free model

def ask_openrouter(prompt, model=None):
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not set.", None
    if model is None:
        model = DEFAULT_TEXT_MODEL
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"], None
        else:
            return f"OpenRouter error: {resp.status_code}", None
    except Exception as e:
        return f"Error: {str(e)}", None

def generate_image_with_openrouter(prompt, image_base64=None):
    if not OPENROUTER_API_KEY:
        return None, "No API key"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "google/gemini-2.0-flash-exp-image-generation",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        message = data["choices"][0]["message"]["content"]
        url_match = re.search(r'(https?://[^\s]+\.(png|jpg|jpeg|gif|webp))', message, re.IGNORECASE)
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
        if "data:image" in message:
            img_data = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', message)
            if img_data:
                os.makedirs("data", exist_ok=True)
                fname = f"generated_image_{int(time.time())}.png"
                path = os.path.join("data", fname)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(img_data.group(1)))
                return path, None
        return None, f"No image URL or base64 found: {message[:200]}"
    except Exception as e:
        return None, str(e)

# ----- session memory (no ChromaDB) -----
def store_memory(user_input, action, output):
    if 'memories' not in session:
        session['memories'] = []
    session['memories'].append({
        "input": user_input,
        "action": action,
        "output": output[:500],
        "timestamp": time.time()
    })
    if len(session['memories']) > 50:
        session['memories'] = session['memories'][-50:]

def recall_memory(query, n_results=3):
    if 'memories' not in session:
        return []
    results = []
    query_lower = query.lower()
    for mem in reversed(session['memories']):
        if query_lower in mem['input'].lower() or any(word in mem['input'].lower() for word in query_lower.split()):
            results.append(mem['output'])
            if len(results) >= n_results:
                break
    return results

# ----- plugins (skip missing) -----
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

def generate_website(task, filename, style_guide, model=None):
    memories = recall_memory("website " + task, n_results=2)
    memory_context = "\n".join(memories) if memories else ""
    prompt = f"Create a complete HTML page. Task: {task}\nStyle: {style_guide}\nMemory: {memory_context}\nOutput ONLY raw HTML starting with <!DOCTYPE html>."
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
        store_memory(task, "website_generation", html_content[:200])
        return {"html": html_content, "path": saved_path}, None
    return {"error": "Could not extract valid HTML", "raw": response}, None

def web_search(query, model=None):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            formatted = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
            store_memory(query, "web_search", formatted[:300])
            return {"results": formatted}, None
    except Exception as e:
        return {"error": str(e)}, None

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route("/chat", methods=["POST"])
def route_chat():
    data = request.get_json()
    message = data.get("message", "")
    model = data.get("model", DEFAULT_TEXT_MODEL)
    
    image_words = ["draw", "generate image", "create an image", "picture of", "image of", "make a picture", "draw a", "paint"]
    if any(word in message.lower() for word in image_words):
        print(f"[IMAGE] Generating image for: {message}")
        save_path, error = generate_image_with_openrouter(message, None)
        if error:
            return jsonify({"reply": f"❌ Image failed: {error}"})
        else:
            image_url = f"/data/{os.path.basename(save_path)}"
            return jsonify({"reply": f"🖼️ Here is your image: [Click to view]({image_url})"})
    
    relevant = recall_memory(message, n_results=5)
    memory_context = ""
    if relevant:
        memory_context = "Relevant past memories:\n" + "\n".join(relevant) + "\n\n"
    if 'conv_history' not in session:
        session['conv_history'] = []
    history = session['conv_history']
    prompt = "You are a helpful AI assistant.\n" + memory_context
    for msg in history[-10:]:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"User: {message}\nAssistant:"
    reply, _ = ask_openrouter(prompt, model=model)
    store_memory(message, "chat_interaction", reply)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    session['conv_history'] = history[-20:]
    return jsonify({"reply": reply})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop('conv_history', None)
    session.pop('memories', None)
    return jsonify({"status": "cleared"})

@app.route("/search", methods=["POST"])
def route_search():
    data = request.get_json()
    query = data.get("query", "")
    res, _ = web_search(query)
    if res.get("error"):
        return jsonify({"error": res["error"]})
    return jsonify({"results": res["results"]})

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

@app.route("/list_plugins", methods=["GET"])
def list_plugins_route():
    return jsonify({"plugins": get_plugins_info()})

@app.route("/run_plugin", methods=["POST"])
def run_plugin_route():
    data = request.get_json()
    plugin_name = data.get("plugin")
    args = data.get("args", {})
    result = run_plugin(plugin_name, args)
    return jsonify(result)

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
        image_words = ["draw", "generate image", "create an image", "picture of", "image of", "make a picture", "draw a"]
        if any(word in request_text.lower() for word in image_words):
            save_path, error = generate_image_with_openrouter(request_text, None)
            if error:
                yield f"data: {json.dumps({'error': f'Image failed: {error}'})}\n\n"
            else:
                yield f"data: {json.dumps({'result': f'🖼️ Image saved to {save_path}', 'path': save_path, 'tool': 'image_generation'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
        memories = recall_memory(request_text, n_results=3)
        memory_context = "\n".join(memories) if memories else ""
        prompt = f"Memory:\n{memory_context}\n\nUser: {request_text}\nAssistant:"
        reply, _ = ask_openrouter(prompt, model=model)
        yield f"data: {json.dumps({'token': reply})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        store_memory(request_text, "smart_agent", reply)
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 AI Studio starting on port {port}")
    print(f"✅ Image generation enabled with {IMAGE_MODEL}")
    app.run(debug=False, host="0.0.0.0", port=port)