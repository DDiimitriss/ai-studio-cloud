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
import base64
import tempfile
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, send_from_directory
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
from playwright.sync_api import sync_playwright
import psutil

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "studio-secret-change-me-in-production")

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
# OpenRouter API configuration
# ----------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Text model: auto‑router to avoid rate limits
DEFAULT_TEXT_MODEL = "openrouter/free"

# Image generation models
IMAGE_MODEL        = "google/gemini-2.5-flash-image-preview:free"   # free tier
IMAGE_MODEL_PAID   = "google/gemini-2.5-flash-image"                # paid fallback

# Whitelist of valid models
VALID_MODELS = {
    "openrouter/free",
    "qwen/qwen3-coder:free",
    "qwen/qwen3.6-plus:free",
    "deepseek/deepseek-v3:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "stepfun/step-3.5-flash:free",
    "qwen/qwen3.6-plus",
    "qwen/qwen3.6-flash",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-flash",
    "stepfun/step-3.5-flash",
    "google/gemini-2.5-flash-image-preview:free",
    "google/gemini-2.5-flash-image",
}

_openrouter_cache = {}

def sanitize_model(requested_model):
    if requested_model in VALID_MODELS:
        return requested_model
    print(f"[sanitize_model] Unknown model '{requested_model}', falling back to {DEFAULT_TEXT_MODEL}")
    return DEFAULT_TEXT_MODEL

def strip_thinking_tags(text):
    """Remove <think>...</think> blocks from reasoning model output."""
    if not isinstance(text, str):
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# ----------------------------------------------------------------------
# Core OpenRouter chat functions
# ----------------------------------------------------------------------
def ask_openrouter(prompt, model=DEFAULT_TEXT_MODEL, use_cache=True):
    model = sanitize_model(model)
    if not OPENROUTER_API_KEY:
        return "Error: OpenRouter API key not set.", None

    cache_key = hashlib.md5((model + prompt).encode()).hexdigest()
    if use_cache and cache_key in _openrouter_cache:
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
            resp_data = resp.json()
            raw_reply = resp_data["choices"][0]["message"]["content"]
            reply = strip_thinking_tags(raw_reply)
            usage = resp_data.get("usage", {})
            token_info = {
                "total_tokens":      usage.get("total_tokens", 0),
                "prompt_tokens":     usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
            if use_cache:
                if len(_openrouter_cache) >= 50:
                    _openrouter_cache.pop(next(iter(_openrouter_cache)))
                _openrouter_cache[cache_key] = reply
            return reply, token_info
        else:
            return f"OpenRouter error: {resp.status_code} - {resp.text}", None
    except Exception as e:
        return f"OpenRouter error: {str(e)}", None

def ask_openrouter_stream(prompt, model=DEFAULT_TEXT_MODEL):
    model = sanitize_model(model)
    if not OPENROUTER_API_KEY:
        yield f"data: {json.dumps({'error': 'OpenRouter API key not set'})}\n\n"
        return
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
        in_think   = False
        think_buf  = ""
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
                                token = delta.get("content", "")
                                if not token:
                                    continue
                                if not in_think:
                                    if '<think>' in token:
                                        parts = token.split('<think>', 1)
                                        if parts[0]:
                                            yield f"data: {json.dumps({'token': parts[0]})}\n\n"
                                        in_think  = True
                                        think_buf = parts[1] if len(parts) > 1 else ""
                                    else:
                                        yield f"data: {json.dumps({'token': token})}\n\n"
                                else:
                                    think_buf += token
                                    if '</think>' in think_buf:
                                        after = think_buf.split('</think>', 1)[1]
                                        in_think  = False
                                        think_buf = ""
                                        if after:
                                            yield f"data: {json.dumps({'token': after})}\n\n"
                        except Exception:
                            continue
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ----------------------------------------------------------------------
# Image generation
# ----------------------------------------------------------------------
def generate_image_with_openrouter(prompt, image_base64=None):
    if not OPENROUTER_API_KEY:
        return None, "OpenRouter API key not set."

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

    for model_id in [IMAGE_MODEL, IMAGE_MODEL_PAID]:
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": content_parts}],
        }
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=120)
            if resp.status_code in (402, 404, 400):
                print(f"[image_gen] Model {model_id} returned {resp.status_code}, trying fallback.")
                continue
            if resp.status_code != 200:
                return None, f"OpenRouter error: {resp.status_code} - {resp.text}"

            resp_json = resp.json()
            message   = resp_json["choices"][0]["message"]
            content   = message.get("content", "")

            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "image_url":
                        url_data = part.get("image_url", {}).get("url", "")
                        if url_data.startswith("data:image"):
                            header_part, b64 = url_data.split(",", 1)
                            ext = header_part.split("/")[1].split(";")[0]
                            return _save_b64_image(b64, ext), None
                        elif url_data.startswith("http"):
                            return _download_and_save_image(url_data), None
                    if ptype == "text":
                        path = _extract_image_from_text(part.get("text", ""))
                        if path:
                            return path, None
                return None, "No image found in multimodal response."

            content_str = str(content)
            path = _extract_image_from_text(content_str)
            if path:
                return path, None
            return None, f"No image found in response: {content_str[:200]}..."

        except Exception as e:
            return None, str(e)

    return None, "All image generation models failed or are unavailable."

def _extract_image_from_text(text):
    m = re.search(r'!\[.*?\]\((https?://[^\s\)]+)\)', text)
    if m:
        return _download_and_save_image(m.group(1))
    m = re.search(r'(https?://[^\s]+\.(png|jpg|jpeg|gif|webp))', text, re.IGNORECASE)
    if m:
        return _download_and_save_image(m.group(1))
    m = re.search(r'data:image/([^;]+);base64,([A-Za-z0-9+/=]+)', text)
    if m:
        ext, b64 = m.group(1), m.group(2)
        return _save_b64_image(b64, ext)
    return None

def _save_b64_image(b64_string, ext="png"):
    try:
        image_data = base64.b64decode(b64_string)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"generated_{timestamp}.{ext}"
        filepath   = os.path.join("static", "generated", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(image_data)
        return filepath
    except Exception as e:
        print(f"[_save_b64_image] Failed: {e}")
        return None

def _download_and_save_image(url):
    try:
        img_resp = requests.get(url, timeout=30)
        if img_resp.status_code == 200:
            ext = url.split('.')[-1].split('?')[0].lower()
            if ext not in ('png', 'jpg', 'jpeg', 'gif', 'webp'):
                ext = 'png'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"generated_{timestamp}.{ext}"
            filepath  = os.path.join("static", "generated", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(img_resp.content)
            return filepath
        return None
    except Exception as e:
        print(f"[_download_and_save_image] Failed: {e}")
        return None

# ----------------------------------------------------------------------
# ChromaDB memory
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
# Plugin system
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
# Core tools
# ----------------------------------------------------------------------
def generate_website(task, filename, style_guide, model=DEFAULT_TEXT_MODEL):
    model = sanitize_model(model)
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

def refine_html(original_html, instruction, model=DEFAULT_TEXT_MODEL):
    model = sanitize_model(model)
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
    return {"error": "OpenRouter did not return valid HTML", "raw": response}, token_info

def convert_code_to_html(code, model=DEFAULT_TEXT_MODEL):
    model = sanitize_model(model)
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

def refine_code(code, instruction, model=DEFAULT_TEXT_MODEL):
    model = sanitize_model(model)
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

def web_search_ddg(query, model=None):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            formatted = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
            result = {"results": formatted, "error": None}
            store_memory(query, "web_search", result.get("results", ""))
            return result, None
    except Exception as e:
        return {"error": str(e)}, None

def generate_app(description, language, filename_base, model=DEFAULT_TEXT_MODEL):
    model = sanitize_model(model)
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
            document.querySelectorAll('link[rel="stylesheet"]').forEach(l => { if (l.href) urls.add(l.href); });
            document.querySelectorAll('script[src]').forEach(s => { if (s.src) urls.add(s.src); });
            document.querySelectorAll('img[src]').forEach(i => { if (i.src) urls.add(i.src); });
            document.querySelectorAll('link[rel="icon"]').forEach(i => { if (i.href) urls.add(i.href); });
            document.querySelectorAll('link[rel="preload"]').forEach(l => { if (l.href) urls.add(l.href); });
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
                    if ('text/' in content_type or content_type == 'application/javascript' or content_type == 'text/css'):
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
# Image intent detection
# ----------------------------------------------------------------------
def is_image_request(message):
    keywords = [
        "draw", "generate image", "create an image", "generate a picture",
        "make an image", "image of", "picture of", "create a picture",
        "generate a photo", "sketch", "illustrate", "visualize",
        "nano banana", "paint a", "design an image"
    ]
    return any(kw in message.lower() for kw in keywords)

# ----------------------------------------------------------------------
# Flask routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "api_key_set": bool(OPENROUTER_API_KEY),
        "default_text_model": DEFAULT_TEXT_MODEL,
        "image_model": IMAGE_MODEL,
        "image_model_paid": IMAGE_MODEL_PAID,
    })

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": sorted(VALID_MODELS)})

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
    model = sanitize_model(data.get("model", DEFAULT_TEXT_MODEL))
    res, token_info = generate_website(task, filename, style_guide, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"html": res["html"], "path": res["path"], "token_info": token_info})

@app.route("/refine", methods=["POST"])
def route_refine():
    data = request.get_json()
    original_html = data.get("html", "")
    instruction = data.get("instruction", "")
    model = sanitize_model(data.get("model", DEFAULT_TEXT_MODEL))
    res, token_info = refine_html(original_html, instruction, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"new_html": res["new_html"], "path": res["path"], "token_info": token_info})

@app.route("/convert_code", methods=["POST"])
def route_convert_code():
    data = request.get_json()
    code = data.get("code", "")
    model = sanitize_model(data.get("model", DEFAULT_TEXT_MODEL))
    res, token_info = convert_code_to_html(code, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"html": res["html"], "path": res["path"], "token_info": token_info})

@app.route("/refine_code", methods=["POST"])
def route_refine_code():
    data = request.get_json()
    code = data.get("code", "")
    instruction = data.get("instruction", "")
    model = sanitize_model(data.get("model", DEFAULT_TEXT_MODEL))
    res, token_info = refine_code(code, instruction, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"refined_code": res["refined_code"], "token_info": token_info})

@app.route("/search", methods=["POST"])
def route_search():
    data = request.get_json()
    query = data.get("query", "")
    res, _ = web_search_ddg(query)
    if res.get("error"):
        return jsonify({"error": res["error"]})
    return jsonify({"results": res["results"]})

@app.route("/generate_app", methods=["POST"])
def route_generate_app():
    data = request.get_json()
    description = data.get("description", "")
    language = data.get("language", "python")
    filename = data.get("filename", "")
    model = sanitize_model(data.get("model", DEFAULT_TEXT_MODEL))
    res, token_info = generate_app(description, language, filename, model)
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"code": res["code"], "path": res["path"], "token_info": token_info})

@app.route("/chat", methods=["POST"])
def route_chat():
    data = request.get_json()
    message = data.get("message", "")
    requested_model = sanitize_model(data.get("model", DEFAULT_TEXT_MODEL))

    if is_image_request(message):
        save_path, error = generate_image_with_openrouter(message)
        if error:
            return jsonify({"reply": f"❌ Image generation failed: {error}"})
        image_url = f"/{save_path.replace(os.sep, '/')}"
        store_memory(message, "image_generation", f"Generated image at {save_path}")
        return jsonify({"reply": f"✅ Image generated!\n![Generated Image]({image_url})"})

    model = requested_model
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
    max_mems = data.get("max_memories", 20)
    try:
        results = memory_collection.get(limit=1000)
        if not results or not results['ids']:
            return jsonify({"error": "No memories found"}), 400
        memories = []
        for i, doc_id in enumerate(results['ids']):
            doc = results['documents'][i]
            meta = results['metadatas'][i] if results['metadatas'] else {}
            timestamp = meta.get("timestamp", 0)
            memories.append({"id": doc_id, "document": doc, "metadata": meta, "timestamp": timestamp})
        now = time.time()
        old_memories = [m for m in memories if (now - m["timestamp"]) > days_old * 86400]
        if not old_memories:
            return jsonify({"message": f"No memories older than {days_old} days found."})
        old_memories.sort(key=lambda x: x["timestamp"])
        to_compress = old_memories[:max_mems]
        docs_text = "\n\n---\n\n".join([f"Memory {i+1}:\n{m['document']}" for i, m in enumerate(to_compress)])
        summary_prompt = f"""You are a memory compression assistant. Summarise the key facts, preferences, and important information from these memories into a short paragraph (max 300 characters).

Memories:
{docs_text}

Summary:"""
        summary, _ = ask_openrouter(summary_prompt, model=DEFAULT_TEXT_MODEL, use_cache=False)
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
# Streaming smart agent
# ----------------------------------------------------------------------
@app.route("/stream_smart_agent", methods=["GET"])
def stream_smart_agent():
    request_text = request.args.get("request", "")
    style_guide = request.args.get("styleGuide", "")
    model = sanitize_model(request.args.get("model", DEFAULT_TEXT_MODEL))

    if not request_text:
        return Response(f"data: {json.dumps({'error': 'No request provided'})}\n\n", mimetype="text/event-stream")

    def generate():
        # Image generation
        if is_image_request(request_text):
            save_path, error = generate_image_with_openrouter(request_text)
            if error:
                yield f"data: {json.dumps({'error': error})}\n\n"
            else:
                image_url = f"/{save_path.replace(os.sep, '/')}"
                result_msg = f"✅ Image generated!\n![Generated Image]({image_url})"
                yield f"data: {json.dumps({'result': result_msg, 'path': save_path, 'tool': 'image_generation'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Email plugin
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
                yield f"data: {json.dumps({'result': res.get('result', ''), 'tool': 'plugin_email_sender'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Screenshot plugin
        screenshot_keywords = ["screenshot", "take a screenshot", "capture screenshot", "screen capture"]
        if any(kw in request_text.lower() for kw in screenshot_keywords):
            url_match = re.search(r'(https?://[^\s]+)', request_text)
            if not url_match:
                yield f"data: {json.dumps({'error': 'No URL found for screenshot'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            res = run_plugin("agent_browser", {"url": url_match.group(0), "action": "screenshot"})
            if "error" in res:
                yield f"data: {json.dumps({'error': res['error']})}\n\n"
            else:
                yield f"data: {json.dumps({'result': res.get('result', ''), 'tool': 'plugin_agent_browser'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Tool routing
        memories = recall_memory(request_text, n_results=5)
        memory_context = "\n".join(memories) if memories else "No similar past tasks."
        plugins_info = get_plugins_info()
        plugins_text = "\n".join([f"- plugin_{name}: {info['description']} (args: any JSON object)" for name, info in plugins_info.items()]) if plugins_info else "No plugins available."

        decision_prompt = f"""You are a strict JSON-only router. Do NOT answer the user. Do NOT greet. Do NOT explain.
Your ONLY job: output a single JSON object or an array of JSON objects describing the tool(s) to use.
Each object must have "tool" and "arguments".

Available tools:
- chat: general conversation. Args: message (string).
- website: generate a website. Args: task (string), filename (optional).
- refine_html: improve HTML. Args: html (full HTML), instruction.
- code_to_html: convert code to HTML. Args: code.
- refine_code: modify code. Args: code, instruction.
- search: web search. Args: query.
- app_gen: generate script. Args: description, language (python|powershell|bash|batch), filename (optional).
- clone: clone a website. Args: url.

Plugins:
{plugins_text}

User request: "{request_text}"
Style guide: {style_guide}

Output ONLY valid JSON (no markdown, no backticks, no extra text). If unsure, use the chat tool.
"""
        decision_raw, _ = ask_openrouter(decision_prompt, model=model, use_cache=False)
        decision_text = strip_thinking_tags(decision_raw)
        decision_text = re.sub(r'^```[^\n]*\n?', '', decision_text)
        decision_text = re.sub(r'\n?```$', '', decision_text).strip()

        try:
            decision = json.loads(decision_text)
        except:
            # Fallback to chat if parsing fails
            decision = {"tool": "chat", "arguments": {"message": request_text}}

        decisions = [decision] if isinstance(decision, dict) else decision

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
                    res, _ = web_search_ddg(query)
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
                    yield f"data: {json.dumps({'result': '📄 Loading webpage...', 'tool': 'clone'})}\n\n"
                    try:
                        index_path, clone_dir = clone_website(url, output_dir)
                        store_memory(request_text, tool, f"Cloned to {clone_dir}")
                        yield f"data: {json.dumps({'result': f'✅ Clone complete! Saved to {clone_dir}', 'path': index_path, 'directory': clone_dir, 'tool': 'clone'})}\n\n"
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
                elif tool == "website":
                    task = args.get("task", request_text)
                    filename = args.get("filename", "website")
                    res, token_info = generate_website(task, filename, style_guide, model)
                    if res.get("error"):
                        error_occurred = True
                        data_obj = {"error": res["error"], "raw": res.get("raw")}
                        yield f"data: {json.dumps(data_obj)}\n\n"
                    else:
                        data_obj = {
                            "result": "Website saved to " + res["path"],
                            "tool": "website",
                            "html_preview": res["html"][:500],
                            "path": res["path"],
                            "token_info": token_info
                        }
                        yield f"data: {json.dumps(data_obj)}\n\n"
                        store_memory(request_text, tool, res.get("html", ""))
                elif tool == "refine_html":
                    html_src = args.get("html", "")
                    instruction = args.get("instruction", "")
                    res, token_info = refine_html(html_src, instruction, model)
                    if res.get("error"):
                        error_occurred = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        data_obj = {
                            "result": "Refined HTML saved to " + res["path"],
                            "tool": "refine_html",
                            "new_html_preview": res["new_html"][:500],
                            "path": res["path"],
                            "token_info": token_info
                        }
                        yield f"data: {json.dumps(data_obj)}\n\n"
                        store_memory(request_text, tool, res.get("new_html", ""))
                elif tool == "code_to_html":
                    code = args.get("code", "")
                    res, token_info = convert_code_to_html(code, model)
                    if res.get("error"):
                        error_occurred = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        data_obj = {
                            "result": "HTML saved to " + res["path"],
                            "tool": "code_to_html",
                            "html_preview": res["html"][:500],
                            "path": res["path"],
                            "token_info": token_info
                        }
                        yield f"data: {json.dumps(data_obj)}\n\n"
                        store_memory(request_text, tool, res.get("html", ""))
                elif tool == "refine_code":
                    code_src = args.get("code", "")
                    instruction = args.get("instruction", "")
                    res, token_info = refine_code(code_src, instruction, model)
                    if res.get("error"):
                        error_occurred = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        data_obj = {
                            "result": "Code refined",
                            "tool": "refine_code",
                            "refined_code": res["refined_code"],
                            "token_info": token_info
                        }
                        yield f"data: {json.dumps(data_obj)}\n\n"
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
                        data_obj = {
                            "result": "App saved to " + res["path"],
                            "tool": "app_gen",
                            "code_preview": res["code"][:500],
                            "path": res["path"],
                            "token_info": token_info
                        }
                        yield f"data: {json.dumps(data_obj)}\n\n"
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

@app.route("/gemini_image", methods=["POST"])
def route_gemini_image():
    data = request.get_json()
    prompt = data.get("prompt", "")
    image_base64 = data.get("image", None)
    if not prompt:
        return jsonify({"error": "No prompt provided"})
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
            if user_input and len(user_input) > 2:
                suggestions.add(user_input[:100])
    plugins = get_plugins_info()
    for pname, info in plugins.items():
        suggestions.add(info["name"])
        suggestions.add(f"Run plugin {info['name']}")
    tool_commands = [
        "generate website", "refine html", "convert code to html",
        "refine code", "search web", "generate app", "clone website",
        "send email", "read emails", "organise files", "check weather",
        "review code", "make presentation", "create video"
    ]
    for cmd in tool_commands:
        if cmd.startswith(prefix.lower()) or prefix.lower() in cmd:
            suggestions.add(cmd)
    return jsonify({"suggestions": sorted(suggestions)[:10]})

@app.route("/system_stats", methods=["GET"])
def system_stats():
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    ram_percent = memory.percent
    ram_used_gb = memory.used / (1024 ** 3)
    ram_total_gb = memory.total / (1024 ** 3)
    gpu_info = {}
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
            gpu_info = {
                "gpu_percent": int(gpu_util),
                "gpu_mem_used_gb": float(mem_used) / 1024,
                "gpu_mem_total_gb": float(mem_total) / 1024,
            }
    except Exception:
        pass
    return jsonify({
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "ram_used_gb": round(ram_used_gb, 1),
        "ram_total_gb": round(ram_total_gb, 1),
        "gpu": gpu_info,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*60}")
    print(f"  AI Studio  –  OpenRouter Edition")
    print(f"  Text model : {DEFAULT_TEXT_MODEL}")
    print(f"  Image model: {IMAGE_MODEL}")
    print(f"  API key set: {'Yes' if OPENROUTER_API_KEY else 'No – set OPENROUTER_API_KEY'}")
    print(f"  Port       : {port}")
    print(f"{'='*60}\n")
    app.run(debug=False, host="0.0.0.0", port=port)