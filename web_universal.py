# -*- coding: utf-8 -*-
# =============================================================================
#  AI Studio – Cloud Edition (OpenRouter)
#  Version: 2.0  –  27 May 2026
#
#  Fixes in this version:
#   1. Deprecated model IDs replaced with current working ones
#   2. Image model correctly set to Gemini Nano Banana (free + paid fallback)
#   3. Stable Flask secret key (session-safe across restarts)
#   4. _openrouter_cache defined before functions that reference it
#   5. refine_html() undefined-variable bug fixed (raw: response)
#   6. Real token usage extracted from API response
#   7. <think>…</think> blocks stripped from reasoning models (stream + non-stream)
#   8. stream_smart_agent() checks for API errors before JSON-parsing
#   9. generate_image_with_openrouter() handles Gemini multimodal list responses
#  10. VALID_MODELS whitelist expanded
#  11. NEW: Full fallback chain for text models (never silently fails)
#  12. NEW: Full fallback chain for image models (free → paid → error)
#  13. NEW: /health endpoint shows active model + API key status
# =============================================================================

import os
import re
import time
import hashlib
import json
import requests
import importlib.util
import base64
import subprocess
from datetime import datetime
from flask import (Flask, render_template, request, jsonify,
                   session, Response, stream_with_context, send_from_directory)
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
from playwright.sync_api import sync_playwright
import psutil

# ===========================================================================
# Flask app
# ===========================================================================
app = Flask(__name__)
# Stable secret key – sessions survive restarts.
# Override by setting FLASK_SECRET_KEY in your Railway environment variables.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "studio-secret-change-me-in-prod")


# ===========================================================================
# Portable home directory
# ===========================================================================
def get_user_home():
    if os.name == 'nt':
        return os.environ.get("USERPROFILE", os.path.expanduser("~"))
    return os.environ.get("HOME", os.path.expanduser("~"))

USER_HOME = get_user_home()


# ===========================================================================
# OpenRouter configuration
# ===========================================================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Text model fallback chain
# Tried in order; skips a model on 400 / 404 / 429 / 402 and tries the next.
# ---------------------------------------------------------------------------
FREE_TEXT_MODELS = [
    "meta-llama/llama-4-scout:free",       # fastest, low-latency, great for chat
    "meta-llama/llama-4-maverick:free",    # 128K context, supports vision input
    "deepseek/deepseek-v3:free",           # strong reasoning & coding
    "qwen/qwen3-coder:free",               # best free coding model (1M ctx)
    "openrouter/free",                     # last-resort wildcard
]
DEFAULT_TEXT_MODEL = FREE_TEXT_MODELS[0]

# ---------------------------------------------------------------------------
# Image model fallback chain
# Only true image-generation models (not chat models) should be here.
# ---------------------------------------------------------------------------
IMAGE_MODELS = [
    "google/gemini-2.5-flash-image-preview:free",  # Nano Banana – free preview
    "google/gemini-2.5-flash-image",               # Nano Banana – paid GA
]

# ---------------------------------------------------------------------------
# Combined whitelist (used by sanitize_model)
# ---------------------------------------------------------------------------
VALID_MODELS = set(FREE_TEXT_MODELS) | set(IMAGE_MODELS) | {
    "qwen/qwen3.6-plus",
    "qwen/qwen3.6-flash",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-flash",
    "stepfun/step-3.5-flash",
    "stepfun/step-3.5-flash:free",
    "nvidia/nemotron-3-nano-omni:free",
    "openrouter/owl-alpha:free",
}

# Response cache – must be defined before any function that references it
_openrouter_cache: dict = {}


def sanitize_model(requested: str) -> str:
    """Return requested model if in whitelist, else fall back to default."""
    if requested in VALID_MODELS:
        return requested
    print(f"[sanitize_model] '{requested}' not in whitelist → using {DEFAULT_TEXT_MODEL}")
    return DEFAULT_TEXT_MODEL


# ===========================================================================
# Thinking-tag stripper  (qwen3-coder, deepseek etc. emit <think>…</think>)
# ===========================================================================
def strip_thinking(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ===========================================================================
# Core OpenRouter – non-streaming  (with full fallback chain)
# ===========================================================================
def ask_openrouter(prompt: str, model: str = None, use_cache: bool = True):
    """
    Send a chat completion request.
    If `model` is None or equals DEFAULT_TEXT_MODEL the full FREE_TEXT_MODELS
    chain is tried.  Otherwise only the requested model is tried (still falls
    back to chain on error).
    Returns (reply_str, token_info_dict | None).
    """
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY is not set.", None

    # Build the list of models to try
    if model and model in IMAGE_MODELS:
        # Caller explicitly wants an image model – handle separately
        models_to_try = [model]
    elif model and model != DEFAULT_TEXT_MODEL and model in VALID_MODELS:
        models_to_try = [model] + [m for m in FREE_TEXT_MODELS if m != model]
    else:
        models_to_try = FREE_TEXT_MODELS

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }

    for try_model in models_to_try:
        cache_key = hashlib.md5((try_model + prompt).encode()).hexdigest()
        if use_cache and cache_key in _openrouter_cache:
            return _openrouter_cache[cache_key], None

        payload = {
            "model":       try_model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }

        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=60)

            if resp.status_code in (400, 402, 404, 429):
                print(f"[ask_openrouter] {try_model} → HTTP {resp.status_code}, trying next…")
                continue

            if resp.status_code == 200:
                data  = resp.json()
                raw   = data["choices"][0]["message"]["content"]
                reply = strip_thinking(raw)

                usage = data.get("usage", {})
                token_info = {
                    "model_used":        try_model,
                    "total_tokens":      usage.get("total_tokens", 0),
                    "prompt_tokens":     usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }

                if use_cache:
                    if len(_openrouter_cache) >= 50:
                        _openrouter_cache.pop(next(iter(_openrouter_cache)))
                    _openrouter_cache[cache_key] = reply

                return reply, token_info

            # Any other HTTP error
            return f"OpenRouter error: {resp.status_code} – {resp.text}", None

        except Exception as exc:
            print(f"[ask_openrouter] {try_model} exception: {exc}, trying next…")
            continue

    return ("All free text models are currently unavailable or rate-limited. "
            "Please try again in a few minutes."), None


# ===========================================================================
# Core OpenRouter – streaming  (with fallback chain)
# ===========================================================================
def ask_openrouter_stream(prompt: str, model: str = None):
    if not OPENROUTER_API_KEY:
        yield f"data: {json.dumps({'error': 'OPENROUTER_API_KEY not set'})}\n\n"
        return

    models_to_try = (
        [model] + [m for m in FREE_TEXT_MODELS if m != model]
        if model and model in VALID_MODELS and model not in IMAGE_MODELS
        else FREE_TEXT_MODELS
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }

    for try_model in models_to_try:
        payload = {
            "model":    try_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   True,
        }
        try:
            with requests.post(OPENROUTER_URL, headers=headers,
                               json=payload, stream=True, timeout=120) as resp:

                if resp.status_code in (400, 402, 404, 429):
                    print(f"[stream] {try_model} → HTTP {resp.status_code}, trying next…")
                    continue

                if resp.status_code != 200:
                    yield f"data: {json.dumps({'error': f'OpenRouter {resp.status_code}'})}\n\n"
                    return

                # State machine to strip <think>…</think> blocks mid-stream
                in_think  = False
                think_buf = ""

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        token = (chunk.get("choices", [{}])[0]
                                     .get("delta", {})
                                     .get("content", ""))
                        if not token:
                            continue

                        if not in_think:
                            if "<think>" in token:
                                before, rest = token.split("<think>", 1)
                                if before:
                                    yield f"data: {json.dumps({'token': before})}\n\n"
                                in_think  = True
                                think_buf = rest
                            else:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        else:
                            think_buf += token
                            if "</think>" in think_buf:
                                _, after  = think_buf.split("</think>", 1)
                                in_think  = False
                                think_buf = ""
                                if after:
                                    yield f"data: {json.dumps({'token': after})}\n\n"
                    except Exception:
                        continue

                yield f"data: {json.dumps({'done': True})}\n\n"
                return   # success – stop trying fallbacks

        except Exception as exc:
            print(f"[stream] {try_model} exception: {exc}, trying next…")
            continue

    yield f"data: {json.dumps({'error': 'All streaming models failed or are rate-limited.'})}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"


# ===========================================================================
# Image generation  (Gemini Nano Banana via OpenRouter, free → paid fallback)
# ===========================================================================
def generate_image_with_openrouter(prompt: str, image_base64: str = None):
    """
    Returns (filepath_or_None, error_str_or_None).
    Tries free preview first, falls back to paid GA automatically.
    """
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY is not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }

    content_parts = [{"type": "text", "text": prompt}]
    if image_base64:
        b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
        content_parts.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    last_error = "Unknown error"
    for img_model in IMAGE_MODELS:
        payload = {
            "model":    img_model,
            "messages": [{"role": "user", "content": content_parts}],
        }
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=120)

            if resp.status_code in (400, 402, 404, 429):
                last_error = f"Model {img_model} returned HTTP {resp.status_code}"
                print(f"[image_gen] {last_error}, trying next…")
                continue

            if resp.status_code != 200:
                last_error = f"OpenRouter error {resp.status_code}: {resp.text}"
                continue

            message = resp.json()["choices"][0]["message"]
            content = message.get("content", "")

            # Gemini returns a list of content parts (multimodal response)
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "image_url":
                        url_data = part.get("image_url", {}).get("url", "")
                        if url_data.startswith("data:image"):
                            hdr, b64 = url_data.split(",", 1)
                            ext = hdr.split("/")[1].split(";")[0]
                            return _save_b64_image(b64, ext), None
                        elif url_data.startswith("http"):
                            return _download_image(url_data), None
                    if ptype == "text":
                        path = _extract_image_from_text(part.get("text", ""))
                        if path:
                            return path, None
                last_error = "No image found in multimodal response"
                continue

            # Plain string response – scan for embedded image data
            path = _extract_image_from_text(str(content))
            if path:
                return path, None

            last_error = f"No image found in response: {str(content)[:200]}"

        except Exception as exc:
            last_error = str(exc)
            print(f"[image_gen] {img_model} exception: {exc}, trying next…")
            continue

    return None, f"Image generation failed: {last_error}"


def _extract_image_from_text(text: str):
    """Scan a text string for an image (markdown URL, plain URL, base64). Returns filepath or None."""
    # Markdown: ![alt](url)
    m = re.search(r'!\[.*?\]\((https?://[^\s\)]+)\)', text)
    if m:
        return _download_image(m.group(1))
    # Plain URL ending in image extension
    m = re.search(r'(https?://[^\s]+\.(png|jpg|jpeg|gif|webp))', text, re.IGNORECASE)
    if m:
        return _download_image(m.group(1))
    # Inline base64 data URI
    m = re.search(r'data:image/([^;]+);base64,([A-Za-z0-9+/=]+)', text)
    if m:
        return _save_b64_image(m.group(2), m.group(1))
    return None


def _save_b64_image(b64_string: str, ext: str = "png"):
    try:
        data     = base64.b64decode(b64_string)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join("static", "generated", f"generated_{ts}.{ext}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(data)
        return filepath
    except Exception as exc:
        print(f"[_save_b64_image] {exc}")
        return None


def _download_image(url: str):
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            ext = url.split(".")[-1].split("?")[0].lower()
            if ext not in ("png", "jpg", "jpeg", "gif", "webp"):
                ext = "png"
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("static", "generated", f"generated_{ts}.{ext}")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(r.content)
            return filepath
        return None
    except Exception as exc:
        print(f"[_download_image] {exc}")
        return None


# ===========================================================================
# ChromaDB memory
# ===========================================================================
CHROMA_PATH = os.path.join(USER_HOME, ".qwen_studio_memory")
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn      = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2")
memory_collection = chroma_client.get_or_create_collection(
                        name="studio_memory",
                        embedding_function=embedding_fn,
                        metadata={"hnsw:space": "cosine"})


def store_memory(user_input: str, action: str, output: str, metadata: dict = None):
    doc_id = f"{int(time.time())}_{hashlib.md5(user_input.encode()).hexdigest()[:8]}"
    doc    = f"User: {user_input}\nAction: {action}\nOutput: {output[:500]}"
    meta   = {"user_input": user_input, "action": action, "timestamp": time.time()}
    if metadata:
        meta.update(metadata)
    memory_collection.upsert(documents=[doc], metadatas=[meta], ids=[doc_id])


def recall_memory(query: str, n_results: int = 3):
    results = memory_collection.query(query_texts=[query], n_results=n_results)
    if results and results["documents"] and results["documents"][0]:
        return results["documents"][0]
    return []


# ===========================================================================
# Plugin system
# ===========================================================================
PLUGINS_DIR = os.path.join(os.path.dirname(__file__), "plugins")
os.makedirs(PLUGINS_DIR, exist_ok=True)
_init = os.path.join(PLUGINS_DIR, "__init__.py")
if not os.path.exists(_init):
    open(_init, "w").close()

_plugins: dict = {}


def load_plugins():
    global _plugins
    _plugins = {}
    for fname in os.listdir(PLUGINS_DIR):
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        mod_name  = fname[:-3]
        file_path = os.path.join(PLUGINS_DIR, fname)
        spec      = importlib.util.spec_from_file_location(mod_name, file_path)
        module    = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            if hasattr(module, "run") and callable(module.run):
                info = {"run": module.run}
                if hasattr(module, "get_info") and callable(module.get_info):
                    info.update(module.get_info())
                else:
                    info["name"]        = mod_name.replace("_", " ").title()
                    info["description"] = f"Plugin {mod_name}"
                key = info["name"].lower().replace(" ", "_")
                _plugins[key] = info
                print(f"[Plugin] Loaded: {info['name']}")
            else:
                print(f"[Plugin] Skipped {fname}: no run() function")
        except Exception as exc:
            print(f"[Plugin] Error loading {fname}: {exc}")


load_plugins()


def get_plugins_info():
    return {k: {"name": v.get("name", k), "description": v.get("description", "")}
            for k, v in _plugins.items()}


def run_plugin(name: str, args: dict):
    if name not in _plugins:
        return {"error": f"Plugin '{name}' not found"}
    try:
        return {"result": _plugins[name]["run"](args)}
    except Exception as exc:
        return {"error": str(exc)}


# ===========================================================================
# Helpers
# ===========================================================================
def clean_html(raw: str) -> str:
    raw = re.sub(r'^```[^\n]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    raw = raw.replace('```', '')
    raw = re.sub(r'`html`', '', raw)
    return raw.strip()


def save_file(content, filename: str) -> str:
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)
    mode = "wb" if isinstance(content, bytes) else "w"
    kw   = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
    with open(path, mode, **kw) as f:
        f.write(content)
    return path


def is_image_request(message: str) -> bool:
    keywords = [
        "draw", "generate image", "create an image", "generate a picture",
        "make an image", "image of", "picture of", "create a picture",
        "generate a photo", "sketch", "illustrate", "visualize",
        "nano banana", "paint a", "design an image",
    ]
    return any(kw in message.lower() for kw in keywords)


def _extract_html(text: str):
    """Return the first complete HTML document found in text, or None."""
    m = re.search(r'<!DOCTYPE\s+html[^>]*>.*?</html>', text, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r'<html[^>]*>.*?</html>', text, re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else None


# ===========================================================================
# Core tools
# ===========================================================================
def generate_website(task, filename, style_guide, model=None):
    model    = sanitize_model(model or DEFAULT_TEXT_MODEL)
    memories = recall_memory("website " + task, n_results=2)
    mem_ctx  = "\n".join(memories) if memories else ""
    prompt   = (
        f"You are an expert front-end developer. The user asks: {task}\n"
        f"Style guide: {style_guide}\n"
        f"Past examples:\n{mem_ctx}\n\n"
        "Create a complete standalone HTML page: semantic HTML5, modern dark theme "
        "with neon accents, rounded corners, CSS Grid/Flexbox, vanilla JS.\n"
        "Output ONLY raw HTML starting with <!DOCTYPE html>. No backticks."
    )
    response, token_info = ask_openrouter(prompt, model=model)
    html = _extract_html(clean_html(response))
    if html:
        safe = re.sub(r'[\\/*?:"<>|]', "", filename)
        path = save_file(html, f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}.html")
        store_memory(task, "website_generation", html)
        return {"html": html, "path": path, "error": None}, token_info
    return {"error": "Could not extract valid HTML", "raw": response}, token_info


def refine_html(original_html, instruction, model=None):
    model    = sanitize_model(model or DEFAULT_TEXT_MODEL)
    prompt   = (
        f"You are an expert front-end developer.\n"
        f"HTML:\n{original_html}\n\n"
        f"The user wants: {instruction}\n"
        "Output the COMPLETE refined HTML starting with <!DOCTYPE html>. No backticks."
    )
    response, token_info = ask_openrouter(prompt, model=model)
    html = _extract_html(clean_html(response))
    if html:
        path = save_file(html, f"refined_{time.strftime('%Y%m%d_%H%M%S')}.html")
        store_memory(instruction, "html_refinement", html)
        return {"new_html": html, "path": path, "error": None}, token_info
    # FIX #5 – was `raw: refined` (undefined variable)
    return {"error": "OpenRouter did not return valid HTML", "raw": response}, token_info


def convert_code_to_html(code, model=None):
    model  = sanitize_model(model or DEFAULT_TEXT_MODEL)
    prompt = (
        f"You are a helpful assistant.\nCode:\n```{code}```\n\n"
        "Create a complete standalone HTML page that displays this code "
        "syntax-highlighted and explains what it does.\n"
        "Output ONLY raw HTML starting with <!DOCTYPE html>. No backticks."
    )
    response, token_info = ask_openrouter(prompt, model=model)
    html = _extract_html(clean_html(response))
    if html:
        path = save_file(html, f"code_display_{time.strftime('%Y%m%d_%H%M%S')}.html")
        store_memory(code[:100], "code_to_html", html)
        return {"html": html, "path": path, "error": None}, token_info
    return {"error": "Failed to generate valid HTML", "raw": response}, token_info


def refine_code(code, instruction, model=None):
    model  = sanitize_model(model or DEFAULT_TEXT_MODEL)
    prompt = (
        f"You are an expert programmer.\nCode:\n```{code}```\n\n"
        f"The user wants: {instruction}\n"
        "Output ONLY the refined code, no explanations, no backticks."
    )
    response, token_info = ask_openrouter(prompt, model=model)
    refined = re.sub(r'^```[^\n]*\n?', '', response)
    refined = re.sub(r'\n?```$', '', refined).strip()
    store_memory(instruction, "code_refinement", refined)
    return {"refined_code": refined, "error": None}, token_info


def web_search_ddg(query: str):
    """DuckDuckGo search – renamed to avoid shadowing the Flask /search route."""
    try:
        with DDGS() as ddgs:
            results   = list(ddgs.text(query, max_results=5))
            formatted = "\n\n".join(
                f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
                for r in results
            )
            store_memory(query, "web_search", formatted)
            return {"results": formatted, "error": None}, None
    except Exception as exc:
        return {"error": str(exc)}, None


def generate_app(description, language, filename_base, model=None):
    model    = sanitize_model(model or DEFAULT_TEXT_MODEL)
    lang_ext = {"python": ".py", "powershell": ".ps1", "bash": ".sh", "batch": ".bat"}
    ext      = lang_ext.get(language, ".txt")
    prompt   = (
        f"You are a senior software engineer. The user wants: {description}\n"
        f"Generate complete ready-to-run {language} code. "
        "Output ONLY the raw code, no backticks."
    )
    code, token_info = ask_openrouter(prompt, model=model)
    code = re.sub(r'^```[^\n]*\n?', '', code)
    code = re.sub(r'\n?```$', '', code).strip()
    base = (re.sub(r'[\\/*?:"<>|]', "", filename_base) if filename_base
            else "_".join(re.findall(r'\b\w+\b', description.lower())[:3]) or "app")
    path = save_file(code, f"{base}{ext}")
    store_memory(description, "app_generation", code)
    return {"code": code, "path": path, "error": None}, token_info


def clone_website(url, output_dir, progress_callback=None):
    clone_root   = os.path.join("data", output_dir)
    os.makedirs(clone_root, exist_ok=True)
    url_to_local = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_context().new_page()

        if progress_callback:
            progress_callback("Loading page…")
        page.goto(url, wait_until="networkidle", timeout=180_000)
        page.wait_for_timeout(2000)
        html = page.content()

        if progress_callback:
            progress_callback("Collecting assets…")
        asset_urls = page.evaluate("""() => {
            const s = new Set();
            document.querySelectorAll('link[rel=stylesheet]').forEach(e => e.href && s.add(e.href));
            document.querySelectorAll('script[src]').forEach(e => e.src && s.add(e.src));
            document.querySelectorAll('img[src]').forEach(e => e.src && s.add(e.src));
            document.querySelectorAll('link[rel=icon]').forEach(e => e.href && s.add(e.href));
            document.querySelectorAll('link[rel=preload]').forEach(e => e.href && s.add(e.href));
            return [...s];
        }""")

        total = len(asset_urls)
        for idx, asset_url in enumerate(asset_urls):
            if progress_callback:
                progress_callback(f"Asset {idx+1}/{total}: {os.path.basename(asset_url)[:40]}…")
            parsed   = urlparse(asset_url)
            full_url = asset_url if parsed.netloc else urljoin(url, asset_url)
            path     = (parsed.path or "/").lstrip("/") or "index.html"
            if path.endswith("/"):
                path += "index.html"
            local_path = os.path.join(clone_root, path)
            try:
                r = requests.get(full_url, timeout=30)
                if r.status_code == 200:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    ct = r.headers.get("content-type", "")
                    if "text/" in ct or ct in ("application/javascript", "text/css"):
                        with open(local_path, "w", encoding="utf-8") as f:
                            f.write(r.text)
                    else:
                        with open(local_path, "wb") as f:
                            f.write(r.content)
                    url_to_local[asset_url] = os.path.relpath(
                        local_path, clone_root).replace("\\", "/")
            except Exception as exc:
                print(f"[clone] Failed {full_url}: {exc}")

        if progress_callback:
            progress_callback("Rewriting paths…")
        for orig, local in url_to_local.items():
            html = re.sub(re.escape(orig), local, html)
        index_path = os.path.join(clone_root, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html)
        browser.close()

    store_memory(url, "website_clone", f"Cloned to {clone_root}")
    return index_path, clone_root


# ===========================================================================
# Flask routes
# ===========================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Quick health-check – shows active models and API key status."""
    return jsonify({
        "status":             "ok",
        "api_key_set":        bool(OPENROUTER_API_KEY),
        "text_model_chain":   FREE_TEXT_MODELS,
        "image_model_chain":  IMAGE_MODELS,
        "default_text_model": DEFAULT_TEXT_MODEL,
    })


@app.route("/models")
def list_models():
    return jsonify({"models": sorted(VALID_MODELS)})


@app.route("/list_plugins")
def list_plugins():
    return jsonify({"plugins": get_plugins_info()})


@app.route("/run_plugin", methods=["POST"])
def run_plugin_endpoint():
    d = request.get_json()
    return jsonify(run_plugin(d.get("plugin"), d.get("args", {})))


@app.route("/generate", methods=["POST"])
def route_generate():
    d = request.get_json()
    res, ti = generate_website(
        d.get("task", ""), d.get("filename", "website"),
        d.get("styleGuide", ""), sanitize_model(d.get("model", DEFAULT_TEXT_MODEL)))
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"html": res["html"], "path": res["path"], "token_info": ti})


@app.route("/refine", methods=["POST"])
def route_refine():
    d = request.get_json()
    res, ti = refine_html(
        d.get("html", ""), d.get("instruction", ""),
        sanitize_model(d.get("model", DEFAULT_TEXT_MODEL)))
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"new_html": res["new_html"], "path": res["path"], "token_info": ti})


@app.route("/convert_code", methods=["POST"])
def route_convert_code():
    d = request.get_json()
    res, ti = convert_code_to_html(
        d.get("code", ""), sanitize_model(d.get("model", DEFAULT_TEXT_MODEL)))
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"html": res["html"], "path": res["path"], "token_info": ti})


@app.route("/refine_code", methods=["POST"])
def route_refine_code():
    d = request.get_json()
    res, ti = refine_code(
        d.get("code", ""), d.get("instruction", ""),
        sanitize_model(d.get("model", DEFAULT_TEXT_MODEL)))
    if res.get("error"):
        return jsonify({"error": res["error"], "raw": res.get("raw")})
    return jsonify({"refined_code": res["refined_code"], "token_info": ti})


@app.route("/search", methods=["POST"])
def route_search():
    d = request.get_json()
    res, _ = web_search_ddg(d.get("query", ""))
    if res.get("error"):
        return jsonify({"error": res["error"]})
    return jsonify({"results": res["results"]})


@app.route("/generate_app", methods=["POST"])
def route_generate_app():
    d = request.get_json()
    res, ti = generate_app(
        d.get("description", ""), d.get("language", "python"),
        d.get("filename", ""), sanitize_model(d.get("model", DEFAULT_TEXT_MODEL)))
    if res.get("error"):
        return jsonify({"error": res["error"]})
    return jsonify({"code": res["code"], "path": res["path"], "token_info": ti})


@app.route("/chat", methods=["POST"])
def route_chat():
    d       = request.get_json()
    message = d.get("message", "")
    model   = sanitize_model(d.get("model", DEFAULT_TEXT_MODEL))

    # Auto-route image requests
    if is_image_request(message):
        path, err = generate_image_with_openrouter(message)
        if err:
            return jsonify({"reply": f"❌ Image generation failed: {err}"})
        img_url = f"/{path.replace(os.sep, '/')}"
        store_memory(message, "image_generation", f"Generated image at {path}")
        return jsonify({"reply": f"✅ Image generated!\n![Generated Image]({img_url})"})

    relevant      = recall_memory(message, n_results=5)
    memory_context = ("Relevant past memories:\n" + "\n".join(relevant) + "\n\n"
                      if relevant else "")

    name_m = re.search(r"(my name is|call me|i am) (\w+)", message, re.IGNORECASE)
    if name_m:
        store_memory(message, "personal_info", f"User's name is {name_m.group(2)}")
        memory_context += f"IMPORTANT: The user's name is {name_m.group(2)}.\n"

    if re.search(r"(i like|i prefer|my favorite|i love)", message, re.IGNORECASE):
        store_memory(message, "preference", message)

    if "conv_history" not in session:
        session["conv_history"] = []
    history = session["conv_history"]

    prompt = "You are a helpful AI assistant that remembers past conversations.\n"
    prompt += memory_context
    for msg in history[-10:]:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"User: {message}\nAssistant:"

    reply, token_info = ask_openrouter(prompt, model=model, use_cache=False)
    store_memory(message, "chat_interaction", reply)
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": reply})
    session["conv_history"] = history[-20:]
    return jsonify({"reply": reply, "token_info": token_info})


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop("conv_history", None)
    return jsonify({"status": "cleared"})


@app.route("/clone", methods=["POST"])
def route_clone():
    d   = request.get_json()
    url = d.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"})
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    ts  = time.strftime("%Y%m%d_%H%M%S")
    try:
        idx, cdir = clone_website(url, f"cloned_site_{ts}")
        return jsonify({"path": idx, "directory": cdir, "status": "success"})
    except Exception as exc:
        return jsonify({"error": str(exc)})


@app.route("/preview/<folder>/<path:filename>")
def serve_clone(folder, filename):
    p = os.path.join("data", folder)
    if not os.path.exists(p):
        return "Clone folder not found", 404
    if ".." in filename or filename.startswith("/"):
        return "Invalid path", 400
    return send_from_directory(p, filename)


@app.route("/gemini_image", methods=["POST"])
def route_gemini_image():
    d = request.get_json()
    if not d.get("prompt"):
        return jsonify({"error": "No prompt provided"})
    path, err = generate_image_with_openrouter(d["prompt"], d.get("image"))
    if err:
        return jsonify({"error": err})
    return jsonify({"path": path, "status": "success"})


@app.route("/list_memories")
def list_memories():
    search  = request.args.get("search", "")
    limit   = int(request.args.get("limit", 50))
    results = (memory_collection.query(query_texts=[search], n_results=limit)
               if search else
               memory_collection.query(query_texts=["a"], n_results=limit))
    mems = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            mems.append({
                "id":       doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
    return jsonify({"memories": mems})


@app.route("/delete_memory", methods=["POST"])
def delete_memory():
    d = request.get_json()
    if not d.get("id"):
        return jsonify({"error": "No ID provided"}), 400
    try:
        memory_collection.delete(ids=[d["id"]])
        return jsonify({"status": "success"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/compress_memory", methods=["POST"])
def compress_memory():
    d        = request.get_json() or {}
    days_old = d.get("days_old", 30)
    max_mems = d.get("max_memories", 20)
    try:
        results = memory_collection.get(limit=1000)
        if not results or not results["ids"]:
            return jsonify({"error": "No memories found"}), 400
        all_mems = [
            {"id": results["ids"][i], "document": results["documents"][i],
             "timestamp": (results["metadatas"][i] if results["metadatas"] else {}).get("timestamp", 0)}
            for i in range(len(results["ids"]))
        ]
        now     = time.time()
        old     = sorted([m for m in all_mems if (now - m["timestamp"]) > days_old * 86400],
                         key=lambda x: x["timestamp"])
        if not old:
            return jsonify({"message": f"No memories older than {days_old} days."})
        batch = old[:max_mems]
        docs  = "\n\n---\n\n".join(
            f"Memory {i+1}:\n{m['document']}" for i, m in enumerate(batch))
        summary, _ = ask_openrouter(
            f"Summarise these memories concisely (max 300 chars):\n\n{docs}\n\nSummary:",
            use_cache=False)
        summary = summary.strip()[:500]
        store_memory("memory_compression", "memory_summary", summary,
                     {"compressed": True, "original_count": len(batch)})
        memory_collection.delete(ids=[m["id"] for m in batch])
        return jsonify({"message": f"Compressed {len(batch)} memories.",
                        "summary": summary, "deleted_count": len(batch)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/backup_memory", methods=["POST"])
def backup_memory():
    return jsonify({"error": "Backup not available in cloud version"}), 501


@app.route("/restore_memory", methods=["POST"])
def restore_memory():
    return jsonify({"error": "Restore not available in cloud version"}), 501


@app.route("/open_file_cleaner", methods=["POST"])
def open_file_cleaner():
    return jsonify({"error": "File cleaner not available in cloud version"})


@app.route("/autocomplete")
def autocomplete():
    prefix = request.args.get("prefix", "").strip()
    if not prefix or len(prefix) < 2:
        return jsonify({"suggestions": []})
    suggestions = set()
    for mem in recall_memory(prefix, n_results=5):
        m = re.search(r"User: (.*?)(?:\n|$)", mem)
        if m and len(m.group(1).strip()) > 2:
            suggestions.add(m.group(1).strip()[:100])
    for info in get_plugins_info().values():
        suggestions.add(info["name"])
    for cmd in ["generate website", "refine html", "convert code to html", "refine code",
                "search web", "generate app", "clone website", "send email"]:
        if prefix.lower() in cmd:
            suggestions.add(cmd)
    return jsonify({"suggestions": sorted(suggestions)[:10]})


@app.route("/system_stats")
def system_stats():
    mem = psutil.virtual_memory()
    gpu = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            gu, mu, mt = r.stdout.strip().split(", ")
            gpu = {"gpu_percent": int(gu),
                   "gpu_mem_used_gb": float(mu)/1024,
                   "gpu_mem_total_gb": float(mt)/1024}
    except Exception:
        pass
    return jsonify({
        "cpu_percent":  psutil.cpu_percent(interval=0.5),
        "ram_percent":  mem.percent,
        "ram_used_gb":  round(mem.used  / 1024**3, 1),
        "ram_total_gb": round(mem.total / 1024**3, 1),
        "gpu":          gpu,
    })


@app.route("/favicon.ico")
def favicon():
    return "", 204


# ===========================================================================
# Streaming smart agent  (with fallback chain + robust JSON parsing)
# ===========================================================================
@app.route("/stream_smart_agent")
def stream_smart_agent():
    request_text = request.args.get("request", "")
    style_guide  = request.args.get("styleGuide", "")
    model        = sanitize_model(request.args.get("model", DEFAULT_TEXT_MODEL))

    if not request_text:
        return Response(
            f"data: {json.dumps({'error': 'No request provided'})}\n\n",
            mimetype="text/event-stream")

    def generate():
        # ── Image ──────────────────────────────────────────────────────
        if is_image_request(request_text):
            path, err = generate_image_with_openrouter(request_text)
            if err:
                yield f"data: {json.dumps({'error': err})}\n\n"
            else:
                img_url    = f"/{path.replace(os.sep, '/')}"
                img_result = "✅ Image generated!\n![Generated Image](" + img_url + ")"
                _payload   = json.dumps({"result": img_result, "path": path, "tool": "image_generation"})
                yield f"data: {_payload}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # ── Email plugin ───────────────────────────────────────────────
        if any(kw in request_text.lower() for kw in
               ["send an email", "send email", "email to", "mail to"]):
            args = {"to": "recipient@example.com"}
            em   = re.search(r'[\w\.-]+@[\w\.-]+', request_text)
            if em:
                args["to"] = em.group(0)
            yield f"data: {json.dumps({'step_start': 'plugin_email_sender', 'step_index': 0})}\n\n"
            res = run_plugin("email_sender", args)
            key = "error" if "error" in res else "result"
            yield f"data: {json.dumps({key: res[key], 'tool': 'plugin_email_sender'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # ── Screenshot plugin ──────────────────────────────────────────
        if any(kw in request_text.lower() for kw in
               ["screenshot", "take a screenshot", "capture screenshot"]):
            um = re.search(r'(https?://[^\s]+)', request_text)
            if not um:
                yield f"data: {json.dumps({'error': 'No URL found for screenshot'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            res = run_plugin("agent_browser", {"url": um.group(0), "action": "screenshot"})
            key = "error" if "error" in res else "result"
            yield f"data: {json.dumps({key: res[key], 'tool': 'plugin_agent_browser'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # ── Tool-routing decision ──────────────────────────────────────
        memories     = recall_memory(request_text, n_results=5)
        mem_ctx      = "\n".join(memories) if memories else "No past tasks."
        plugins_text = (
            "\n".join(f"- plugin_{k}: {v['description']}"
                      for k, v in get_plugins_info().items())
            or "No plugins available."
        )

        decision_prompt = f"""You are a smart assistant that plans tool sequences.

Memory:
{mem_ctx}

Built-in tools:
- website: generate HTML site. Args: task, filename (optional)
- refine_html: improve HTML. Args: html, instruction
- code_to_html: convert code to HTML. Args: code
- refine_code: modify code. Args: code, instruction
- search: web search. Args: query
- app_gen: generate script. Args: description, language (python|powershell|bash|batch), filename (optional)
- clone: clone a website. Args: url
- chat: general conversation. Args: message

Plugins:
{plugins_text}

Respond with ONLY a JSON object or array — no prose, no markdown, no backticks.
Format: {{"tool": "...", "arguments": {{...}}}}
or an array of such objects for multi-step plans.

User request: "{request_text}"
Style guide: {style_guide}
"""
        raw_decision, _ = ask_openrouter(decision_prompt, model=model, use_cache=False)

        # FIX #8 – Check for API error before JSON parsing
        if (not raw_decision or
                raw_decision.startswith("OpenRouter error") or
                raw_decision.startswith("Error:") or
                raw_decision.startswith("All free")):
            yield f"data: {json.dumps({'error': f'Routing failed: {raw_decision}'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # FIX #7/#8 – Strip think tags and markdown fences before parsing
        clean = strip_thinking(raw_decision)
        clean = re.sub(r'^```[^\n]*\n?', '', clean)
        clean = re.sub(r'\n?```$', '', clean).strip()

        try:
            decision = json.loads(clean)
        except Exception:
            # Try to salvage a JSON object/array from the text
            m = re.search(r'(\[.*\]|\{.*\})', clean, re.DOTALL)
            if m:
                try:
                    decision = json.loads(m.group(1))
                except Exception:
                    yield f"data: {json.dumps({'error': f'Could not parse routing decision: {clean[:200]}'})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
            else:
                yield f"data: {json.dumps({'error': f'Could not parse routing decision: {clean[:200]}'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

        decisions = [decision] if isinstance(decision, dict) else decision

        for step_idx, action in enumerate(decisions):
            tool = action.get("tool", "")
            args = action.get("arguments", {})
            yield f"data: {json.dumps({'step_start': tool, 'step_index': step_idx})}\n\n"
            err_flag = False

            try:
                # ── chat ───────────────────────────────────────────────
                if tool == "chat":
                    msg     = args.get("message", request_text)
                    mem     = recall_memory(msg, n_results=3)
                    prompt  = ("Memory:\n" + "\n".join(mem) + "\n\n" if mem else "")
                    prompt += f"User: {msg}\nAssistant:"
                    for chunk in ask_openrouter_stream(prompt, model=model):
                        yield chunk

                # ── search ─────────────────────────────────────────────
                elif tool == "search":
                    res, _ = web_search_ddg(args.get("query", request_text))
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        yield f"data: {json.dumps({'result': res['results'], 'tool': 'search'})}\n\n"

                # ── clone ──────────────────────────────────────────────
                elif tool == "clone":
                    url = args.get("url", request_text)
                    if not url.startswith(("http://", "https://")):
                        url = "https://" + url
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    yield f"data: {json.dumps({'result': '🌐 Starting clone…', 'tool': 'clone'})}\n\n"
                    try:
                        idx_path, cdir = clone_website(url, f"cloned_site_{ts}")
                        yield f"data: {json.dumps({'result': f'✅ Clone complete! Saved to {cdir}', 'path': idx_path, 'directory': cdir, 'tool': 'clone'})}\n\n"
                        yield f"data: {json.dumps({'clone_preview': cdir})}\n\n"
                    except Exception as exc:
                        err_flag = True
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n"

                # ── plugin_* ───────────────────────────────────────────
                elif tool.startswith("plugin_"):
                    pname = tool[7:]
                    res   = run_plugin(pname, args)
                    key   = "error" if "error" in res else "result"
                    if key == "error":
                        err_flag = True
                    yield f"data: {json.dumps({key: res[key], 'tool': tool})}\n\n"

                # ── website ────────────────────────────────────────────
                elif tool == "website":
                    res, ti = generate_website(
                        args.get("task", request_text),
                        args.get("filename", "website"),
                        style_guide, model)
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        _p = res["path"]
                        _payload = json.dumps({"result": f"Website saved to {_p}", "tool": "website", "html_preview": res["html"][:500], "path": _p, "token_info": ti})
                        yield f"data: {_payload}\n\n"

                # ── refine_html ────────────────────────────────────────
                elif tool == "refine_html":
                    res, ti = refine_html(args.get("html", ""), args.get("instruction", ""), model)
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        _p = res["path"]
                        _payload = json.dumps({"result": f"Refined HTML saved to {_p}", "tool": "refine_html", "path": _p, "token_info": ti})
                        yield f"data: {_payload}\n\n"

                # ── code_to_html ───────────────────────────────────────
                elif tool == "code_to_html":
                    res, ti = convert_code_to_html(args.get("code", ""), model)
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        _p = res["path"]
                        _payload = json.dumps({"result": f"HTML saved to {_p}", "tool": "code_to_html", "path": _p, "token_info": ti})
                        yield f"data: {_payload}\n\n"

                # ── refine_code ────────────────────────────────────────
                elif tool == "refine_code":
                    res, ti = refine_code(args.get("code", ""), args.get("instruction", ""), model)
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        _payload = json.dumps({"result": "Code refined", "tool": "refine_code", "refined_code": res["refined_code"], "token_info": ti})
                        yield f"data: {_payload}\n\n"

                # ── app_gen ────────────────────────────────────────────
                elif tool == "app_gen":
                    res, ti = generate_app(
                        args.get("description", request_text),
                        args.get("language", "python"),
                        args.get("filename", ""), model)
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        _p = res["path"]
                        _payload = json.dumps({"result": f"App saved to {_p}", "tool": "app_gen", "path": _p, "token_info": ti})
                        yield f"data: {_payload}\n\n"

                else:
                    err_flag = True
                    yield f"data: {json.dumps({'error': f'Unknown tool: {tool}'})}\n\n"

            except Exception as exc:
                err_flag = True
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

            if err_flag:
                break

        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*62}")
    print(f"  AI Studio – Cloud Edition  |  v2.0  |  27 May 2026")
    print(f"  API key set : {'✅ Yes' if OPENROUTER_API_KEY else '❌ No – set OPENROUTER_API_KEY'}")
    print(f"  Text chain  : {' → '.join(FREE_TEXT_MODELS)}")
    print(f"  Image chain : {' → '.join(IMAGE_MODELS)}")
    print(f"  Port        : {port}")
    print(f"{'='*62}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
