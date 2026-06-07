# -*- coding: utf-8 -*-

# =============================================================================
# AI Studio – Cloud Edition (OpenRouter)
# Version: 2.1 – 07 June 2026
#
# Fixes in this version:
# 1. Laguna M.1 set as DEFAULT model (agentic coding beast)
# 2. Qwen 3.6 Plus as fallback #1
# 3. Enhanced logging to track model selection
# 4. All previous fixes preserved
# =============================================================================

import os
import re
import time
import hashlib
import json
import threading
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


# =============================================================================
# Flask app
# =============================================================================

app = Flask(__name__)

# Stable secret key – sessions survive restarts.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "studio-secret-change-me-in-prod")


# =============================================================================
# Portable home directory
# =============================================================================

def get_user_home():
    if os.name == 'nt':
        return os.environ.get("USERPROFILE", os.path.expanduser("~"))
    return os.environ.get("HOME", os.path.expanduser("~"))

USER_HOME = get_user_home()


# =============================================================================
# OpenRouter configuration
# =============================================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# -----------------------------------------------------------------------------
# Text model fallback chain
# Tried in order; skips a model on 400 / 404 / 429 / 402 and tries the next.
# -----------------------------------------------------------------------------

FREE_TEXT_MODELS = [
    "poolside/laguna-m.1:free",       # ← DEFAULT: Agentic Coding Beast
    "qwen/qwen3.6-plus",              # ← Fallback #1: Big Brain
    "qwen/qwen3.6-flash",             # ← Fallback #2: Fast & Smart
    "meta-llama/llama-4-scout:free",  # fallback — fastest, low-latency
    "meta-llama/llama-4-maverick:free", # 1M context, vision input
    "deepseek/deepseek-v3:free",      # strong reasoning & coding
    "deepseek/deepseek-v4-flash:free", # smart reasoning
    "openai/gpt-oss-120b:free",       # 117B MoE, great all-rounder
]
DEFAULT_TEXT_MODEL = FREE_TEXT_MODELS[0]  # Now Laguna M.1


# -----------------------------------------------------------------------------
# Image model fallback chain
# Only true image-generation models (not chat models) should be here.
# -----------------------------------------------------------------------------

IMAGE_MODELS = [
    "google/gemini-2.5-flash-image",          # Nano Banana 1 (paid GA, confirmed working)
    "google/gemini-3.1-flash-image-preview",  # Nano Banana 2 (paid, newer)
    "x-ai/grok-imagine-image-quality:free",   # Last-resort fallback
]


# -----------------------------------------------------------------------------
# Video generation model
# -----------------------------------------------------------------------------

VIDEO_MODEL = "x-ai/grok-imagine-video"
OPENROUTER_VIDEOS_URL = "https://openrouter.ai/api/v1/videos"


# -----------------------------------------------------------------------------
# Combined whitelist (used by sanitize_model)
# -----------------------------------------------------------------------------

VALID_MODELS = set(FREE_TEXT_MODELS) | set(IMAGE_MODELS) | {VIDEO_MODEL} | {
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-flash",
    "stepfun/step-3.5-flash",
    "stepfun/step-3.5-flash:free",
    "nvidia/nemotron-3-nano-omni:free",
    "openrouter/owl-alpha:free",
}


# Response cache – must be defined before any function that references it
_openrouter_cache: dict = {}

# Persistent HTTP session – reuses TLS connections across all OpenRouter calls
_http_session = requests.Session()

# Server-side video session state.
_video_sessions: dict = {}


# =============================================================================
# System stats – non-blocking CPU sampling + one-time GPU probe
# =============================================================================

_cpu_percent_cache: float = 0.0
_HAS_GPU: bool = False

def _probe_gpu() -> bool:
    """Check once at startup whether nvidia-smi exists and has a GPU."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"],
                          capture_output=True, text=True, timeout=1)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False

_HAS_GPU = _probe_gpu()

def _cpu_sampler_loop():
    """Background daemon: prime the counter then sample every 5 s (non-blocking)."""
    global _cpu_percent_cache
    psutil.cpu_percent()  # prime — first call always returns 0.0
    while True:
        time.sleep(5)
        _cpu_percent_cache = psutil.cpu_percent(interval=None)

threading.Thread(target=_cpu_sampler_loop, daemon=True).start()

def _vid_sid() -> str:
    """Return (and create if needed) a stable video-session ID for this browser."""
    if "vid_sid" not in session:
        session["vid_sid"] = os.urandom(8).hex()
        session.modified = True
    return session["vid_sid"]

def sanitize_model(requested: str) -> str:
    """Return requested model if in whitelist, else fall back to default."""
    if requested in VALID_MODELS:
        print(f"[sanitize_model] ✅ Using requested model: {requested}")
        return requested
    print(f"[sanitize_model] ❌ '{requested}' not in whitelist → falling back to {DEFAULT_TEXT_MODEL}")
    return DEFAULT_TEXT_MODEL


# =============================================================================
# Thinking-tag stripper (qwen3-coder, deepseek etc. emit <think>…</think>)
# =============================================================================

def strip_thinking(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# =============================================================================
# Core OpenRouter – non-streaming (with full fallback chain)
# =============================================================================

def ask_openrouter(prompt: str, model: str = None, use_cache: bool = True):
    """
    Send a chat completion request.
    If model is None or equals DEFAULT_TEXT_MODEL the full FREE_TEXT_MODELS
    chain is tried. Otherwise only the requested model is tried (still falls
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
        "Content-Type": "application/json",
    }

    for try_model in models_to_try:
        print(f"[ask_openrouter] 🔄 Trying model: {try_model}")
        cache_key = hashlib.md5((try_model + prompt).encode()).hexdigest()
        if use_cache and cache_key in _openrouter_cache:
            return _openrouter_cache[cache_key], None

        payload = {
            "model": try_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }

        try:
            resp = _http_session.post(OPENROUTER_URL, headers=headers,
                                     json=payload, timeout=60)

            if resp.status_code in (400, 402, 404, 429):
                print(f"[ask_openrouter] ❌ {try_model} → HTTP {resp.status_code}, trying next…")
                continue

            if resp.status_code == 200:
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
                reply = strip_thinking(raw)

                usage = data.get("usage", {})
                token_info = {
                    "model_used": try_model,
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
                
                print(f"[ask_openrouter] ✅ SUCCESS using model: {try_model}")

                if use_cache:
                    if len(_openrouter_cache) >= 50:
                        _openrouter_cache.pop(next(iter(_openrouter_cache)))
                    _openrouter_cache[cache_key] = reply

                return reply, token_info

            # Any other HTTP error
            return f"OpenRouter error: {resp.status_code} – {resp.text}", None

        except Exception as exc:
            print(f"[ask_openrouter] ❌ {try_model} exception: {exc}, trying next…")
            continue

    return ("All free text models are currently unavailable or rate-limited. "
            "Please try again in a few minutes."), None


# =============================================================================
# Core OpenRouter – streaming (with fallback chain)
# =============================================================================

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
        "Content-Type": "application/json",
    }

    for try_model in models_to_try:
        print(f"[stream] 🔄 Trying model: {try_model}")
        payload = {
            "model": try_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        try:
            with _http_session.post(OPENROUTER_URL, headers=headers,
                                   json=payload, stream=True, timeout=120) as resp:

                if resp.status_code in (400, 402, 404, 429):
                    print(f"[stream] ❌ {try_model} → HTTP {resp.status_code}, trying next…")
                    continue

                if resp.status_code != 200:
                    yield f"data: {json.dumps({'error': f'OpenRouter {resp.status_code}'})}\n\n"
                    return
                
                print(f"[stream] ✅ SUCCESS streaming from model: {try_model}")

                # State machine to strip <think>…</think> blocks mid-stream
                in_think = False
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
                                in_think = True
                                think_buf = rest
                            else:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        else:
                            think_buf += token
                            if "</think>" in think_buf:
                                _, after = think_buf.split("</think>", 1)
                                in_think = False
                                think_buf = ""
                                if after:
                                    yield f"data: {json.dumps({'token': after})}\n\n"
                    except Exception:
                        continue

                yield f"data: {json.dumps({'done': True})}\n\n"
                return  # success – stop trying fallbacks

        except Exception as exc:
            print(f"[stream] ❌ {try_model} exception: {exc}, trying next…")
            continue

    yield f"data: {json.dumps({'error': 'All streaming models failed or are rate-limited.'})}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"


# =============================================================================
# Image generation (Gemini Nano Banana via OpenRouter, free → paid fallback)
# =============================================================================

def generate_image_with_openrouter(prompt: str, image_base64: str = None):
    """
    Returns (filepath_or_None, error_str_or_None).
    Tries free preview first, falls back to paid GA automatically.
    """
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY is not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    content_parts = [{"type": "text", "text": prompt}]
    if image_base64:
        b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    last_error = "Unknown error"
    for img_model in IMAGE_MODELS:
        payload = {
            "model": img_model,
            "messages": [{"role": "user", "content": content_parts}],
            "modalities": ["image", "text"],  # ★ Tells OpenRouter we want an image back
        }
        try:
            resp = _http_session.post(OPENROUTER_URL, headers=headers,
                                     json=payload, timeout=120)

            if resp.status_code in (400, 402, 404, 429):
                last_error = f"Model {img_model} returned HTTP {resp.status_code}"
                print(f"[image_gen] {last_error}, trying next…")
                continue

            if resp.status_code != 200:
                last_error = f"OpenRouter error {resp.status_code}: {resp.text}"
                continue

            message = resp.json()["choices"][0]["message"]

            # ★ FIX: OpenRouter returns images in message["images"], NOT message["content"]
            images = message.get("images", [])
            if images and isinstance(images, list):
                for img in images:
                    if not isinstance(img, dict):
                        continue
                    url_data = img.get("image_url", {}).get("url", "")
                    if url_data.startswith("data:image"):
                        hdr, b64 = url_data.split(",", 1)
                        ext = hdr.split("/")[1].split(";")[0]
                        return _save_b64_image(b64, ext), None
                    elif url_data.startswith("http"):
                        return _download_image(url_data), None

            # Fallback: check content field (older response format)
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
        data = base64.b64decode(b64_string)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("static", "generated", f"generated_{ts}.{ext}")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(r.content)
            return filepath
        return None
    except Exception as exc:
        print(f"[_download_image] {exc}")
        return None

def _download_video(url: str, headers: dict = None):
    try:
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code == 200:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("static", "generated", f"generated_{ts}.mp4")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(r.content)
            return filepath
        return None
    except Exception as exc:
        print(f"[_download_video] {exc}")
        return None


# =============================================================================
# Video generation (x-ai/grok-imagine-video via OpenRouter async jobs API)
# =============================================================================

def generate_video_with_openrouter(prompt: str,
                                  duration: int = 3,
                                  resolution: str = "480p",
                                  aspect_ratio: str = "16:9",
                                  image_b64: str = None):
    """
    Submit an async video job, poll until complete, download the mp4.
    Returns (filepath_or_None, error_str_or_None).

    duration : 1–15 seconds. Default 3 (~$0.15). Pass 8 for the upgrade path (~$0.40).
    resolution : "480p" (cheap) or "720p" (HD).
    aspect_ratio : "16:9" | "9:16" | "1:1" | "4:3" | "3:4" | "3:2" | "2:3".
    image_b64 : optional base64 data URL (or raw b64 string) to use as the first frame
    for image-to-video generation. Omit for text-only video.
    """
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY is not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VIDEO_MODEL,
        "prompt": prompt,
        "duration": max(1, min(15, int(duration))),
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
    }

    # ── Image-to-video: attach first frame + consistency suffix ──────
    if image_b64:
        payload["prompt"] = prompt + _IMG2VID_CONSISTENCY
        # Strip the data URI prefix if present — keep raw base64
        raw_b64 = image_b64.split(",")[1] if "," in image_b64 else image_b64
        payload["frame_images"] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{raw_b64}"},
                "frame_type": "first_frame",
            }
        ]

    # ── Submit job ────────────────────────────────────────────────
    try:
        resp = requests.post(OPENROUTER_VIDEOS_URL, headers=headers,
                            json=payload, timeout=30)
    except Exception as exc:
        return None, f"Video submit request failed: {exc}"

    # 200 = immediate response, 202 = accepted / queued (both are success)
    if resp.status_code not in (200, 202):
        return None, f"Video submit failed: HTTP {resp.status_code} – {resp.text[:300]}"

    try:
        job = resp.json()
    except Exception:
        return None, f"Could not parse video job response: {resp.text[:200]}"

    job_id = job.get("id")
    if not job_id:
        return None, f"No job ID in response: {job}"

    # Use the polling_url from the response if provided, otherwise build it
    poll_url = job.get("polling_url") or f"{OPENROUTER_VIDEOS_URL}/{job_id}"

    print(f"[video_gen] Job submitted: {job_id} poll_url={poll_url}")

    # ── Poll until done (max 5 minutes, 5-second intervals) ───────
    for attempt in range(60):
        time.sleep(5)
        try:
            poll = requests.get(poll_url, headers=headers, timeout=30)
        except Exception as exc:
            print(f"[video_gen] Poll {attempt+1} failed: {exc}")
            continue

        # 200 = data ready, 202 = still processing — both are non-error
        if poll.status_code not in (200, 202):
            return None, f"Poll failed: HTTP {poll.status_code} – {poll.text[:200]}"

        try:
            pdata = poll.json()
        except Exception:
            continue

        status = pdata.get("status", "")
        print(f"[video_gen] Poll {attempt+1}: status={status}")

        if status == "completed":
            unsigned = pdata.get("unsigned_urls")
            video_url = (
                (unsigned[0] if isinstance(unsigned, list) and unsigned else None) or
                pdata.get("url") or
                pdata.get("video_url") or
                pdata.get("data", {}).get("url")
            )
            if not video_url:
                return None, f"No video URL in completed response: {pdata}"
            filepath = _download_video(video_url, headers=headers)
            if filepath:
                return filepath, None
            return None, "Video download failed"

        if status in ("failed", "error", "cancelled"):
            reason = pdata.get("error") or pdata.get("message") or status
            return None, f"Video job {status}: {reason}"

        # still pending/processing — keep polling

    return None, "Video generation timed out after 5 minutes"


# =============================================================================
# ChromaDB memory – lazy init (loads on first use, not at import time)
# =============================================================================

CHROMA_PATH = os.path.join(USER_HOME, ".qwen_studio_memory")
os.makedirs(CHROMA_PATH, exist_ok=True)

chroma_client: object = None
memory_collection: object = None
_memory_ready: bool = False

def _init_memory():
    """Initialize ChromaDB + embedding model on first use. Safe to call multiple times."""
    global chroma_client, memory_collection, _memory_ready
    if _memory_ready:
        return
    _memory_ready = True  # set before init to block re-entrant calls
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2")
            memory_collection = chroma_client.get_or_create_collection(
                name="studio_memory",
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"})
            print("[Memory] ChromaDB + SentenceTransformer ready")
        except Exception as emb_err:
            print(f"[Memory] SentenceTransformer unavailable ({emb_err}), using default embedding")
            memory_collection = chroma_client.get_or_create_collection(
                name="studio_memory",
                metadata={"hnsw:space": "cosine"})
    except Exception as chroma_err:
        print(f"[Memory] ChromaDB failed ({chroma_err}), using in-memory fallback")
        chroma_client = chromadb.EphemeralClient()
        memory_collection = chroma_client.get_or_create_collection(name="studio_memory")


# -----------------------------------------------------------------------------
# Memory tuning constants (easy to adjust without touching logic)
# -----------------------------------------------------------------------------

MEMORY_RELEVANCE_THRESHOLD = 0.75  # cosine distance; lower = stricter relevance filter
MEMORY_DUPLICATE_THRESHOLD = 0.15  # skip storing if a near-identical entry already exists


# -----------------------------------------------------------------------------
# Prompt quality constants
# -----------------------------------------------------------------------------

_CHAT_SYSTEM = (
    "You are a knowledgeable, direct assistant. "
    "Give clear and well-structured answers. "
    "Be concise for simple questions; go into detail when the topic genuinely needs it. "
    "Do not pad answers with filler phrases or unnecessary affirmations."
)

_IMG_QUALITY_SUFFIX = (
    ", professional lighting, high resolution, sharp focus, "
    "detailed, high quality, photorealistic"
)

# Keywords that indicate the user already specified a style — don't override
_IMG_STYLE_KEYWORDS = (
    "photorealistic", "8k", "4k", "hd", "high resolution", "cinematic",
    "oil painting", "watercolor", "anime", "sketch", "illustration",
    "cartoon", "3d render", "digital art", "low poly", "pixel art",
)

def enhance_image_prompt(prompt: str) -> str:
    """Append quality/style suffix unless the user already specified style intent."""
    lower = prompt.lower()
    if any(kw in lower for kw in _IMG_STYLE_KEYWORDS):
        return prompt
    return prompt + _IMG_QUALITY_SUFFIX


# -----------------------------------------------------------------------------
# Multi-step chain constants
# -----------------------------------------------------------------------------

MAX_CHAIN_STEPS = 5

_TOOL_LABELS = {
    "chat": "Chat reply",
    "search": "Web search",
    "website": "Website",
    "app_gen": "Code file",
    "image_generation": "Image",
    "video_generation": "Video",
    "clone": "Site clone",
    "refine_html": "HTML refinement",
    "refine_code": "Code refinement",
    "code_to_html": "HTML conversion",
}


# -----------------------------------------------------------------------------
# User profile – JSON sidecar for durable personal facts
# Always injected at the top of every chat prompt regardless of query.
# -----------------------------------------------------------------------------

USER_PROFILE_PATH = os.path.join(CHROMA_PATH, "user_profile.json")

def load_user_profile() -> dict:
    try:
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_user_profile(profile: dict):
    with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

def update_user_profile(key: str, value: str):
    p = load_user_profile()
    p[key.lower().replace(" ", "_")] = value
    save_user_profile(p)

def get_profile_context() -> str:
    """Return a compact [User Profile] block, or '' if no facts are known."""
    p = load_user_profile()
    if not p:
        return ""
    lines = " | ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in p.items())
    return f"[User Profile]\n{lines}\n\n"


# -----------------------------------------------------------------------------
# Fact extraction – runs on every user message; populates user_profile.json
# and handles explicit "remember that X" → ChromaDB store.
# -----------------------------------------------------------------------------

_REMEMBER_RE = re.compile(
    r"(?:please\s+)?(?:remember(?:\s+that)?|note(?:\s+that)?|don'?t\s+forget(?:\s+that)?"
    r"|keep\s+in\s+mind(?:\s+that)?)[:\s]+(.+)",
    re.IGNORECASE | re.DOTALL,
)

def extract_and_store_facts(message: str) -> bool:
    """
    Extract durable personal facts → user_profile.json.
    Explicit 'remember that X' → ChromaDB store_memory.
    Returns True if at least one fact was found/stored.
    """
    found = False

    # Explicit memory requests → store in ChromaDB
    rem = _REMEMBER_RE.search(message)
    if rem:
        store_memory(message, "explicit_memory", rem.group(1).strip())
        found = True

    # Name
    nm = re.search(r"(?:my name is|call me)\s+(\w+(?:\s+\w+)?)", message, re.IGNORECASE)
    if nm:
        update_user_profile("name", nm.group(1).strip())
        found = True

    # Business / project
    bm = re.search(
        r"(?:my\s+(?:business|company|startup|app|site|website|product)\s+is"
        r"|i(?:'m|\s+am)\s+building)\s+([^.,!?\n]{3,60})",
        message, re.IGNORECASE)
    if bm:
        update_user_profile("business", bm.group(1).strip())
        found = True

    # Role
    rm = re.search(
        r"i(?:'m|\s+am)\s+(?:a\s+|an\s+)?([a-zA-Z][a-zA-Z ]{2,30}?)"
        r"(?:\s+(?:by trade|professionally|by profession))"
        r"(?:[.,!?]|$)", message)
    if rm:
        update_user_profile("role", rm.group(1).strip())
        found = True

    # Preference – accumulate as comma-separated
    pm = re.search(r"i\s+(?:prefer|love|like|enjoy)\s+([^.,!?\n]{3,60})",
                  message, re.IGNORECASE)
    if pm:
        profile = load_user_profile()
        existing = profile.get("preferences", "")
        pref_val = pm.group(1).strip()
        if pref_val.lower() not in existing.lower():
            update_user_profile("preferences",
                              (existing + ", " + pref_val).lstrip(", ") if existing else pref_val)
            found = True

    # Location
    lm = re.search(
        r"i(?:'m|\s+am)\s+(?:based\s+in|from|living\s+in)\s+"
        r"([A-Z][a-zA-Z ,]{2,40}?)(?:[.,!?]|$)", message)
    if lm:
        update_user_profile("location", lm.group(1).strip())
        found = True

    return found

def store_memory(user_input: str, action: str, output: str, metadata: dict = None):
    _init_memory()
    doc_id = f"{int(time.time())}_{hashlib.md5(user_input.encode()).hexdigest()[:8]}"
    doc = f"User: {user_input}\nAction: {action}\nOutput: {output[:500]}"
    # Change 6: duplicate guard – skip if a near-identical entry already exists
    try:
        if memory_collection.count() > 0:
            chk = memory_collection.query(
                query_texts=[doc], n_results=1, include=["distances"])
            if (chk and chk["distances"] and chk["distances"][0] and
                chk["distances"][0][0] < MEMORY_DUPLICATE_THRESHOLD):
                return  # near-duplicate found, skip
    except Exception:
        pass
    meta = {"user_input": user_input, "action": action, "timestamp": time.time()}
    if metadata:
        meta.update(metadata)
    memory_collection.upsert(documents=[doc], metadatas=[meta], ids=[doc_id])

def recall_memory(query: str, n_results: int = 3) -> list:
    _init_memory()
    # Change 2: filter by relevance threshold; return [] rather than noise
    try:
        count = memory_collection.count()
    except Exception:
        return []
    if count == 0:
        return []
    n = min(n_results, count)
    results = memory_collection.query(
        query_texts=[query], n_results=n,
        include=["documents", "distances"])
    if not results or not results["documents"] or not results["documents"][0]:
        return []
    return [
        doc for doc, dist in zip(results["documents"][0], results["distances"][0])
        if dist <= MEMORY_RELEVANCE_THRESHOLD
    ]


# =============================================================================
# Plugin system
# =============================================================================

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
        mod_name = fname[:-3]
        file_path = os.path.join(PLUGINS_DIR, fname)
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            if hasattr(module, "run") and callable(module.run):
                info = {"run": module.run}
                if hasattr(module, "get_info") and callable(module.get_info):
                    info.update(module.get_info())
                else:
                    info["name"] = mod_name.replace("_", " ").title()
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


# =============================================================================
# Helpers
# =============================================================================

def clean_html(raw: str) -> str:
    raw = re.sub(r'^```[^\n]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    raw = raw.replace('📋 Copy', '')
    raw = re.sub(r'```html', '', raw)
    return raw.strip()

def save_file(content, filename: str) -> str:
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)
    mode = "wb" if isinstance(content, bytes) else "w"
    kw = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
    with open(path, mode, **kw) as f:
        f.write(content)
    return path

def is_video_followup(message: str):
    """
    Detect follow-up commands for a video draft already in session.
    Returns an action string or None:
      "upgrade"  – user wants the full/longer version (optionally with explicit seconds)
      "retry"    – user wants a new draft at the same settings
      "confirm"  – user is happy; clear the session state
    """
    lower = message.lower().strip()
    # "make it 5 seconds" / "make it 10s" / "make it 3 sec" → upgrade
    if re.match(r'^make it\s+\d', lower):
        return "upgrade"
    upgrade_kws = ["make it longer", "make it", "longer", "full version", "make full",
                  "upgrade", "longer version", "full video"]
    retry_kws   = ["try again", "regenerate", "another one", "another",
                  "redo", "new draft", "different one"]
    confirm_kws = ["that's perfect", "thats perfect", "keep it", "perfect",
                  "looks good", "love it", "great", "done"]
    if any(lower == kw or lower.startswith(kw) for kw in upgrade_kws):
        return "upgrade"
    if any(lower == kw or lower.startswith(kw) for kw in retry_kws):
        return "retry"
    if any(lower == kw or lower.startswith(kw) for kw in confirm_kws):
        return "confirm"
    return None

def extract_video_duration(message: str) -> int:
    """
    Parse an explicit duration from the user's request, e.g. "make a 10 second video".
    Handles both singular ("second") and plural ("seconds").
    Returns an int clamped to 1–15, defaulting to 3 if nothing found.
    """
    m = re.search(r'(\d+)\s(?:seconds?|sec|s)\b', message.lower())
    if m:
        return max(1, min(15, int(m.group(1))))
    return 3

_VIDEO_FOLLOWUP_MSG = (
    "\n\n💬 What next?  "
    "· make it longer (or say a length, e.g. 'make it 10 seconds')  "
    "· try again (new draft, ~$0.15)  "
    "· that's perfect to keep it")

# Appended to the prompt only for image-to-video (image_b64 is not None).
# Encourages the model to keep the product physically consistent across frames.
_IMG2VID_CONSISTENCY = (
    " The product in the image must keep its exact shape, proportions, label,"
    " color, and design throughout the entire video. Do not morph, transform,"
    " or change the object into anything else. Keep it physically consistent"
    " and stable from the first frame to the last. Camera movement and"
    " background may change, but the product itself stays identical.")

def is_video_request(message: str) -> bool:
    keywords = [
        "make a video", "make me a video", "create a video", "generate a video",
        "generate video", "generate me a video", "video of", "video clip",
        "video clip of", "video showing", "short video", "create a clip",
        "make a clip", "animate", "animation of", "make an animation",
        "create an animation", "create animation",
    ]
    return any(kw in message.lower() for kw in keywords)

def is_image_request(message: str) -> bool:
    keywords = [
        "draw me ", "draw me a ",
        "generate image", "generate an image", "create an image",
        "generate a picture", "make an image", "make a picture",
        "image of", "picture of", "create a picture", "generate a photo",
        "create a photo", "generate art", "create artwork", "render a ",
        "make a drawing", "sketch of", "illustrate", "visualize",
        "show me an image of", "nano banana", "paint a", "design an image",
    ]
    lower = message.lower()
    return any(kw in lower for kw in keywords)

def _extract_html(text: str):
    """Return the first complete HTML document found in text, or None."""
    m = re.search(r'<!DOCTYPE\s+html[^>]*>.*?</html>', text, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r'<html[^>]*>.*?</html>', text, re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else None


# =============================================================================
# Core tools
# =============================================================================

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
        safe = re.sub(r'[\\/?:"<>|]', "", filename)
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
    # FIX #5 – was raw: refined (undefined variable)
    return {"error": "OpenRouter did not return valid HTML", "raw": response}, token_info

def convert_code_to_html(code, model=None):
    model  = sanitize_model(model or DEFAULT_TEXT_MODEL)
    prompt = (
        f"You are a helpful assistant.\nCode:\n```\n{code}\n```\n\n"
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
        f"You are an expert programmer.\nCode:\n```\n{code}\n```\n\n"
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
    base = (re.sub(r'[\\/\n?:"<>|]', "", filename_base) if filename_base
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


# =============================================================================
# Flask routes
# =============================================================================

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
        "video_model":        VIDEO_MODEL,
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
    vsid = _vid_sid()   # ensure vid_sid is in the cookie before any return

    # ── Video follow-up commands (BEFORE new video check) ────────────
    followup = is_video_followup(message)
    if followup:
        last = _video_sessions.get(vsid)
        if not last:
            return jsonify({"reply": "No video in progress yet — send a video request first."})
        if followup == "confirm":
            _video_sessions.pop(vsid, None)
            return jsonify({"reply": "Glad you liked it! 🎉 The video has been saved."})
        if followup == "upgrade":
            _m = re.search(r'(\d+)\s(?:seconds?|sec|s)\b', message.lower())
            dur = max(1, min(15, int(_m.group(1)))) if _m else 8
            reply_prefix = f"🎬 Generating {dur}-second version…\n\n"
        else:  # retry
            dur = last.get("duration", 3)
            reply_prefix = "🎬 Generating a new draft…\n\n"
        _ar      = last.get("aspect_ratio", "16:9")
        _img_b64 = last.get("image_b64")
        path, err = generate_video_with_openrouter(
            last["prompt"], duration=dur,
            aspect_ratio=_ar, image_b64=_img_b64)
        if err:
            return jsonify({"reply": f"❌ Video generation failed: {err}"})
        video_url = f"/{path.replace(os.sep, '/')}"
        _video_sessions[vsid] = {
            "prompt":       last["prompt"],
            "duration":     dur,
            "resolution":   last.get("resolution", "480p"),
            "aspect_ratio": _ar,
            "image_b64":    _img_b64,
        }
        store_memory(last["prompt"], "video_generation", f"Generated video at {path}")
        return jsonify({"reply": reply_prefix + f"![Generated Video]({video_url})" + _VIDEO_FOLLOWUP_MSG})

    # ── New video request ─────────────────────────────────────────────
    if is_video_request(message):
        dur  = extract_video_duration(message)
        cost = round(dur * 0.05, 2)
        path, err = generate_video_with_openrouter(message, duration=dur)
        if err:
            return jsonify({"reply": f"❌ Video generation failed: {err}"})
        video_url = f"/{path.replace(os.sep, '/')}"
        _video_sessions[vsid] = {"prompt": message, "duration": dur, "resolution": "480p"}
        store_memory(message, "video_generation", f"Generated video at {path}")
        return jsonify({"reply": f"✅ {dur}-second draft (~${cost})!\n\n![Generated Video]({video_url})" + _VIDEO_FOLLOWUP_MSG})

    # Auto-route image requests
    if is_image_request(message):
        _img_enhance_flag = d.get("img_enhance", True)
        _img_p = enhance_image_prompt(message) if _img_enhance_flag else message
        path, err = generate_image_with_openrouter(_img_p)
        if err:
            return jsonify({"reply": f"❌ Image generation failed: {err}"})
        img_url = f"/{path.replace(os.sep, '/')}"
        store_memory(message, "image_generation", f"Generated image at {path}")
        return jsonify({"reply": f"✅ Image generated!\n![Generated Image]({img_url})"})

    # Change 4: extract durable facts + handle "remember that X"
    extract_and_store_facts(message)

    # Change 5: profile always first, then relevant memories
    profile_ctx = get_profile_context()
    relevant    = recall_memory(message, n_results=5)
    mem_block   = ("Relevant context:\n" +
                  "\n".join(f"• {m}" for m in relevant) + "\n\n") if relevant else ""

    if "conv_history" not in session:
        session["conv_history"] = []
    history = session["conv_history"]
    prompt  = _CHAT_SYSTEM + "\n"
    prompt += profile_ctx
    prompt += mem_block
    for msg in history[-10:]:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"User: {message}\nAssistant:"

    reply, token_info = ask_openrouter(prompt, model=model, use_cache=False)

    # Change 3: no longer store every chat reply as a memory (low-value noise).
    # explicit "remember that X" is already stored by extract_and_store_facts.
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
    _img_enhance_flag = d.get("img_enhance", True)
    _prompt = enhance_image_prompt(d["prompt"]) if _img_enhance_flag else d["prompt"]
    path, err = generate_image_with_openrouter(_prompt, d.get("image"))
    if err:
        return jsonify({"error": err})
    return jsonify({"path": path, "status": "success"})

# -----------------------------------------------------------------------------
# POST /generate_video
# Accepts JSON: { "prompt": str, "image": "<base64 data URL or null>",
#                "duration": int (optional), "resolution": str (optional) }
# Used by the frontend for image-to-video (base64 can't ride in a GET URL).
# Also works for text-only video when "image" is absent/null.
# -----------------------------------------------------------------------------

_MAX_IMAGE_B64_BYTES = 8 * 1024 * 1024   # 8 MB raw base64 limit

@app.route("/generate_video", methods=["POST"])
def route_generate_video():
    d      = request.get_json() or {}
    prompt = (d.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    image_b64 = d.get("image") or None
    if image_b64 and len(image_b64.encode()) > _MAX_IMAGE_B64_BYTES:
        return jsonify({"error":
            "Attached image is too large (over 8 MB). "
            "Please resize it to under 2000×2000 px and try again."}), 400

    dur          = max(1, min(15, int(d.get("duration", extract_video_duration(prompt)))))
    resolution   = d.get("resolution", "480p")
    aspect_ratio = d.get("aspect_ratio", "16:9")
    vsid         = _vid_sid()

    path, err = generate_video_with_openrouter(
        prompt, duration=dur, resolution=resolution,
        aspect_ratio=aspect_ratio, image_b64=image_b64)

    if err:
        return jsonify({"error": err}), 500

    video_url = f"/{path.replace(os.sep, '/')}"
    _video_sessions[vsid] = {
        "prompt":       prompt,
        "duration":     dur,
        "resolution":   resolution,
        "aspect_ratio": aspect_ratio,
        "image_b64":    image_b64,   # None for text-only; preserved for follow-ups
    }
    store_memory(prompt, "video_generation", f"Generated video at {path}")
    cost = round(dur * 0.05, 2)
    mode = "image-to-video" if image_b64 else "text-to-video"
    return jsonify({
        "path":      path,
        "video_url": video_url,
        "duration":  dur,
        "mode":      mode,
        "cost":      cost,
        "status":    "success",
    })

@app.route("/get_profile")
def get_profile():
    return jsonify({"profile": load_user_profile()})

@app.route("/update_profile", methods=["POST"])
def route_update_profile():
    d = request.get_json() or {}
    key   = (d.get("key") or "").strip()
    value = (d.get("value") or "").strip()
    if not key or not value:
        return jsonify({"error": "key and value required"}), 400
    update_user_profile(key, value)
    return jsonify({"status": "ok", "profile": load_user_profile()})

@app.route("/clear_profile", methods=["POST"])
def route_clear_profile():
    save_user_profile({})
    return jsonify({"status": "cleared"})

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
    if _HAS_GPU:
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
        "cpu_percent":  _cpu_percent_cache,
        "ram_percent":  mem.percent,
        "ram_used_gb":  round(mem.used  / 1024**3, 1),
        "ram_total_gb": round(mem.total / 1024**3, 1),
        "gpu":          gpu,
    })

@app.route("/favicon.ico")
def favicon():
    return "", 204


# =============================================================================
# Streaming smart agent  (with fallback chain + robust JSON parsing)
# =============================================================================

@app.route("/stream_smart_agent")
def stream_smart_agent():
    request_text = request.args.get("request", "")
    style_guide  = request.args.get("styleGuide", "")
    model        = sanitize_model(request.args.get("model", DEFAULT_TEXT_MODEL))
    img_enhance  = request.args.get("img_enhance", "true").lower() != "false"

    if not request_text:
        return Response(
            f"data: {json.dumps({'error': 'No request provided'})}\n\n",
            mimetype="text/event-stream")

    # Capture vid_sid HERE in the route body (not inside the generator) so
    # the Flask session cookie is saved with vid_sid before streaming starts.
    _vsid = _vid_sid()

    def generate():
        # ── Pending chain video continuation ──────────────────────────
        if re.search(r'\byes\b.*\bvideo\b', request_text.lower()):
            last = _video_sessions.get(_vsid, {})
            if last.get("pending_chain"):
                vid_prompt = last.get("prompt", request_text)
                dur  = last.get("duration", 3)
                cost = round(dur * 0.05, 2)
                yield f"data: {json.dumps({'result': f'Video job submitted ({dur}s, ~${cost})…', 'tool': 'video_generation'})}\n\n"
                path, err = generate_video_with_openrouter(vid_prompt, duration=dur)
                if err:
                    yield f"data: {json.dumps({'error': err})}\n\n"
                else:
                    video_url    = f"/{path.replace(os.sep, '/')}"
                    video_result = (f"{dur}-second draft (~${cost})!\n\n"
                                  f"![Generated Video]({video_url})" + _VIDEO_FOLLOWUP_MSG)
                    _video_sessions[_vsid] = {"prompt": vid_prompt, "duration": dur, "resolution": "480p"}
                    store_memory(vid_prompt, "video_generation", f"Generated video at {path}")
                    yield f"data: {json.dumps({'result': video_result, 'path': path, 'tool': 'video_generation'})}\n\n"
                yield f"data: {json.dumps({'done': True, 'model_used': model})}\n\n"
                return

        # ── Video follow-up commands (BEFORE new video check) ──────────
        followup = is_video_followup(request_text)
        if followup:
            last = _video_sessions.get(_vsid)
            if not last:
                yield f"data: {json.dumps({'result': 'No video in progress yet — send a video request first.', 'tool': 'video_followup'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            if followup == "confirm":
                _video_sessions.pop(_vsid, None)
                yield f"data: {json.dumps({'result': 'Glad you liked it! 🎉 The video has been saved.', 'tool': 'video_followup'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            if followup == "upgrade":
                _m  = re.search(r'(\d+)\s(?:seconds?|sec|s)\b', request_text.lower())
                dur = max(1, min(15, int(_m.group(1)))) if _m else 8
            else:
                dur = last.get("duration", 3)
            _ar      = last.get("aspect_ratio", "16:9")
            _img_b64 = last.get("image_b64")
            wait_msg = f"🎬 Generating {dur}-second version…" if followup == "upgrade" else "🎬 Generating a new draft…"
            yield f"data: {json.dumps({'result': wait_msg, 'tool': 'video_generation'})}\n\n"
            path, err = generate_video_with_openrouter(
                last["prompt"], duration=dur,
                aspect_ratio=_ar, image_b64=_img_b64)
            if err:
                yield f"data: {json.dumps({'error': err})}\n\n"
            else:
                video_url    = f"/{path.replace(os.sep, '/')}"
                video_result = "![Generated Video](" + video_url + ")" + _VIDEO_FOLLOWUP_MSG
                _video_sessions[_vsid] = {
                    "prompt":       last["prompt"],
                    "duration":     dur,
                    "resolution":   last.get("resolution", "480p"),
                    "aspect_ratio": _ar,
                    "image_b64":    _img_b64,
                }
                store_memory(last["prompt"], "video_generation", f"Generated video at {path}")
                yield f"data: {json.dumps({'result': video_result, 'path': path, 'tool': 'video_generation'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # ── New video request ───────────────────────────────────────────
        if is_video_request(request_text):
            dur  = extract_video_duration(request_text)
            cost = round(dur * 0.05, 2)
            yield f"data: {json.dumps({'result': f'🎬 Video job submitted ({dur}s, ~${cost}) — this may take a few minutes…', 'tool': 'video_generation'})}\n\n"
            path, err = generate_video_with_openrouter(request_text, duration=dur)
            if err:
                yield f"data: {json.dumps({'error': err})}\n\n"
            else:
                video_url    = f"/{path.replace(os.sep, '/')}"
                video_result = f"✅ {dur}-second draft (~${cost})!\n\n![Generated Video](" + video_url + ")" + _VIDEO_FOLLOWUP_MSG
                _video_sessions[_vsid] = {"prompt": request_text, "duration": dur, "resolution": "480p"}
                store_memory(request_text, "video_generation", f"Generated video at {path}")
                yield f"data: {json.dumps({'result': video_result, 'path': path, 'tool': 'video_generation'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

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
        extract_and_store_facts(request_text)
        memories     = recall_memory(request_text, n_results=5)
        mem_ctx      = "\n".join(memories) if memories else "No past tasks."
        profile_ctx  = get_profile_context()
        plugins_text = (
            "\n".join(f"- plugin_{k}: {v['description']}"
                     for k, v in get_plugins_info().items())
            or "No plugins available."
        )
        decision_prompt = f"""You are a routing assistant. Pick the right tool for the user's request.
{profile_ctx}
Recent memory:
{mem_ctx}
Available tools — choose EXACTLY one (or an ordered array for multi-step):
- website: USE when asked to build/create/generate a webpage, site, landing page, or HTML. Args: task, filename (optional)
- app_gen: USE when asked to write a script, program, or code file. Args: description, language (python|powershell|bash|batch), filename (optional)
- search: USE ONLY when the user explicitly says "search", "look up", "find online", or asks for live/recent information. Do NOT use for creative or generative tasks. Args: query
- clone: USE when asked to clone or copy a website from a URL. Args: url
- image_generation: USE when asked to draw, generate, create, or render an image or picture. Args: prompt
- refine_html: USE when asked to improve or modify existing HTML. Args: html, instruction
- refine_code: USE when asked to modify existing code. Args: code, instruction
- code_to_html: USE when asked to convert code into an HTML page. Args: code
- chat: USE for everything else — questions, explanations, conversation, opinions, general knowledge. Args: message
Plugins:
{plugins_text}
Rules:
1. Respond with ONLY valid JSON — no prose, no markdown, no backticks.
2. Format: {{"tool": "...", "arguments": {{...}}}} or an array for multi-step.
3. When in doubt, use "chat".
4. Never use "search" for image, video, or creative requests.
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
                    # Change 6: fall back to chat instead of erroring
                    decision = {"tool": "chat", "arguments": {"message": request_text}}
            else:
                # Change 6: fall back to chat instead of erroring
                decision = {"tool": "chat", "arguments": {"message": request_text}}

        decisions = [decision] if isinstance(decision, dict) else decision

        # Step C: cap chain length
        if len(decisions) > MAX_CHAIN_STEPS:
            decisions = decisions[:MAX_CHAIN_STEPS]
            yield f"data: {json.dumps({'warning': f'Chain capped at {MAX_CHAIN_STEPS} steps.'})}\n\n"

        is_chain = len(decisions) > 1
        step_results = []  # Step A: [{tool, summary, label}]

        def _prev_ctx():
            """Step B: build forwarding context block from completed steps."""
            if not step_results:
                return ""
            lines = "\n".join(
                f"[Step {i+1} – {r['tool']}] {r['summary']}"
                for i, r in enumerate(step_results)
            )
            return f"Previous steps context:\n{lines}\n\n"

        for step_idx, action in enumerate(decisions):
            tool = action.get("tool", "")
            args = action.get("arguments", {})
            yield f"data: {json.dumps({'step_start': tool, 'step_index': step_idx})}\n\n"
            err_flag = False
            step_summary = ""

            try:
                # ── chat ───────────────────────────────────────────────
                if tool == "chat":
                    msg = args.get("message", request_text)
                    # facts + memories already fetched in routing phase; reuse to avoid
                    # a second ChromaDB embedding call on the same request text
                    profile_ctx = get_profile_context()
                    mem = memories[:3]  # routing fetched n_results=5; we need 3
                    mem_block = ("Relevant context:\n" +
                               "\n".join(f"• {m}" for m in mem) + "\n\n") if mem else ""
                    prompt = _CHAT_SYSTEM + "\n"
                    prompt += profile_ctx
                    prompt += _prev_ctx()  # Step B: inject prior results
                    prompt += mem_block
                    prompt += f"User: {msg}\nAssistant:"
                    chat_buf = []  # Step F: capture reply for forwarding
                    for chunk in ask_openrouter_stream(prompt, model=model):
                        yield chunk
                        if chunk.startswith("data: "):
                            try:
                                cd = json.loads(chunk[6:])
                                if "token" in cd:
                                    chat_buf.append(cd["token"])
                            except Exception:
                                pass
                    step_summary = "".join(chat_buf)[:600]

                # ── search ─────────────────────────────────────────────
                elif tool == "search":
                    res, _ = web_search_ddg(args.get("query", request_text))
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        yield f"data: {json.dumps({'result': res['results'], 'tool': 'search'})}\n\n"
                        step_summary = res["results"][:600]

                # ── clone ──────────────────────────────────────────────
                elif tool == "clone":
                    url = args.get("url", request_text)
                    if not url.startswith(("http://", "https://")):
                        url = "https://" + url
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    yield f"data: {json.dumps({'result': 'Starting clone…', 'tool': 'clone'})}\n\n"
                    try:
                        idx_path, cdir = clone_website(url, f"cloned_site_{ts}")
                        step_summary = f"Cloned site saved to {cdir}"
                        yield f"data: {json.dumps({'result': f'Clone complete! Saved to {cdir}', 'path': idx_path, 'directory': cdir, 'tool': 'clone'})}\n\n"
                        yield f"data: {json.dumps({'clone_preview': cdir})}\n\n"
                    except Exception as exc:
                        err_flag = True
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n"

                # ── plugin_ ───────────────────────────────────────────
                elif tool.startswith("plugin_"):
                    pname = tool[7:]
                    res = run_plugin(pname, args)
                    key = "error" if "error" in res else "result"
                    if key == "error":
                        err_flag = True
                    else:
                        step_summary = str(res.get("result", ""))[:400]
                        yield f"data: {json.dumps({key: res[key], 'tool': tool})}\n\n"

                # ── website ────────────────────────────────────────────
                elif tool == "website":
                    prev = _prev_ctx()
                    task = (prev + args.get("task", request_text)) if prev else args.get("task", request_text)
                    res, ti = generate_website(task, args.get("filename", "website"), style_guide, model)
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        _p = res["path"]
                        step_summary = f"Website saved to {_p}"
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
                        step_summary = f"Refined HTML saved to {_p}"
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
                        step_summary = f"HTML saved to {_p}"
                        _payload = json.dumps({"result": f"HTML saved to {_p}", "tool": "code_to_html", "path": _p, "token_info": ti})
                        yield f"data: {_payload}\n\n"

                # ── refine_code ────────────────────────────────────────
                elif tool == "refine_code":
                    res, ti = refine_code(args.get("code", ""), args.get("instruction", ""), model)
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        step_summary = "Code refined"
                        _payload = json.dumps({"result": "Code refined", "tool": "refine_code", "refined_code": res["refined_code"], "token_info": ti})
                        yield f"data: {_payload}\n\n"

                # ── app_gen ────────────────────────────────────────────
                elif tool == "app_gen":
                    prev = _prev_ctx()
                    desc = (prev + args.get("description", request_text)) if prev else args.get("description", request_text)
                    res, ti = generate_app(desc, args.get("language", "python"), args.get("filename", ""), model)
                    if res.get("error"):
                        err_flag = True
                        yield f"data: {json.dumps({'error': res['error']})}\n\n"
                    else:
                        _p = res["path"]
                        step_summary = f"App saved to {_p}"
                        _payload = json.dumps({"result": f"App saved to {_p}", "tool": "app_gen", "path": _p, "token_info": ti})
                        yield f"data: {_payload}\n\n"

                # ── image_generation ───────────────────────────────────
                elif tool == "image_generation":
                    if is_chain:
                        yield f"data: {json.dumps({'cost_notice': 'Image generation uses credits (~$0.01–0.04).', 'tool': 'image_generation'})}\n\n"
                    prev = _prev_ctx()
                    img_prompt = args.get("prompt", request_text)
                    if prev:
                        img_prompt = img_prompt + "\n\nAdditional context: " + prev.strip()
                    if img_enhance:
                        img_prompt = enhance_image_prompt(img_prompt)
                    path, err = generate_image_with_openrouter(img_prompt)
                    if err:
                        err_flag = True
                        yield f"data: {json.dumps({'error': err})}\n\n"
                    else:
                        img_url = f"/{path.replace(os.sep, '/')}"
                        img_result = "Image generated!\n![Generated Image](" + img_url + ")"
                        step_summary = f"Image at {img_url}"
                        yield f"data: {json.dumps({'result': img_result, 'path': path, 'tool': 'image_generation'})}\n\n"

                # ── video_generation ───────────────────────────────────
                elif tool == "video_generation":
                    vid_prompt = args.get("prompt", request_text)
                    dur = extract_video_duration(vid_prompt)
                    cost_lo = round(dur * 0.03, 2)
                    cost_hi = round(dur * 0.08, 2)

                    if is_chain:
                        # Cost safety: stop chain, store pending plan, ask user
                        prev = _prev_ctx()
                        if prev:
                            vid_prompt = vid_prompt + "\n\nContext: " + prev.strip()
                        _video_sessions[_vsid] = {
                            "prompt": vid_prompt,
                            "duration": dur,
                            "resolution": "480p",
                            "pending_chain": True,
                        }
                        stop_msg = (
                            f"The next step would generate a {dur}-second video "
                            f"(~${cost_lo}–${cost_hi}). "
                            f"Reply 'yes make the video' to proceed, or ignore to skip."
                        )
                        yield f"data: {json.dumps({'result': stop_msg, 'tool': 'video_pending'})}\n\n"
                        break  # end chain gracefully — don't set err_flag
                    else:
                        # Single step: unchanged behaviour
                        cost = round(dur * 0.05, 2)
                        yield f"data: {json.dumps({'result': f'Video job submitted ({dur}s, ~${cost})…', 'tool': 'video_generation'})}\n\n"
                        path, err = generate_video_with_openrouter(vid_prompt, duration=dur)
                        if err:
                            err_flag = True
                            yield f"data: {json.dumps({'error': err})}\n\n"
                        else:
                            video_url = f"/{path.replace(os.sep, '/')}"
                            video_result = (f"{dur}-second draft (~${cost})!\n\n"
                                          f"![Generated Video]({video_url})" + _VIDEO_FOLLOWUP_MSG)
                            _video_sessions[_vsid] = {"prompt": vid_prompt, "duration": dur, "resolution": "480p"}
                            store_memory(vid_prompt, "video_generation", f"Generated video at {path}")
                            _vp = json.dumps({"result": video_result, "path": path, "tool": "video_generation"})
                            yield f"data: {_vp}\n\n"

                else:
                    # Unknown tool → fall back to chat, never error
                    msg = request_text
                    mem = recall_memory(msg, n_results=3)
                    fb_p = _CHAT_SYSTEM + "\n"
                    fb_p += ("Memory:\n" + "\n".join(mem) + "\n\n" if mem else "")
                    fb_p += f"User: {msg}\nAssistant:"
                    chat_buf = []
                    for chunk in ask_openrouter_stream(fb_p, model=model):
                        yield chunk
                        if chunk.startswith("data: "):
                            try:
                                cd = json.loads(chunk[6:])
                                if "token" in cd:
                                    chat_buf.append(cd["token"])
                            except Exception:
                                pass
                    step_summary = "".join(chat_buf)[:600]

            except Exception as exc:
                err_flag = True
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

            # Step A: record result for forwarding (only on success)
            if not err_flag and step_summary:
                step_results.append({
                    "tool": tool,
                    "summary": step_summary,
                    "label": _TOOL_LABELS.get(tool, tool),
                })

            if err_flag:
                break

        # Step E: chain summary card
        if is_chain and step_results:
            summary_items = [
                {"step": i + 1, "tool": r["tool"], "label": r["label"]}
                for i, r in enumerate(step_results)
            ]
            yield f"data: {json.dumps({'chain_summary': summary_items})}\n\n"

        yield f"data: {json.dumps({'done': True, 'model_used': model})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*62}")
    print(f" AI Studio - Cloud Edition | v2.1 | 07 June 2026")
    print(f" API key set : {'Yes' if OPENROUTER_API_KEY else 'No - set OPENROUTER_API_KEY'}")
    print(f" Default model: {DEFAULT_TEXT_MODEL} (Laguna M.1)")
    print(f" Text chain   : {' -> '.join(FREE_TEXT_MODELS)}")
    print(f" Image chain  : {' -> '.join(IMAGE_MODELS)}")
    print(f" Port         : {port}")
    print(f"{'='*62}\n")
    app.run(debug=False, host="0.0.0.0", port=port)