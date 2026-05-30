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
IMAGE_MODEL = "google/gemini-2.5-flash-image-preview:free"   # ✅ working free image model

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
        "model": "google/gemini-2.5-flash-image-preview:free",   # ✅ fixed
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

# ----- rest of the file unchanged (keep your existing functions below) -----
# [The remaining part of your code (store_memory, recall_memory, plugins, routes, etc.) 
# should be exactly as in your current working file – just ensure the above changes are made.]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 AI Studio starting on port {port}")
    print(f"✅ Image generation enabled with {IMAGE_MODEL}")
    app.run(debug=False, host="0.0.0.0", port=port)