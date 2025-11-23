#!/usr/bin/env python3
"""
Flask server for pictogram -> natural sentence composition using Ollama.
The model (catalan-composer) is already instructed to create proper phrases from primitives.
"""

import os
import json
import threading
import time
from typing import List

import ollama
from flask import Flask, request, jsonify

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "catalan-composer")
CACHE_PATH = os.environ.get("PHRASE_CACHE_PATH", "phrase_cache_ollama.json")

# Ollama generation options
GENERATION_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 25,
}

# Keep model loaded in memory (prevents unloading)
# Options: -1 (keep forever), 0 (unload immediately), or time in seconds (e.g., "5m", "1h")
KEEP_ALIVE = "5m"  # Keep model loaded indefinitely

# Debug mode - enable detailed timing logs
DEBUG_TIMING = os.environ.get("DEBUG_TIMING", "false").lower() == "false"

# ------------------------------
# Cache utilities
# ------------------------------
def load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

_cache_lock = threading.Lock()
_phrase_cache = load_cache(CACHE_PATH)

def cache_key(pictos: List[str], language: str) -> str:
    return language + "|" + " ".join(pictos)

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Pictogram to Sentence API (Ollama)",
        "model": MODEL_NAME,
        "endpoints": {
            "/compose": "POST - Compose sentence from pictos",
            "/health": "GET - Health check",
            "/cache/clear": "POST - Clear cache",
        },
        "example": {"pictos": ["Jo", "Voler", "Sopa"], "language": "ca"},
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        ollama.list()
        return jsonify({
            "status": "healthy",
            "model": MODEL_NAME,
            "cache_size": len(_phrase_cache),
            "ollama": "connected"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "message": "Make sure Ollama is running"
        }), 503

@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    global _phrase_cache
    with _cache_lock:
        _phrase_cache.clear()
        try:
            save_cache(CACHE_PATH, _phrase_cache)
        except Exception as e:
            return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500
    return jsonify({"status": "cache cleared", "cache_size": 0})

@app.route('/compose', methods=['POST'])
def compose_sentence():
    request_start = time.time() if DEBUG_TIMING else None
    try:
        # Timing: Request parsing
        parse_start = time.time() if DEBUG_TIMING else None
        data = request.get_json()
        if not data or 'pictos' not in data:
            return jsonify({"error": "Missing 'pictos' field in request"}), 400

        pictos = data['pictos']
        language = data.get('language', 'ca')

        if not isinstance(pictos, list) or not pictos:
            return jsonify({"error": "'pictos' must be a non-empty list"}), 400

        # Join pictos with spaces
        input_text = " ".join(pictos)
        if DEBUG_TIMING:
            parse_time = time.time() - parse_start
            print(f"⏱️  Request parsing: {parse_time*1000:.2f}ms")

        # Timing: Cache lookup
        cache_start = time.time() if DEBUG_TIMING else None
        key = cache_key(pictos, language)
        with _cache_lock:
            if key in _phrase_cache:
                cached = _phrase_cache[key]
                if DEBUG_TIMING:
                    cache_time = time.time() - cache_start
                    total_time = time.time() - request_start
                    print(f"⏱️  Cache lookup: {cache_time*1000:.2f}ms")
                    print(f"✅ TOTAL (cached): {total_time*1000:.2f}ms\n")
                return jsonify({
                    "input": input_text,
                    "output": cached,
                    "pictos": pictos,
                    "language": language,
                    "cached": True
                })
        if DEBUG_TIMING:
            cache_time = time.time() - cache_start
            print(f"⏱️  Cache lookup (miss): {cache_time*1000:.2f}ms")

        # Timing: Ollama generation
        if DEBUG_TIMING:
            print(f"🔄 Generating for: {input_text}")
        ollama_start = time.time() if DEBUG_TIMING else None
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=input_text,
            options=GENERATION_OPTIONS,
            keep_alive=KEEP_ALIVE  # Keep model loaded in memory
        )
        if DEBUG_TIMING:
            ollama_time = time.time() - ollama_start
            print(f"⏱️  Ollama generation: {ollama_time*1000:.2f}ms ⚠️ BOTTLENECK")
        
        output = response['response'].strip()

        # Timing: Cache save
        save_start = time.time() if DEBUG_TIMING else None
        with _cache_lock:
            _phrase_cache[key] = output
            try:
                save_cache(CACHE_PATH, _phrase_cache)
            except Exception:
                app.logger.exception("Failed to save cache")
        if DEBUG_TIMING:
            save_time = time.time() - save_start
            print(f"⏱️  Cache save: {save_time*1000:.2f}ms")

            total_time = time.time() - request_start
            print(f"✅ TOTAL: {total_time*1000:.2f}ms\n")

        response_data = {
            "input": input_text,
            "output": output,
            "pictos": pictos,
            "language": language,
            "cached": False
        }
        
        # Only include timing data if debug mode is enabled
        if DEBUG_TIMING:
            response_data["timing_ms"] = {
                "total": round(total_time * 1000, 2),
                "ollama": round(ollama_time * 1000, 2),
                "cache_save": round(save_time * 1000, 2),
                "parse": round(parse_time * 1000, 2)
            }
        
        return jsonify(response_data)

    except Exception as e:
        app.logger.exception("compose error")
        return jsonify({"error": str(e)}), 500


def warmup_model():
    """Warm up the model by making a dummy request to keep it loaded"""
    print(f"🔥 Warming up model: {MODEL_NAME}...")
    try:
        start = time.time()
        ollama.generate(
            model=MODEL_NAME,
            prompt="test",
            options={"num_predict": 1},
            keep_alive=KEEP_ALIVE
        )
        warmup_time = time.time() - start
        print(f"✅ Model warmed up in {warmup_time:.2f}s and will stay loaded")
    except Exception as e:
        print(f"⚠️  Failed to warm up model: {e}")
        print("   Make sure Ollama is running: ollama serve")


if __name__ == '__main__':
    # Ensure cache directory exists
    try:
        base = os.path.dirname(CACHE_PATH)
        if base and not os.path.exists(base):
            os.makedirs(base, exist_ok=True)
    except Exception:
        pass

    print(f"Starting Flask server on http://0.0.0.0:5000")
    print(f"Using Ollama model: {MODEL_NAME}")
    print("Model is pre-instructed to compose phrases from primitives")
    
    # Warm up the model before starting the server
    warmup_model()
    
    app.run(host='0.0.0.0', port=5000)