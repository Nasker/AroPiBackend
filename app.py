#!/usr/bin/env python3
"""
Flask server for pictogram -> natural sentence composition using Ollama.
The model (catalan-composer) is already instructed to create proper phrases from primitives.
"""

import os
import json
import threading
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
    "num_predict": 50,
}

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
    try:
        data = request.get_json()
        if not data or 'pictos' not in data:
            return jsonify({"error": "Missing 'pictos' field in request"}), 400

        pictos = data['pictos']
        language = data.get('language', 'ca')

        if not isinstance(pictos, list) or not pictos:
            return jsonify({"error": "'pictos' must be a non-empty list"}), 400

        # Join pictos with spaces
        input_text = " ".join(pictos)

        # Check cache first
        key = cache_key(pictos, language)
        with _cache_lock:
            if key in _phrase_cache:
                cached = _phrase_cache[key]
                return jsonify({
                    "input": input_text,
                    "output": cached,
                    "pictos": pictos,
                    "language": language,
                    "cached": True
                })

        # Call Ollama - model is already instructed, just pass the primitives
        print(f"Generating for: {input_text}")
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=input_text,
            options=GENERATION_OPTIONS
        )
        
        output = response['response'].strip()

        # Save to cache
        with _cache_lock:
            _phrase_cache[key] = output
            try:
                save_cache(CACHE_PATH, _phrase_cache)
            except Exception:
                app.logger.exception("Failed to save cache")

        return jsonify({
            "input": input_text,
            "output": output,
            "pictos": pictos,
            "language": language,
            "cached": False
        })

    except Exception as e:
        app.logger.exception("compose error")
        return jsonify({"error": str(e)}), 500


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
    app.run(host='0.0.0.0', port=5000)