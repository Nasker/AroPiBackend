#!/usr/bin/env python3
"""
Flask-based backend for pictogram -> natural sentence composition.
Improved version with:
 - input normalization
 - language-specific few-shot prompts
 - deterministic decoding (beam search)
 - output post-processing
 - persistent cache (JSON) to avoid repeated LLM calls
 - simple health & root endpoints

Requirements:
 pip install torch transformers flask sentencepiece

Run:
 python compose_server.py

Note: This version forces CPU (useful for machines without supported CUDA GPU).
 If you have a compatible GPU and want to use it, change DEVICE to cuda when available.
"""

import os
import json
import re
import threading
from functools import lru_cache
from typing import List

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "bigscience/mt0-large")
CACHE_PATH = os.environ.get("PHRASE_CACHE_PATH", "phrase_cache.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force CPU if MX230 or incompatible GPU; you can override by setting TORCH_DEVICE env var
if os.environ.get("FORCE_CPU", "1") == "1":
    DEVICE = torch.device("cpu")

# Generation settings (deterministic)
GENERATION_KWARGS = dict(
    max_new_tokens=40,
    num_beams=4,
    do_sample=False,  # deterministic
    early_stopping=True,
    no_repeat_ngram_size=3,
    repetition_penalty=1.0,
)

# ------------------------------
# Utilities
# ------------------------------

def load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # corrupted cache -> reset
            return {}
    return {}

def save_cache(path: str, data: dict):
    # atomic write
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

_cache_lock = threading.Lock()
_phrase_cache = load_cache(CACHE_PATH)

# Normalize pictogram tokens into canonical lemmas
def normalize_pictos(pictos: List[str]) -> List[str]:
    out = []
    for w in pictos:
        if not isinstance(w, str):
            continue
        # Basic cleanup: remove extra punctuation
        token = re.sub(r"[^\wÀ-ÿ'-]", " ", w, flags=re.UNICODE).strip()
        token = re.sub(r"\s+", " ", token)
        if not token:
            continue
        # Heuristics: if it looks like a name (capitalized), keep capitalization
        if token[0].isupper():
            out.append(token)
            continue
        token = token.lower()
        # Quick replacements / common lemma fixes (extend as needed)
        repl = {
            "yo": "yo",
            "jo": "jo",
            "tu": "tu",
            "vull": "vull",
            "voler": "voler",
            "querer": "querer",
            "aigua": "aigua",
            "agua": "agua",
            "menjar": "menjar",
            "comer": "comer",
            "ser": "ser",
            "estar": "estar",
            "tenir": "tenir",
            "tener": "tener",
        }
        out.append(repl.get(token, token))
    return out

# Clean generated output for consistent punctuation/capitalization
def clean_output(text: str, language: str = "es") -> str:
    if not text:
        return text
    t = text.strip()
    # Remove leading labels if model echoes them
    t = re.sub(r"(?i)^(frase|sentence)\s*[:.-]*\s*", "", t).strip()
    # Remove stray tokens like <extra_id_0>
    t = re.sub(r"<extra_id_\d+>", "", t)
    # Strip again
    t = t.strip()
    # Capitalize first letter in a unicode-safe way
    if len(t) > 0 and t[0].islower():
        t = t[0].upper() + t[1:]
    # Ensure punctuation at the end
    if not t.endswith((".", "?", "!", "…")):
        t = t + "."
    return t

# Deterministic key generator for caching
def cache_key(pictos: List[str], language: str) -> str:
    # canonical key: language + '|' + tokens joined
    return language + "|" + "|".join(pictos)

# ------------------------------
# Few-shot prompts per language
# Keep examples short and child-friendly
# ------------------------------
FEW_SHOT_PROMPTS = {
    "es": (
        "Convierte las palabras en una frase corta y natural en español.\n\n"
        "Ejemplos:\n"
        "Palabras: yo querer agua\nFrase: Yo quiero agua.\n\n"
        "Palabras: ella tener hambre\nFrase: Ella tiene hambre.\n\n"
        "Palabras: yo ser Oscar\nFrase: Yo soy Oscar.\n\n"
        "Ahora hazlo tú:\nPalabras: {root_sequence}\nFrase:"
    ),
    "ca": (
        "Converteix les paraules en una frase curta i natural en català.\n\n"
        "Exemples:\n"
        "Paraules: jo voler aigua\nFrase: Jo vull aigua.\n\n"
        "Paraules: ella tenir gana\nFrase: Ella té gana.\n\n"
        "Paraules: jo ser Oscar\nFrase: Jo sóc l'Oscar.\n\n"
        "Ara fes-ho tu:\nParaules: {root_sequence}\nFrase:"
    ),
    "en": (
        "Convert the words into a short, natural English sentence.\n\n"
        "Examples:\n"
        "Words: I want water\nSentence: I want water.\n\n"
        "Words: she be hungry\nSentence: She is hungry.\n\n"
        "Words: I be Oscar\nSentence: I am Oscar.\n\n"
        "Now do this:\nWords: {root_sequence}\nSentence:"
    ),
}

# ------------------------------
# Model loading
# ------------------------------
print(f"Loading {MODEL_NAME} on {DEVICE}...")
# Use AutoTokenizer / AutoModel to support mt0 / mt5 variants
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Pictogram to Sentence API",
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "endpoints": {
            "/compose": "POST - Compose sentence from pictos",
            "/health": "GET - Health check",
        },
        "example": {"pictos": ["jo", "voler", "aigua"], "language": "ca"},
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": MODEL_NAME, "device": str(DEVICE), "cache_size": len(_phrase_cache)})

@app.route('/compose', methods=['POST'])
def compose_sentence():
    try:
        data = request.get_json()
        if not data or 'pictos' not in data:
            return jsonify({"error": "Missing 'pictos' field in request"}), 400

        pictos = data['pictos']
        language = data.get('language', 'es')

        if not isinstance(pictos, list) or not pictos:
            return jsonify({"error": "'pictos' must be a non-empty list"}), 400

        # Normalize pictograms
        normalized = normalize_pictos(pictos)
        root_sequence = " ".join(normalized)

        # Check cache first
        key = cache_key(normalized, language)
        with _cache_lock:
            if key in _phrase_cache:
                cached = _phrase_cache[key]
                return jsonify({"input": root_sequence, "output": cached, "pictos": pictos, "language": language, "cached": True})

        # Build few-shot prompt
        template = FEW_SHOT_PROMPTS.get(language, FEW_SHOT_PROMPTS['es'])
        prompt = template.format(root_sequence=root_sequence)
        app.logger.debug(f"Prompt: {prompt}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)

        # Generate (deterministic)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GENERATION_KWARGS
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned = clean_output(generated_text, language=language)

        # Save to cache
        with _cache_lock:
            _phrase_cache[key] = cleaned
            # Persist cache to disk
            try:
                save_cache(CACHE_PATH, _phrase_cache)
            except Exception:
                app.logger.exception("Failed to save cache")

        return jsonify({"input": root_sequence, "output": cleaned, "pictos": pictos, "language": language, "cached": False})

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

    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
