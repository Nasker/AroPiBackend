#!/usr/bin/env python3
"""
Flask server for pictogram -> natural sentence composition.

Runtime composition only: reads pre-generated subject+verb conjugations
from `pictogram_phrases.db` and translates remaining pictos via
`pictos.db`. No LLM at request time.
"""

import json
import os
import sqlite3
import sys
import threading
import time
from typing import List

from flask import Flask, jsonify, request, send_from_directory

# Make project/python importable so we can reuse PhraseComposer.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_PY = os.path.join(REPO_ROOT, "project", "python")
sys.path.insert(0, PROJECT_PY)

from composer import (  # noqa: E402
    HeadNotFoundError,
    PhraseComposer,
    SUPPORTED_LANGUAGES,
    UnknownPictoError,
)

# ------------------------------
# Config
# ------------------------------
PHRASES_DB = os.environ.get(
    "PHRASES_DB", os.path.join(REPO_ROOT, "pictogram_phrases.db")
)
PICTOS_DB = os.environ.get(
    "PICTOS_DB", os.path.join(REPO_ROOT, "project", "output", "pictos.db")
)
PICTOS_IMAGES_DIR = os.environ.get(
    "PICTOS_IMAGES_DIR", os.path.join(REPO_ROOT, "project", "output", "png")
)
BUNDLES_DIR = os.environ.get(
    "BUNDLES_DIR", os.path.join(REPO_ROOT, "bundles")
)
CACHE_PATH = os.environ.get(
    "PHRASE_CACHE_PATH", os.path.join(REPO_ROOT, "phrase_cache_compose.json")
)
DEBUG_TIMING = os.environ.get("DEBUG_TIMING", "false").lower() == "true"

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


def save_cache(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


_cache_lock = threading.Lock()
_phrase_cache = load_cache(CACHE_PATH)


def cache_key(pictos: List[str], language: str) -> str:
    return language + "|" + " ".join(pictos)


# ------------------------------
# Composer (module-level singleton)
# ------------------------------
composer = PhraseComposer(PHRASES_DB, PICTOS_DB)


# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)


GRAMMAR_TO_ROLE = {
    "pronoun":   "subject",
    "verb":      "verb",
    "noun":      "object",
    "adjective": "modifier",
    "adverb":    "modifier",
}


def _image_basename(image_path: str | None) -> str | None:
    if not image_path:
        return None
    # DB paths mix slashes (e.g. "../output/images\\I.svg"); take basename.
    name = image_path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    return f"{stem}.png"


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Pictogram to Sentence API (compositional)",
        "endpoints": {
            "/compose":                    "POST - Compose sentence from pictos",
            "/pictos":                     "GET  - Catalogue, ?language=ca|es",
            "/pictos/image/<id>":          "GET  - Picto image (PNG)",
            "/bundle/latest":              "GET  - Offline bundle version pointer",
            "/bundle/<version>/<file>":    "GET  - Download offline bundle zip",
            "/health":                     "GET  - Health check",
            "/cache/clear":                "POST - Clear phrase cache",
        },
        "example": {"pictos": ["I", "eat", "apple"], "language": "ca"},
    })


@app.route("/pictos", methods=["GET"])
def list_pictos():
    language = request.args.get("language", "ca")
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": f"Unsupported language {language!r}; expected one of {list(SUPPORTED_LANGUAGES)}"
        }), 400

    column = f"word_{language}"
    conn = sqlite3.connect(PICTOS_DB)
    cur = conn.cursor()
    cur.execute(
        f"SELECT word, grammar_type, {column}, image_path "
        "FROM pictos ORDER BY grammar_type, word"
    )
    rows = cur.fetchall()
    conn.close()

    items = []
    for word, grammar_type, translation, image_path in rows:
        basename = _image_basename(image_path)
        items.append({
            "id": word,
            "word": word,
            "translation": translation,
            "language": language,
            "grammar_type": grammar_type,
            "role": GRAMMAR_TO_ROLE.get(grammar_type),
            "image_url": f"/pictos/image/{basename}" if basename else None,
        })
    return jsonify({"language": language, "count": len(items), "pictos": items})


@app.route("/pictos/image/<path:filename>", methods=["GET"])
def get_picto_image(filename: str):
    return send_from_directory(PICTOS_IMAGES_DIR, filename)


# ------------------------------
# Offline bundle distribution
# ------------------------------
@app.route("/bundle/latest", methods=["GET"])
def bundle_latest():
    """Mobile clients poll this to check for an updated offline bundle.

    Compare the returned `version` against the client's local version; if
    newer, download `url` (serve via /bundle/<version>/<filename>).
    """
    pointer = os.path.join(BUNDLES_DIR, "latest.json")
    if not os.path.exists(pointer):
        return jsonify({
            "error": "No bundle has been built yet.",
            "hint": "Run `python make_bundle.py` on the server.",
        }), 404
    try:
        with open(pointer, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception as exc:
        return jsonify({"error": f"Corrupt latest.json: {exc}"}), 500


@app.route("/bundle/<version>/<path:filename>", methods=["GET"])
def bundle_download(version: str, filename: str):
    # `version` and `filename` are sanitised by Flask's URL converter; we
    # additionally constrain the served directory to BUNDLES_DIR/<version>.
    bundle_dir = os.path.join(BUNDLES_DIR, version)
    if not os.path.isdir(bundle_dir):
        return jsonify({"error": f"Unknown bundle version {version!r}"}), 404
    return send_from_directory(bundle_dir, filename, as_attachment=True)


@app.route("/health", methods=["GET"])
def health():
    try:
        # Force a load so we can report counts & surface any DB issue early.
        composer._load_heads()
        return jsonify({
            "status": "healthy",
            "heads_loaded": len(composer._heads),
            "cache_size": len(_phrase_cache),
            "phrases_db": PHRASES_DB,
            "pictos_db": PICTOS_DB,
        })
    except Exception as exc:
        return jsonify({"status": "unhealthy", "error": str(exc)}), 503


@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    with _cache_lock:
        _phrase_cache.clear()
        try:
            save_cache(CACHE_PATH, _phrase_cache)
        except Exception as exc:
            return jsonify({"error": f"Failed to clear cache: {exc}"}), 500
    return jsonify({"status": "cache cleared", "cache_size": 0})


@app.route("/compose", methods=["POST"])
def compose_sentence():
    request_start = time.time() if DEBUG_TIMING else None

    data = request.get_json(silent=True)
    if not data or "pictos" not in data:
        return jsonify({"error": "Missing 'pictos' field in request"}), 400

    pictos = data["pictos"]
    language = data.get("language", "ca")

    if not isinstance(pictos, list) or len(pictos) < 2:
        return jsonify({
            "error": "'pictos' must be a list of at least 2 items (subject, verb, ...)"
        }), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": f"Unsupported language {language!r}; expected one of {list(SUPPORTED_LANGUAGES)}"
        }), 400

    input_text = " ".join(pictos)
    key = cache_key(pictos, language)

    # Cache hit
    with _cache_lock:
        cached = _phrase_cache.get(key)
    if cached is not None:
        if DEBUG_TIMING:
            print(f"✅ cached in {(time.time()-request_start)*1000:.2f}ms")
        return jsonify({
            "input": input_text,
            "output": cached,
            "pictos": pictos,
            "language": language,
            "cached": True,
        })

    # Compose
    try:
        output = composer.compose(pictos, language=language)
    except HeadNotFoundError as exc:
        return jsonify({
            "error": str(exc),
            "hint": "Re-run batch_generate.py to populate this pair.",
            "pictos": pictos,
            "language": language,
        }), 404
    except UnknownPictoError as exc:
        return jsonify({
            "error": str(exc),
            "hint": "Ensure the picto exists in pictos.db with a translation for this language.",
            "pictos": pictos,
            "language": language,
        }), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        app.logger.exception("compose error")
        return jsonify({"error": str(exc)}), 500

    # Cache
    with _cache_lock:
        _phrase_cache[key] = output
        try:
            save_cache(CACHE_PATH, _phrase_cache)
        except Exception:
            app.logger.exception("Failed to save cache")

    if DEBUG_TIMING:
        print(f"✅ composed in {(time.time()-request_start)*1000:.2f}ms")

    return jsonify({
        "input": input_text,
        "output": output,
        "pictos": pictos,
        "language": language,
        "cached": False,
    })


if __name__ == "__main__":
    print(f"Starting Flask server on http://0.0.0.0:5000")
    print(f"Phrases DB: {PHRASES_DB}")
    print(f"Pictos DB:  {PICTOS_DB}")
    # Warm caches so the first request is fast.
    composer._load_heads()
    for lang in SUPPORTED_LANGUAGES:
        try:
            composer._load_tails(lang)
        except Exception:
            pass
    print(f"Loaded {len(composer._heads)} pre-generated heads")
    app.run(host="0.0.0.0", port=5000)
