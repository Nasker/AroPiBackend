#!/usr/bin/env python3
"""
Populate translation columns (word_ca / word_es) in pictos.db using a
dedicated Ollama model (e.g. cat-translator, es-translator).

Idempotent and resumable: only rows where the target column is NULL are
translated, and each row is committed immediately.

Usage:
    python translate_pictos.py --column word_ca --model cat-translator
    python translate_pictos.py --column word_es --model es-translator
"""

import argparse
import csv
import os
import re
import sqlite3
import sys
import time

import ollama

from config import DB_PATH
from db_utils import ensure_translation_columns, update_translation


VALID_COLUMNS = ("word_ca", "word_es")

# A valid translation is a single token (allow hyphens/apostrophes, common
# in Catalan e.g. "l'aigua"). Reject multi-word outputs, punctuation noise,
# or the sentinel "ERROR".
_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÿ'\-]+$")

# Grammar type hints appended to the prompt on retry, to help the translator
# disambiguate (e.g. "he" as a pronoun, "up" as an adverb).
_GRAMMAR_HINT_CA = {
    "pronoun":   "pronom",
    "verb":      "verb (infinitiu)",
    "noun":      "substantiu",
    "adjective": "adjectiu (masculí singular)",
    "adverb":    "adverbi",
}


def clean_translation(raw: str) -> str | None:
    """Extract a single canonical token from the model output.

    Accepts outputs like:
      - "aigua"                  -> "aigua"
      - "ells/elles"             -> "ells"     (first of gendered forms)
      - "agafar, atrapar, captar"-> "agafar"   (first of synonyms)
      - "enfadat/ada"            -> "enfadat"
      - "ERROR"                  -> None
    """
    if raw is None:
        return None
    text = raw.strip().strip(".,;:!?\"'`")
    if not text:
        return None
    # Split on any of: whitespace, slash, comma, semicolon, parentheses.
    # Take the first non-empty piece.
    pieces = re.split(r"[\s/,;()]+", text)
    pieces = [p for p in pieces if p]
    if not pieces:
        return None
    token = pieces[0].lower()
    if token == "error":
        return None
    if not _TOKEN_RE.match(token):
        return None
    return token


def load_overrides(path: str) -> dict[str, str]:
    """Load a CSV of manual (word -> translation) overrides."""
    overrides: dict[str, str] = {}
    if not path or not os.path.exists(path):
        return overrides
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("word") or "").strip().lower()
            val = (row.get("word_ca") or row.get("word_es") or "").strip()
            if key and val:
                overrides[key] = val
    return overrides


def fetch_pending(db_path: str, column: str) -> list[tuple[int, str, str | None]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, word, grammar_type FROM pictos WHERE {column} IS NULL ORDER BY id"
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def _ask_model(model: str, prompt: str, keep_alive: str) -> tuple[str | None, str]:
    """Call Ollama and return (cleaned_translation, raw_response)."""
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.2, "num_predict": 16, "num_ctx": 256},
        keep_alive=keep_alive,
    )
    raw = resp.get("response", "")
    return clean_translation(raw), raw


def translate_missing(
    db_path: str,
    column: str,
    model: str,
    overrides_path: str | None = None,
    keep_alive: str = "24h",
) -> None:
    if column not in VALID_COLUMNS:
        raise ValueError(f"Invalid column {column!r}; expected one of {VALID_COLUMNS}")

    ensure_translation_columns(db_path)
    overrides = load_overrides(overrides_path) if overrides_path else {}
    if overrides:
        print(f"Loaded {len(overrides)} manual overrides from {overrides_path}")

    pending = fetch_pending(db_path, column)
    total = len(pending)
    if total == 0:
        print(f"Nothing to translate: every row already has {column}.")
        return

    print(f"Translating {total} rows into {column} using model '{model}'...")
    start = time.time()
    ok = failed = overridden = 0

    for idx, (picto_id, word, grammar_type) in enumerate(pending, 1):
        # 1. Manual override wins.
        if word.lower() in overrides:
            update_translation(db_path, picto_id, column, overrides[word.lower()])
            overridden += 1
            ok += 1
            print(f"  [{idx}/{total}] {word!r} -> {overrides[word.lower()]!r}   (override)")
            continue

        # 2. First attempt: bare word.
        try:
            translation, raw = _ask_model(model, word, keep_alive)
        except Exception as exc:
            print(f"  [{idx}/{total}] {word!r} -> ollama error: {exc}")
            failed += 1
            continue

        # 3. Retry with a grammar-type hint if the first attempt failed.
        if translation is None and grammar_type in _GRAMMAR_HINT_CA:
            hint = _GRAMMAR_HINT_CA[grammar_type]
            retry_prompt = f"{word} ({hint})"
            try:
                translation, raw = _ask_model(model, retry_prompt, keep_alive)
            except Exception as exc:
                print(f"  [{idx}/{total}] {word!r} -> retry ollama error: {exc}")
                failed += 1
                continue

        if translation is None:
            print(f"  [{idx}/{total}] {word!r} -> rejected: {raw!r}")
            failed += 1
            continue

        update_translation(db_path, picto_id, column, translation)
        ok += 1
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start
            rate = idx / elapsed if elapsed else 0
            print(f"  [{idx}/{total}] {word!r} -> {translation!r}   ({rate:.2f}/s)")

    print(
        f"\nDone. ok={ok} (overridden={overridden}) failed={failed} "
        f"elapsed={time.time()-start:.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column", choices=VALID_COLUMNS, default="word_ca",
                        help="Translation column to populate.")
    parser.add_argument("--model", default="cat-translator",
                        help="Ollama model to use for translation.")
    parser.add_argument("--db", default=DB_PATH,
                        help="Path to pictos.db (defaults to config.DB_PATH).")
    parser.add_argument("--overrides", default="../input/translation_overrides_ca.csv",
                        help="CSV of manual (word,word_ca) overrides. "
                             "Applied before querying the model.")
    args = parser.parse_args()

    translate_missing(args.db, args.column, args.model, overrides_path=args.overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
