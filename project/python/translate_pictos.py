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


def clean_translation(raw: str) -> str | None:
    if raw is None:
        return None
    candidate = raw.strip().strip(".,;:!?\"'`").split()
    if not candidate:
        return None
    token = candidate[0].lower()
    if token == "error":
        return None
    if not _TOKEN_RE.match(token):
        return None
    return token


def fetch_pending(db_path: str, column: str) -> list[tuple[int, str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT id, word FROM pictos WHERE {column} IS NULL ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows


def translate_missing(db_path: str, column: str, model: str, keep_alive: str = "24h") -> None:
    if column not in VALID_COLUMNS:
        raise ValueError(f"Invalid column {column!r}; expected one of {VALID_COLUMNS}")

    ensure_translation_columns(db_path)
    pending = fetch_pending(db_path, column)
    total = len(pending)
    if total == 0:
        print(f"Nothing to translate: every row already has {column}.")
        return

    print(f"Translating {total} rows into {column} using model '{model}'...")
    start = time.time()
    ok = failed = 0

    for idx, (picto_id, word) in enumerate(pending, 1):
        try:
            resp = ollama.generate(
                model=model,
                prompt=word,
                options={"temperature": 0.2, "num_predict": 16, "num_ctx": 256},
                keep_alive=keep_alive,
            )
            translation = clean_translation(resp.get("response", ""))
        except Exception as exc:
            print(f"  [{idx}/{total}] {word!r} -> ollama error: {exc}")
            failed += 1
            continue

        if translation is None:
            print(f"  [{idx}/{total}] {word!r} -> rejected: {resp.get('response', '')!r}")
            failed += 1
            continue

        update_translation(db_path, picto_id, column, translation)
        ok += 1
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start
            rate = idx / elapsed if elapsed else 0
            print(f"  [{idx}/{total}] {word!r} -> {translation!r}   ({rate:.2f}/s)")

    print(f"\nDone. ok={ok} failed={failed} elapsed={time.time()-start:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column", choices=VALID_COLUMNS, default="word_ca",
                        help="Translation column to populate.")
    parser.add_argument("--model", default="cat-translator",
                        help="Ollama model to use for translation.")
    parser.add_argument("--db", default=DB_PATH,
                        help="Path to pictos.db (defaults to config.DB_PATH).")
    args = parser.parse_args()

    translate_missing(args.db, args.column, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
