#!/usr/bin/env python3
"""
Pre-generate conjugated subject+verb "heads" in each target language.

For every (pronoun_en, verb_en) pair in pictos.db:
  - look up the Catalan translations (word_ca)
  - ask the cat-composer Ollama model to produce a correctly conjugated
    two-word phrase
  - store it in pictogram_phrases.db keyed by the English picto IDs and
    the target language

Keys remain English so the mobile app (and the runtime composer) can
address rows regardless of the user's UI language.

Features:
  - Checkpoint file for resume (batch_checkpoint.json).
  - DB-level duplicate skip (safety net against stale checkpoints).
  - Rolling-window ETA.
  - Persistent keep_alive + small num_ctx for throughput.
"""

import json
import os
import sqlite3
import sys
import time
from collections import deque
from typing import Dict, List, Tuple

import ollama

# Make project/python importable so we can reuse CombinationGenerator.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_PY = os.path.join(REPO_ROOT, "project", "python")
sys.path.insert(0, PROJECT_PY)

from combination_generator import CombinationGenerator  # noqa: E402


# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME = "cat-composer"
BATCH_SIZE = 10  # Commit + checkpoint every N items (lower = safer resume)
KEEP_ALIVE = "24h"

PICTOS_DB = os.path.join(REPO_ROOT, "project", "output", "pictos.db")
OUTPUT_DB = os.path.join(REPO_ROOT, "pictogram_phrases.db")
CHECKPOINT_FILE = os.path.join(REPO_ROOT, "batch_checkpoint.json")

TARGET_LANGUAGES = ["ca"]  # add "es" later once word_es is populated

GENERATION_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 25,
    "num_ctx": 256,
}


# ------------------------------
# DB setup
# ------------------------------
def init_phrases_db() -> sqlite3.Connection:
    conn = sqlite3.connect(OUTPUT_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS phrases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pictos TEXT NOT NULL,
            language TEXT NOT NULL,
            output TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pictos, language)
        )
    """)
    conn.commit()
    return conn


def load_translation_map(language: str) -> Dict[str, str]:
    """Return {english_word: translation} for the given target language."""
    column = f"word_{language}"
    conn = sqlite3.connect(PICTOS_DB)
    cur = conn.cursor()
    cur.execute(f"SELECT word, {column} FROM pictos WHERE {column} IS NOT NULL")
    mapping = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return mapping


# ------------------------------
# Checkpoint helpers
# ------------------------------
def load_checkpoint() -> dict:
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"processed": 0, "total": 0}


def save_checkpoint(processed: int, total: int) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed": processed, "total": total}, f)


# ------------------------------
# Job building
# ------------------------------
def build_jobs() -> List[Tuple[str, str, str, str, str]]:
    """Return a deterministic list of jobs.

    Each job: (pronoun_en, verb_en, language, pronoun_local, verb_local)
    Pairs whose translations are missing are silently skipped — they will
    show up in `get_stats()` as untranslated rows.
    """
    pairs = CombinationGenerator(PICTOS_DB).generate_subject_verb_combinations()
    jobs: List[Tuple[str, str, str, str, str]] = []
    for language in TARGET_LANGUAGES:
        translations = load_translation_map(language)
        for pronoun_en, verb_en in pairs:
            p_loc = translations.get(pronoun_en)
            v_loc = translations.get(verb_en)
            if not p_loc or not v_loc:
                continue
            jobs.append((pronoun_en, verb_en, language, p_loc, v_loc))
    return jobs


# ------------------------------
# Main batch loop
# ------------------------------
def batch_generate(jobs: List[Tuple[str, str, str, str, str]], conn: sqlite3.Connection) -> None:
    checkpoint = load_checkpoint()
    start_idx = checkpoint["processed"] if checkpoint.get("total") == len(jobs) else 0
    total = len(jobs)

    cursor = conn.cursor()
    cursor.execute("SELECT pictos, language FROM phrases")
    already_done = {(r[0], r[1]) for r in cursor.fetchall()}

    print("🚀 Starting batch generation")
    print(f"📊 Total jobs:                {total}")
    print(f"✅ Checkpoint processed:      {start_idx}")
    print(f"💾 Already in DB:             {len(already_done)}")
    print(f"⏳ Remaining (at most):       {total - start_idx}\n")

    start_time = time.time()
    recent_durations = deque(maxlen=max(BATCH_SIZE * 5, 50))
    processed_in_run = 0
    errors = 0

    for i in range(start_idx, total):
        pronoun_en, verb_en, language, p_loc, v_loc = jobs[i]
        key = f"{pronoun_en} {verb_en}"

        if (key, language) in already_done:
            continue

        prompt = f"{p_loc} {v_loc}"
        try:
            iter_start = time.time()
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options=GENERATION_OPTIONS,
                keep_alive=KEEP_ALIVE,
            )
            output = response["response"].strip()
            recent_durations.append(time.time() - iter_start)
            processed_in_run += 1

            if not output or output.upper().startswith("ERROR"):
                print(f"⚠️  [{i+1}/{total}] {key!r} [{language}] "
                      f"prompt={prompt!r} -> ERROR, skipped")
                errors += 1
                continue

            cursor.execute(
                "INSERT OR REPLACE INTO phrases (pictos, language, output) "
                "VALUES (?, ?, ?)",
                (key, language, output),
            )

            if (i + 1) % BATCH_SIZE == 0:
                conn.commit()
                save_checkpoint(i + 1, total)

                elapsed = time.time() - start_time
                avg_rate = processed_in_run / elapsed if elapsed > 0 else 0
                recent_rate = (len(recent_durations) / sum(recent_durations)
                               if recent_durations else avg_rate)
                remaining = (total - i - 1) / recent_rate if recent_rate > 0 else 0

                print(f"✅ {i+1}/{total} ({(i+1)/total*100:.1f}%) | "
                      f"Recent: {recent_rate:.2f}/s | Avg: {avg_rate:.2f}/s | "
                      f"ETA: {remaining/60:.1f}min | "
                      f"Last [{language}] {prompt!r} → {output!r}")

        except Exception as exc:
            print(f"❌ Error processing {prompt!r} [{language}]: {exc}")
            errors += 1
            continue

    conn.commit()
    save_checkpoint(total, total)

    elapsed = time.time() - start_time
    print("\n🎉 Batch generation complete!")
    print(f"⏱️  Total time:     {elapsed/60:.1f} minutes")
    print(f"📊 Jobs:           {total}")
    print(f"⚠️  Errors:         {errors}")
    print(f"💾 Database:       {OUTPUT_DB}")


def warmup_model() -> bool:
    print(f"🔥 Warming up model: {MODEL_NAME}...")
    try:
        ollama.generate(
            model=MODEL_NAME,
            prompt="jo voler",
            options={"num_predict": 1, "num_ctx": 256},
            keep_alive=KEEP_ALIVE,
        )
        print("✅ Model ready!\n")
        return True
    except Exception as exc:
        print(f"❌ Warmup failed: {exc}")
        print("   Make sure Ollama is running and "
              f"`ollama create {MODEL_NAME} -f Modelfile_cat_composer` has been run.")
        return False


def main() -> int:
    if not os.path.exists(PICTOS_DB):
        print(f"❌ pictos.db not found at {PICTOS_DB}")
        return 1

    if not warmup_model():
        return 1

    conn = init_phrases_db()
    try:
        print("📝 Building job list from pictos.db...")
        jobs = build_jobs()
        if not jobs:
            print("❌ No jobs to run. Have you populated word_ca "
                  "via translate_pictos.py?")
            return 1
        print(f"✅ Built {len(jobs)} jobs "
              f"({len(TARGET_LANGUAGES)} language(s))\n")

        batch_generate(jobs, conn)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
