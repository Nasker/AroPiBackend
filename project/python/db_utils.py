import sqlite3
import os

def create_schema(path):
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE pictos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      word TEXT NOT NULL,
      language TEXT NOT NULL,
      grammar_type TEXT,
      word_ca TEXT,
      word_es TEXT,
      source TEXT,
      symbol_id TEXT,
      keywords TEXT,
      image_path TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()

def ensure_translation_columns(db_path):
    """Idempotently add word_ca / word_es columns to an existing pictos.db."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(pictos)")
    existing = {row[1] for row in cur.fetchall()}
    for col in ("word_ca", "word_es"):
        if col not in existing:
            cur.execute(f"ALTER TABLE pictos ADD COLUMN {col} TEXT")
    conn.commit()
    conn.close()

def insert_picto(db_path, entry):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
      INSERT INTO pictos (word, language, grammar_type, source, symbol_id, keywords, image_path)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        entry["word"],
        entry["language"],
        entry.get("grammar_type"),
        entry.get("source"),
        entry.get("symbol_id"),
        entry.get("keywords"),
        entry["image_path"]
    ))

    conn.commit()
    conn.close()

def update_translation(db_path, picto_id, column, value):
    """Set a translation column (word_ca / word_es) for a given picto row."""
    if column not in ("word_ca", "word_es"):
        raise ValueError(f"Unsupported translation column: {column}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"UPDATE pictos SET {column} = ? WHERE id = ?", (value, picto_id))
    conn.commit()
    conn.close()
