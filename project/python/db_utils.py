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
      source TEXT,
      symbol_id TEXT,
      keywords TEXT,
      image_path TEXT NOT NULL
    );
    """)

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
