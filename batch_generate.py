#!/usr/bin/env python3
"""
Batch processing script to pre-generate pictogram combinations.
Generates phrases for all combinations and stores them in a database.
"""

import json
import time
import ollama
from itertools import product
from typing import List, Dict
import sqlite3

# Configuration
MODEL_NAME = "catalan-composer"
BATCH_SIZE = 100  # Process in batches to show progress
OUTPUT_DB = "pictogram_phrases.db"
CHECKPOINT_FILE = "batch_checkpoint.json"

# Ollama options optimized for batch processing
GENERATION_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 25,
}

# Example pictogram vocabulary (expand this!)
SUBJECTS = ["jo", "tu", "ell", "ella", "nosaltres", "yo", "tú", "él", "ella", "nosotros"]
VERBS = ["voler", "tenir", "ser", "estar", "anar", "querer", "tener", "ir", "comer", "beber"]
OBJECTS = ["aigua", "sopa", "pa", "llet", "menjar", "agua", "comida", "pan", "leche"]
MODIFIERS = ["molt", "poc", "ara", "després", "mucho", "poco", "ahora", "después"]

def init_database():
    """Initialize SQLite database for storing generated phrases"""
    conn = sqlite3.connect(OUTPUT_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS phrases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pictos TEXT NOT NULL,
            language TEXT NOT NULL,
            output TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pictos, language)
        )
    ''')
    conn.commit()
    return conn

def load_checkpoint():
    """Load checkpoint to resume interrupted batch"""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"processed": 0, "total": 0}

def save_checkpoint(processed, total):
    """Save checkpoint for resuming"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"processed": processed, "total": total}, f)

def generate_combinations() -> List[List[str]]:
    """Generate all pictogram combinations"""
    combinations = []
    
    # 2-word combinations: subject + verb
    for subj, verb in product(SUBJECTS, VERBS):
        combinations.append([subj, verb])
    
    # 3-word combinations: subject + verb + object
    for subj, verb, obj in product(SUBJECTS, VERBS, OBJECTS):
        combinations.append([subj, verb, obj])
    
    # 4-word combinations: subject + verb + object + modifier
    for subj, verb, obj, mod in product(SUBJECTS, VERBS, OBJECTS, MODIFIERS):
        combinations.append([subj, verb, obj, mod])
    
    return combinations

def detect_language(pictos: List[str]) -> str:
    """Detect language from pictograms"""
    catalan_words = {"jo", "voler", "tenir", "aigua", "sopa", "molt", "poc"}
    spanish_words = {"yo", "querer", "tener", "agua", "comida", "mucho", "poco"}
    
    pictos_set = set(p.lower() for p in pictos)
    
    if pictos_set & catalan_words:
        return "ca"
    elif pictos_set & spanish_words:
        return "es"
    return "ca"  # default

def batch_generate(combinations: List[List[str]], conn: sqlite3.Connection):
    """Generate phrases for all combinations"""
    checkpoint = load_checkpoint()
    start_idx = checkpoint["processed"]
    total = len(combinations)
    
    print(f"🚀 Starting batch generation")
    print(f"📊 Total combinations: {total}")
    print(f"✅ Already processed: {start_idx}")
    print(f"⏳ Remaining: {total - start_idx}\n")
    
    start_time = time.time()
    cursor = conn.cursor()
    
    for i in range(start_idx, total):
        pictos = combinations[i]
        input_text = " ".join(pictos)
        language = detect_language(pictos)
        
        try:
            # Generate phrase
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=input_text,
                options=GENERATION_OPTIONS,
                keep_alive="5m"
            )
            output = response['response'].strip()
            
            # Store in database
            cursor.execute(
                "INSERT OR REPLACE INTO phrases (pictos, language, output) VALUES (?, ?, ?)",
                (input_text, language, output)
            )
            
            # Progress update every BATCH_SIZE items
            if (i + 1) % BATCH_SIZE == 0:
                conn.commit()
                save_checkpoint(i + 1, total)
                
                elapsed = time.time() - start_time
                rate = (i + 1 - start_idx) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                
                print(f"✅ {i + 1}/{total} ({(i+1)/total*100:.1f}%) | "
                      f"Rate: {rate:.1f}/s | "
                      f"ETA: {remaining/60:.1f}min | "
                      f"Last: {input_text} → {output}")
        
        except Exception as e:
            print(f"❌ Error processing {input_text}: {e}")
            continue
    
    # Final commit
    conn.commit()
    save_checkpoint(total, total)
    
    elapsed = time.time() - start_time
    print(f"\n🎉 Batch generation complete!")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"📊 Processed: {total} combinations")
    print(f"⚡ Average rate: {total/elapsed:.2f} phrases/second")
    print(f"💾 Database: {OUTPUT_DB}")

def main():
    print("🔥 Warming up model...")
    try:
        ollama.generate(
            model=MODEL_NAME,
            prompt="test",
            options={"num_predict": 1},
            keep_alive="5m"
        )
        print("✅ Model ready!\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Ollama is running and model is pulled")
        return
    
    # Initialize database
    conn = init_database()
    
    # Generate combinations
    print("📝 Generating combinations...")
    combinations = generate_combinations()
    print(f"✅ Generated {len(combinations)} combinations\n")
    
    # Estimate time
    estimated_time = len(combinations) * 0.15 / 60  # 150ms per phrase
    print(f"⏱️  Estimated time: {estimated_time:.1f} minutes\n")
    
    # Run batch generation
    batch_generate(combinations, conn)
    
    conn.close()

if __name__ == "__main__":
    main()
