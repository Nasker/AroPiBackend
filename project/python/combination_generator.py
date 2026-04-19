"""
CombinationGenerator: reads pictos.db and yields combinations for the
batch conjugation step.

Pictos are indexed by their English word (stable identifier, 1:1 with the
image file). Translations live in `word_ca` / `word_es` columns.

The DB uses the grammar taxonomy produced by `build_db.py`
(`pronoun`, `verb`, `noun`, `adjective`, `adverb`). This module maps those
to semantic roles used by the composer:

    subject  <- pronoun
    verb     <- verb
    object   <- noun
    modifier <- adjective, adverb
"""

import sqlite3
from typing import List, Tuple

from config import DB_PATH


ROLE_TO_TYPES = {
    "subject":  ["pronoun"],
    "verb":     ["verb"],
    "object":   ["noun"],
    "modifier": ["adjective", "adverb"],
}


class CombinationGenerator:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _words_for_role(self, role: str) -> List[str]:
        grammar_types = ROLE_TO_TYPES.get(role)
        if not grammar_types:
            raise ValueError(f"Unknown role: {role!r}")

        placeholders = ",".join("?" * len(grammar_types))
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            f"SELECT word FROM pictos WHERE grammar_type IN ({placeholders}) ORDER BY word",
            grammar_types,
        )
        words = [row[0] for row in cur.fetchall()]
        conn.close()
        return words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_subject_verb_combinations(self) -> List[Tuple[str, str]]:
        """Return all (subject_en, verb_en) pairs from the DB."""
        subjects = self._words_for_role("subject")
        verbs = self._words_for_role("verb")
        return [(s, v) for s in subjects for v in verbs]

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM pictos")
        total = cur.fetchone()[0]

        cur.execute("SELECT grammar_type, COUNT(*) FROM pictos GROUP BY grammar_type")
        by_type = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM pictos WHERE word_ca IS NOT NULL")
        translated_ca = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM pictos WHERE word_es IS NOT NULL")
        translated_es = cur.fetchone()[0]

        conn.close()
        return {
            "total_pictos": total,
            "by_grammar_type": by_type,
            "translated_ca": translated_ca,
            "translated_es": translated_es,
        }


if __name__ == "__main__":
    generator = CombinationGenerator()
    stats = generator.get_stats()
    print("Database Statistics:")
    print(f"  Total pictos:       {stats['total_pictos']}")
    print(f"  By grammar type:    {stats['by_grammar_type']}")
    print(f"  Translated (ca):    {stats['translated_ca']}")
    print(f"  Translated (es):    {stats['translated_es']}")
    print()

    combos = generator.generate_subject_verb_combinations()
    print(f"Generated {len(combos)} subject+verb combinations")
    print("First 10:")
    for pair in combos[:10]:
        print(f"  {pair}")
