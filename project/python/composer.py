"""
PhraseComposer: produces a natural sentence from a list of pictogram IDs
without calling any LLM at request time.

Strategy:
  - The first two pictos must be (subject, verb). Their conjugated form in
    the target language is looked up from `pictogram_phrases.db`.
  - Remaining pictos are translated via `pictos.db` (`word_ca` / `word_es`)
    and appended to the conjugated head.
  - The result is punctuated and returned.
"""

import os
import sqlite3
from typing import Dict, List, Optional


SUPPORTED_LANGUAGES = ("ca", "es")


class HeadNotFoundError(LookupError):
    """The (subject, verb) pair is not present in the phrases DB."""


class UnknownPictoError(LookupError):
    """A tail picto isn't in pictos.db for the requested language."""


class PhraseComposer:
    def __init__(self, phrases_db_path: str, pictos_db_path: str):
        if not os.path.exists(phrases_db_path):
            raise FileNotFoundError(f"phrases DB not found: {phrases_db_path}")
        if not os.path.exists(pictos_db_path):
            raise FileNotFoundError(f"pictos DB not found: {pictos_db_path}")
        self.phrases_db_path = phrases_db_path
        self.pictos_db_path = pictos_db_path
        # Small in-process caches. These are rebuilt on process start; the
        # phrases table is small (<1000 rows per language).
        self._heads: Dict[tuple[str, str], str] = {}
        self._tails: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------
    def _load_heads(self) -> None:
        if self._heads:
            return
        conn = sqlite3.connect(self.phrases_db_path)
        cur = conn.cursor()
        cur.execute("SELECT pictos, language, output FROM phrases")
        for pictos, language, output in cur.fetchall():
            self._heads[(pictos, language)] = output
        conn.close()

    def _load_tails(self, language: str) -> Dict[str, str]:
        if language in self._tails:
            return self._tails[language]
        column = f"word_{language}"
        conn = sqlite3.connect(self.pictos_db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT word, {column} FROM pictos WHERE {column} IS NOT NULL")
        mapping = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()
        self._tails[language] = mapping
        return mapping

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compose(self, pictos: List[str], language: str = "ca") -> str:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}; expected one of {SUPPORTED_LANGUAGES}"
            )
        if not isinstance(pictos, list) or len(pictos) < 2:
            raise ValueError("`pictos` must be a list of at least 2 items (subject, verb, ...)")

        head_key = f"{pictos[0]} {pictos[1]}"
        self._load_heads()
        head_output = self._heads.get((head_key, language))
        if head_output is None:
            raise HeadNotFoundError(
                f"No pre-generated phrase for ({pictos[0]!r}, {pictos[1]!r}) in {language!r}"
            )

        tail_pictos = pictos[2:]
        if not tail_pictos:
            return self._finalise(head_output)

        tails = self._load_tails(language)
        tail_words: List[str] = []
        for picto in tail_pictos:
            translated = tails.get(picto)
            if translated is None:
                raise UnknownPictoError(
                    f"No {language!r} translation found for picto {picto!r}"
                )
            tail_words.append(translated)

        # Strip trailing punctuation from the head so we can re-punctuate cleanly
        base = head_output.rstrip(".!? ")
        return self._finalise(f"{base} {' '.join(tail_words)}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finalise(text: str) -> str:
        text = text.strip()
        if not text:
            return text
        if text[-1] not in ".!?":
            text += "."
        return text[0].upper() + text[1:]


# ---------------------------------------------------------------------------
# CLI for quick manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_PHRASES = os.path.join(REPO_ROOT, "pictogram_phrases.db")
    DEFAULT_PICTOS = os.path.join(REPO_ROOT, "project", "output", "pictos.db")

    parser = argparse.ArgumentParser(description="Compose a sentence from pictogram IDs.")
    parser.add_argument("pictos", nargs="+", help="Ordered picto IDs (English), e.g. 'I eat apple'")
    parser.add_argument("--language", default="ca", choices=SUPPORTED_LANGUAGES)
    parser.add_argument("--phrases-db", default=DEFAULT_PHRASES)
    parser.add_argument("--pictos-db", default=DEFAULT_PICTOS)
    args = parser.parse_args()

    composer = PhraseComposer(args.phrases_db, args.pictos_db)
    try:
        print(composer.compose(args.pictos, language=args.language))
    except (HeadNotFoundError, UnknownPictoError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
