# Implementation Plan — Subject+Verb-Only Conjugation Refactor

Goal: replace the brute-force 8200-combination batch job with a compact
subject+verb conjugation DB, plus a runtime composition step that appends
object/modifier pictograms as-is in the target language.

See `ARCHITECTURE.md` for the final target design.

---

## Phase 0 — Pre-flight (DONE)

- [x] Audit `pictos.db`: all rows are `language='en'`; grammar types are
      `pronoun`/`verb`/`noun`/`adjective`/`adverb`. Picto IDs are the
      **English word**.
- [x] `Modelfile_cat_translator` already exists (model: `cat-translator`,
      prompt: English word → Catalan). It is the designated translator.
- [x] Snapshot current state:
      ```
      cp pictogram_phrases.db pictogram_phrases.db.legacy
      cp batch_checkpoint.json batch_checkpoint.json.legacy
      ```

## Phase 1 — Extend `pictos.db` schema

- [ ] Add nullable columns `word_ca TEXT` and `word_es TEXT` to
      `pictos` (idempotent `ALTER TABLE ... ADD COLUMN`).
- [ ] Update `create_schema()` in `project/python/db_utils.py` to include
      the new columns so fresh builds are consistent.
- [ ] Add helper `update_translation(db_path, picto_id, column, value)`
      for use by the translator.

## Phase 2 — Translator script `project/python/translate_pictos.py`

- [ ] Build `cat-translator` locally if not already:
      `ollama create cat-translator -f Modelfile_cat_translator`.
- [ ] Implement `translate_missing(db_path, target_column="word_ca",
      model="cat-translator")`:
      - Query rows where `<target_column> IS NULL`.
      - For each, call `ollama.generate(...)` with the English `word`.
      - Reject outputs that equal `"ERROR"` or contain whitespace /
        punctuation beyond the translated token (strip + validate).
      - Commit after every row (fast, <1s each) so it's fully resumable.
- [ ] CLI entry point: `python translate_pictos.py --column word_ca`
      (and later `--column word_es` with a different modelfile).
- [ ] Run once for Catalan. Spot-check a dozen outputs.

## Phase 3 — Fix `CombinationGenerator`

- [ ] Replace the hardcoded `grammar_type` strings with a role mapping:
      ```python
      ROLE_TO_TYPES = {
          "subject":  ["pronoun"],
          "verb":     ["verb"],
          "object":   ["noun"],
          "modifier": ["adjective", "adverb"],
      }
      ```
- [ ] `generate_subject_verb_combinations()` returns list of
      `(pronoun_en, verb_en)` tuples drawn from `pictos.db`. No language
      filter needed — picto IDs are language-agnostic.
- [ ] Keep the public API small: one method for subject+verb, plus
      `get_stats()`. Future methods (e.g. subject+verb+object) can be
      added without breaking callers.

## Phase 4 — Refactor `batch_generate.py`

- [ ] Remove hardcoded `SUBJECTS`/`VERBS`/`OBJECTS`/`MODIFIERS` and
      `detect_language()`.
- [ ] Use `CombinationGenerator` to get `(pronoun_en, verb_en)` pairs.
- [ ] For each pair × each `target_language` in `["ca"]` (add `"es"`
      later):
      1. Look up Catalan words via `word_ca` in `pictos.db`.
         Skip pair if either translation is missing.
      2. Build prompt: `"<pronoun_ca> <verb_ca>"`.
      3. Call `cat-composer` via Ollama.
      4. Store in `phrases` keyed by `"<pronoun_en> <verb_en>"` and
         `language=target_language`.
- [ ] Delete legacy `pictogram_phrases.db` before running.
- [ ] Expected row count: ~10 pronouns × ~41 verbs = ~410 rows per
      language. Runtime ≈ a few minutes.

## Phase 5 — Add a composition layer

Create `project/python/composer.py` with:

- [ ] `class PhraseComposer`:
      - `__init__(self, phrases_db_path, pictos_db_path)`
      - `compose(pictos: list[str], language: str) -> str`
- [ ] `compose()` steps:
      1. Validate `len(pictos) >= 2`.
      2. `head = pictos[:2]`, `tail = pictos[2:]`.
      3. Look up `(head_joined, language)` in `phrases` table.
      4. For each word in `tail`, look up its `word` column in `pictos`
         (if input is a picto id) or pass through (if input is already a
         target-language word).
      5. Join with spaces, append `.`, return.
- [ ] Unit test with a handful of known inputs per language.

## Phase 6 — Wire composer into Flask `app.py`

- [ ] Replace the `ollama.generate(...)` call in `/compose` with
      `PhraseComposer.compose(pictos, language)`.
- [ ] Keep the existing `_phrase_cache` layer — it still applies.
- [ ] Drop `warmup_model()` (no longer needed at request time; still
      useful during batch generation only).
- [ ] Return 404 with a clear error when the head isn't found in the DB.

## Phase 7 — Expose the catalogue to the mobile app

- [ ] Add `GET /pictos?language=ca` endpoint returning:
      ```json
      [{"word": "jo", "grammar_type": "subject",
        "image_url": "/static/pictos/jo.png"}, ...]
      ```
- [ ] Serve pictogram images from `static/` (or proxy from
      `project/output/images/`).
- [ ] Document the endpoint in `ARCHITECTURE.md` §4.

## Phase 8 — Mobile-side integration (Kotlin)

Out-of-repo work, but listed here for completeness:

- [ ] Add Retrofit interface matching `ARCHITECTURE.md` §5.
- [ ] Cache `/pictos` response on disk; invalidate by language + version
      header.
- [ ] LRU cache for recent `/compose` responses.
- [ ] TTS pipeline: feed `output` to `TextToSpeech` with locale from
      `language` field.

## Phase 9 — Cleanup & docs

- [ ] Remove `pictogram_phrases.db.legacy` once the new DB is validated.
- [ ] Update README with new run instructions.
- [ ] Tag the repo: `v0.2-compositional`.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Naive concatenation produces awkward Catalan (e.g. `molt aigua` vs `molta aigua`) | Accept for v0.2; add per-noun modifier variants in a follow-up if user testing flags it. |
| Missing `grammar_type` in `pictos.db` breaks combination generation | Phase 0 pre-flight check; fail loud with a clear error in `CombinationGenerator`. |
| Subject+verb head not found at compose time | Return 404 with the missing pair so the mobile app (and logs) can surface it for re-batching. |
| Users build sentences with non-subject first picto | Validate at `/compose`; return 400. Long-term: add a client-side ordering hint. |

## Done criteria

- `pictogram_phrases.db` contains ≤500 rows per target language
  (≈ pronouns × verbs from `pictos.db`).
- `/compose` returns correctly conjugated sentences in <50 ms for cached
  and <200 ms for uncached lookups.
- Kotlin app can build a 2–5 picto sentence end-to-end against a freshly
  provisioned backend with no LLM inference at request time.
