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

## Phase 1 — Extend `pictos.db` schema (DONE)

- [x] Added nullable columns `word_ca` and `word_es` to `pictos`
      (idempotent `ensure_translation_columns()`).
- [x] `create_schema()` in `db_utils.py` now includes both columns.
- [x] `update_translation()` helper added.

## Phase 2 — Translator script (DONE)

- [x] `project/python/translate_pictos.py` populates translation columns
      via `cat-translator` with a three-tier fallback (manual override,
      bare prompt, grammar-hinted retry).
- [x] `project/input/translation_overrides_ca.csv` holds manual fixes.
- [x] Catalan translations populated.

## Phase 3 — Fix `CombinationGenerator` (DONE)

- [x] Replaced hardcoded grammar-type strings with role mapping:
      ```python
      ROLE_TO_TYPES = {
          "subject":  ["pronoun"],
          "verb":     ["verb"],
          "object":   ["noun"],
          "modifier": ["adjective", "adverb"],
      }
      ```
- [x] `generate_subject_verb_combinations()` returns `(pronoun_en,
      verb_en)` tuples drawn from `pictos.db`.
- [x] Public API kept small (subject+verb + `get_stats`).

## Phase 4 — Refactor `batch_generate.py` (DONE)

- [x] Hardcoded word lists and `detect_language()` removed.
- [x] Uses `CombinationGenerator` + `word_ca` translations.
- [x] Stores rows keyed by English IDs + target language.
- [x] Ran successfully: 410 jobs in ~32 min, 9 errors, 401 rows stored.

## Phase 5 — Add a composition layer (DONE)

- [x] `project/python/composer.py` provides `PhraseComposer` with
      `compose(pictos, language)`.
- [x] Structured errors (`HeadNotFoundError`, `UnknownPictoError`).
- [x] CLI for ad-hoc testing.
- [x] Manually verified against several inputs.

## Phase 6 — Wire composer into Flask `app.py` (DONE)

- [x] `/compose` now calls `PhraseComposer` — no runtime LLM.
- [x] Phrase cache layer preserved.
- [x] Startup warms both DB caches in-process.
- [x] 404 for missing head, 400 for validation errors.

## Phase 7 — Expose the catalogue to the mobile app (DONE)

- [x] `GET /pictos?language=ca` returns catalogue with translation,
      grammar type, role, and image URL per item.
- [x] `GET /pictos/image/<filename>` serves PNGs from
      `project/output/png/`.
- [x] Both endpoints documented in `ARCHITECTURE.md` §4.

## Phase 7.5 — Offline bundle distribution (DONE)

- [x] `make_bundle.py` packages `pictos.db` + `pictogram_phrases.db` +
      PNGs into `bundles/<version>/aropi-bundle-<version>.zip` with a
      `manifest.json` and top-level `bundles/latest.json` pointer.
- [x] `GET /bundle/latest` exposes the pointer (version, url, size,
      sha256).
- [x] `GET /bundle/<version>/<filename>` streams the zip.
- [x] `bundles/` ignored by git; rebuilt on demand.
- [x] Architecture doc updated with offline-first mobile flow and
      on-device Kotlin composer spec (§5).

## Phase 8 — Mobile-side integration (Kotlin, offline-first)

Out-of-repo work, but fully specced in `ARCHITECTURE.md` §5.

- [ ] Ship initial bundle (`assets/aropi-bundle-<version>.zip`) in APK.
- [ ] On first launch, unzip to `filesDir/aropi/<version>/` and record
      `active_version` in SharedPreferences.
- [ ] Open both SQLite files via Room in read-only mode.
- [ ] Port `PhraseComposer` to Kotlin (`OnDevicePhraseComposer` in §5.3).
- [ ] BundleUpdater (WorkManager, Wi-Fi constrained, weekly): GET
      `/bundle/latest`, compare versions, download + sha256-verify +
      atomic swap on mismatch.
- [ ] TTS pipeline: feed composer output to `TextToSpeech` with locale
      from the user's language setting.
- [ ] (Optional) Online fallback to `POST /compose` when
      `HeadNotFound` is returned by the on-device composer.

## Phase 9 — Cleanup & docs (DONE)

- [x] `README.md` written with quick-start, rebuild, bundle, and
      endpoint summary.
- [x] `ARCHITECTURE.md` updated for English-keyed data model, new
      catalogue endpoints, and offline-first mobile flow.
- [x] Removed `*.legacy` artefacts:
      `pictogram_phrases.db.legacy`, `batch_checkpoint.json.legacy`,
      `app.py.legacy`, `batch_generate.py.legacy`,
      `combination_generator.py.new`, and the unused
      `phrase_cache_ollama.json`.
- [x] `.gitignore` updated to exclude future `*.legacy`, generated
      bundles, and runtime caches.
- [ ] Tag the repo: `git tag v0.2-compositional` (manual).

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
