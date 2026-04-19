# Aropi — System Architecture

This document describes the full Aropi system so that client applications
(notably the Android/Kotlin mobile app) can understand how to leverage the
backend.

---

## 1. High-level overview

Aropi converts sequences of **pictograms** (picked by a user in the mobile app)
into **natural-language sentences** in Catalan or Spanish.

The design separates two concerns:

1. **Morphology (the hard part)** — conjugating a verb for its subject.
   This is done by an LLM ahead of time and stored in a database.
2. **Composition (the easy part)** — concatenating the conjugated head
   (`subject + verb`) with any remaining object/modifier pictograms, which
   can be appended as-is in the user's target language.

Result: the mobile app, via a single HTTP request, gets a grammatically
correct phrase in milliseconds without needing any on-device LLM.

```
┌─────────────────┐      HTTP       ┌──────────────────┐
│  Mobile app     │ ──────────────▶ │  Flask backend   │
│  (Kotlin)       │ ◀────────────── │  (app.py)        │
└─────────────────┘                 └────────┬─────────┘
                                             │
                                 ┌───────────┴────────────┐
                                 │                        │
                          ┌──────▼──────┐          ┌──────▼──────┐
                          │ pictos.db   │          │ phrases.db  │
                          │ (picto →    │          │ (subject+   │
                          │  word, img) │          │  verb →     │
                          └─────────────┘          │  conjug.)   │
                                                   └─────────────┘
```

---

## 2. Data model

### 2.1 `pictos.db` — pictogram catalogue

Built by `project/python/build_db.py` from `project/input/words.csv` using the
OpenSymbols API. Each pictogram is identified by its **English word** (stable
ID, 1:1 with the image file). Translations into target languages live in
dedicated columns populated by `project/python/translate_pictos.py` using the
`cat-translator` Ollama model.

Schema (`project/python/db_utils.py`):

```sql
CREATE TABLE pictos (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  word         TEXT NOT NULL,         -- English picto identifier (e.g. "eat")
  language     TEXT NOT NULL,         -- source language of `word`, currently always 'en'
  grammar_type TEXT,                  -- 'pronoun'|'verb'|'noun'|'adjective'|'adverb'
  word_ca      TEXT,                  -- Catalan translation (primary target)
  word_es      TEXT,                  -- Spanish translation (optional, later)
  source       TEXT,
  symbol_id    TEXT,
  keywords     TEXT,
  image_path   TEXT NOT NULL
);
```

Grammar-type → semantic-role mapping used by the composer:

| DB `grammar_type` | Semantic role |
|---|---|
| `pronoun` | subject |
| `verb`    | verb |
| `noun`    | object |
| `adjective`, `adverb` | modifier |

### 2.2 `pictogram_phrases.db` — pre-computed subject + verb conjugations

Populated by `batch_generate.py`. Stores only the grammatically tricky
head of a sentence.

```sql
CREATE TABLE phrases (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  pictos     TEXT NOT NULL,           -- English picto IDs, e.g. "I eat"
  language   TEXT NOT NULL,           -- target language, 'ca' | 'es'
  output     TEXT NOT NULL,           -- conjugated head, e.g. "Jo menjo"
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(pictos, language)
);
```

Keys are **English picto identifiers** (not translated forms) so the same
row is reachable regardless of the user's UI language.

Size: O(subjects × verbs) per language — typically a few dozen rows.

---

## 3. Pipeline components

### 3.1 Ingestion — `project/python/build_db.py`
Reads `project/input/words.csv`, queries OpenSymbols for each word,
downloads the image to `project/output/images/`, and inserts a row in
`pictos.db`.

### 3.2 Translation — `project/python/translate_pictos.py`
One-shot script. For every row with `word_ca IS NULL`, calls Ollama with
the `cat-translator` model (see `Modelfile_cat_translator`) and stores the
result in `word_ca`. Idempotent and resumable. The same script, with a
different target column and model prompt, is used to populate `word_es`
later.

### 3.3 Combination generation — `project/python/combination_generator.py`
Exposes `CombinationGenerator`, which reads `pictos.db` and produces
subject + verb combinations. Internally maps DB `grammar_type` values
(`pronoun`, `verb`, …) to semantic roles. This is the single source of
truth for the batch generator.

### 3.4 Batch conjugation — `batch_generate.py`
For each `(pronoun_en, verb_en, target_language)` triple, resolves the
Catalan/Spanish translations via `pictos.db`, calls Ollama with the
`cat-composer` model, and stores the conjugated output in
`pictogram_phrases.db`. Features:

- Checkpoint file (`batch_checkpoint.json`) for resume.
- DB-level duplicate skip (safety net if the checkpoint is stale).
- Rolling-window ETA.
- Small `num_ctx` and persistent `keep_alive` for throughput.

### 3.4 Runtime composition — `app.py` (Flask)
Serves the mobile app. For each request it:

1. Splits input pictos into `head = [subject, verb]` and `tail = rest`.
2. Looks up the conjugated head in `pictogram_phrases.db`.
3. Translates each `tail` picto into the target language via `pictos.db`.
4. Concatenates `head + tail` and returns the sentence.

A small in-memory + JSON cache (`phrase_cache_ollama.json`) fronts the
endpoint for zero-cost repeat lookups.

---

## 4. HTTP API contract (what the Kotlin app consumes)

Base URL: `http://<host>:5000`

### 4.1 `POST /compose`

Compose a natural sentence from a list of pictogram identifiers.

**Request**
```json
{
  "pictos": ["jo", "voler", "aigua"],
  "language": "ca"
}
```

Fields:
- `pictos` *(array<string>, required)* — ordered pictogram words. The
  first two elements MUST be the subject and verb.
- `language` *(string, optional, default `"ca"`)* — `"ca"` or `"es"`.

**Response 200**
```json
{
  "input": "jo voler aigua",
  "output": "Jo vull aigua.",
  "pictos": ["jo", "voler", "aigua"],
  "language": "ca",
  "cached": false
}
```

**Errors**
- `400` — missing/invalid `pictos` field.
- `404` — subject+verb conjugation not found in DB (should be rare once
  the DB is fully populated).
- `503` — backend unavailable.

### 4.2 `GET /health`
Returns `{"status": "healthy", ...}` when the service and Ollama are up.
Use for connectivity checks from the mobile app.

### 4.3 `POST /cache/clear`
Clears the in-memory phrase cache. Dev/admin only.

### 4.4 `GET /pictos?language=ca` *(planned)*
Returns the catalogue (word + image URL + grammar type) so the mobile app
can render a picker without hardcoding the list.

---

## 5. How the Kotlin mobile app uses this

1. **On first launch**, fetch `/pictos?language=<user-lang>` and cache
   the catalogue locally (words + images).
2. **User builds a sentence** by tapping pictograms in order. The app
   keeps an ordered list of picto `word` identifiers.
3. **To speak the sentence**, the app `POST`s `/compose` with that list
   and the user's language, receives `output`, and pipes it to
   text-to-speech.
4. **Offline fallback**: the app may cache recent `(pictos, language) →
   output` pairs locally; since outputs are deterministic for a given
   input, this is safe.

Recommended Kotlin client shape:

```kotlin
data class ComposeRequest(val pictos: List<String>, val language: String)
data class ComposeResponse(
    val input: String,
    val output: String,
    val pictos: List<String>,
    val language: String,
    val cached: Boolean
)

interface AropiApi {
    @POST("compose")
    suspend fun compose(@Body req: ComposeRequest): ComposeResponse

    @GET("health")
    suspend fun health(): Map<String, Any>
}
```

---

## 6. Why this architecture

- **Bounded pre-computation.** Only `|subjects| × |verbs|` entries per
  language — trivial to regenerate when vocabulary grows.
- **No on-device LLM.** The phone never runs inference; it only does
  HTTP + string concatenation (handled server-side).
- **Deterministic, cacheable outputs.** Same input → same sentence,
  perfect for client-side caching.
- **Graceful growth.** Adding new object/modifier pictograms does NOT
  require regenerating the conjugation DB — they are slotted in at
  composition time.

---

## 7. Repository layout (relevant files)

```
AropiBackend/
├── app.py                          # Flask server (runtime composition)
├── batch_generate.py               # Pre-compute subject+verb conjugations
├── pictogram_phrases.db            # Output of batch_generate.py
├── project/
│   ├── input/words.csv             # Seed vocabulary
│   ├── output/
│   │   ├── pictos.db               # Pictogram catalogue
│   │   └── images/                 # Downloaded pictogram images
│   └── python/
│       ├── build_db.py             # Builds pictos.db from words.csv
│       ├── combination_generator.py# Reads pictos.db, yields (subj,verb) pairs
│       ├── db_utils.py             # pictos.db schema + inserts
│       ├── config.py               # Paths + API secrets
│       └── opensymbols_api.py      # OpenSymbols client
└── ARCHITECTURE.md                 # This file
```
