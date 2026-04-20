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

Result: the mobile app composes grammatically correct phrases locally in
milliseconds, with **no LLM on the device and no network required at
runtime**. The backend's only responsibilities are (a) producing the
databases via offline batch jobs and (b) distributing them to clients as
versioned bundles.

```
┌─────────────────────────── runtime (offline) ──────────────────────────┐
│ ┌──────────────────┐    SQLite    ┌──────────────────────────────┐    │
│ │  Kotlin app      │ ◀──────────▶ │  pictos.db + phrases.db      │    │
│ │  OnDevice-       │              │  (bundled in APK / updated   │    │
│ │  PhraseComposer  │              │   via /bundle endpoints)     │    │
│ └──────────────────┘              └──────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────── once (server, no client involvement) ────────────────┐
│  words.csv ─▶ build_db.py ─▶ pictos.db                                 │
│                │                 │                                     │
│                ▼                 ▼                                     │
│           OpenSymbols     translate_pictos.py (cat-translator)         │
│              images        └─▶ word_ca / word_es populated             │
│                                  │                                     │
│                                  ▼                                     │
│                     batch_generate.py (cat-composer)                   │
│                            └─▶ pictogram_phrases.db                    │
│                                  │                                     │
│                                  ▼                                     │
│                            make_bundle.py                              │
│                            └─▶ bundles/<ver>/*.zip + latest.json       │
└────────────────────────────────────────────────────────────────────────┘

┌──────────────────────── sync (periodic, optional) ─────────────────────┐
│  Mobile ──GET /bundle/latest──▶ Flask ──GET /bundle/<ver>/<file>──▶    │
│      verify sha256 · unzip to filesDir · swap active version           │
└────────────────────────────────────────────────────────────────────────┘
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

### 4.4 `GET /pictos?language=ca`

Returns the catalogue so the mobile app can render a picker without
hardcoding the list.

**Response 200**
```json
{
  "language": "ca",
  "count": 185,
  "pictos": [
    {
      "id": "angry",
      "word": "angry",
      "translation": "enfadat",
      "language": "ca",
      "grammar_type": "adjective",
      "role": "modifier",
      "image_url": "/pictos/image/angry.png"
    }
  ]
}
```

- `id` / `word` — stable English identifier; this is what `/compose`
  expects in the `pictos` array.
- `translation` — localised label to display (may be `null` if the
  translation column isn't populated yet).
- `role` — one of `subject`, `verb`, `object`, `modifier`.
- `image_url` — relative path; prepend the backend base URL.

### 4.5 `GET /pictos/image/<filename>`

Serves the PNG for a pictogram. The filename comes from the `image_url`
field of the catalogue response; clients should not try to construct it
themselves.

### 4.6 `GET /bundle/latest`

Returns a pointer to the newest offline bundle. Mobile clients poll this
periodically (e.g. weekly, on Wi-Fi) and download the bundle if the
`version` differs from the one currently on device.

**Response 200**
```json
{
  "version": "20260420-1330",
  "filename": "aropi-bundle-20260420-1330.zip",
  "url": "/bundle/20260420-1330/aropi-bundle-20260420-1330.zip",
  "size": 1119315,
  "sha256": "ae99973267cb8687d34d0b8d792b642ba02d7de6bebeb7073fb39d06fb036b58",
  "created_at": "2026-04-20T13:30:55+00:00"
}
```

`404` is returned when no bundle has been built yet.

### 4.7 `GET /bundle/<version>/<filename>`

Serves the zip file referenced by `/bundle/latest`. Clients should verify
the downloaded file against the advertised `sha256` before swapping it
in.

---

## 5. How the Kotlin mobile app uses this

The app is **offline-first**. Both databases and all pictogram images are
shipped inside the APK as a bundle, and the entire `/compose` logic runs
on-device against the local SQLite files. The server is only consulted
to check for bundle updates.

### 5.1 Offline bundle lifecycle

```
┌───────────────────────── first launch ─────────────────────────┐
│  APK ships with assets/aropi-bundle/<version>.zip              │
│  → unzip into filesDir/aropi/<version>/                        │
│       ├── pictos.db                                            │
│       ├── pictogram_phrases.db                                 │
│       └── png/*.png                                            │
│  → open both DBs with Room, in read-only mode                  │
└────────────────────────────────────────────────────────────────┘

┌──────────────────────── periodic update ───────────────────────┐
│  BundleUpdater (WorkManager, constrained to Wi-Fi, weekly):    │
│    1. GET /bundle/latest                                       │
│    2. If version > local version:                              │
│         GET /bundle/<version>/<file>                           │
│         verify sha256                                          │
│         unzip to filesDir/aropi/<version>/                     │
│         atomically switch `active_version` in prefs            │
│         delete old version folders                             │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Building a sentence (zero-network path)

1. **Populate the picker** by reading `pictos` table from `pictos.db`
   filtered by `grammar_type` (map to role as in §2.1).
2. **User taps pictograms** → app keeps an ordered `List<String>` of
   English picto IDs (the `word` column).
3. **Compose on-device** — see `OnDevicePhraseComposer` below.
4. **Speak** — pipe the output string to Android's `TextToSpeech` with
   the chosen locale (`ca-ES`, `es-ES`).

### 5.3 On-device composer (Kotlin)

Mirrors `project/python/composer.py` one-for-one:

```kotlin
class OnDevicePhraseComposer(
    private val phrasesDb: SupportSQLiteDatabase,   // pictogram_phrases.db
    private val pictosDb:  SupportSQLiteDatabase    // pictos.db
) {
    sealed class Result {
        data class Ok(val sentence: String) : Result()
        data class HeadNotFound(val subject: String, val verb: String) : Result()
        data class UnknownPicto(val picto: String) : Result()
        object NotEnoughPictos : Result()
    }

    fun compose(pictos: List<String>, language: String): Result {
        if (pictos.size < 2) return Result.NotEnoughPictos

        val head = "${pictos[0]} ${pictos[1]}"
        val headOut = phrasesDb.query(
            "SELECT output FROM phrases WHERE pictos = ? AND language = ?",
            arrayOf(head, language)
        ).use { if (it.moveToFirst()) it.getString(0) else null }
            ?: return Result.HeadNotFound(pictos[0], pictos[1])

        val tail = pictos.drop(2)
        if (tail.isEmpty()) return Result.Ok(finalise(headOut))

        val translations = mutableListOf<String>()
        val column = "word_$language"
        for (p in tail) {
            val t = pictosDb.query(
                "SELECT $column FROM pictos WHERE word = ?",
                arrayOf(p)
            ).use { if (it.moveToFirst()) it.getString(0) else null }
                ?: return Result.UnknownPicto(p)
            translations.add(t)
        }

        val base = headOut.trimEnd('.', '!', '?', ' ')
        return Result.Ok(finalise("$base ${translations.joinToString(" ")}"))
    }

    private fun finalise(text: String): String {
        val t = text.trim()
        if (t.isEmpty()) return t
        val punctuated = if (t.last() in ".!?") t else "$t."
        return punctuated.replaceFirstChar { it.uppercase() }
    }
}
```

### 5.4 Retrofit interface (for bundle updates only)

The mobile app no longer needs `/compose` in the typical flow, but it
still hits `/bundle/latest` for updates:

```kotlin
data class BundlePointer(
    val version: String,
    val filename: String,
    val url: String,
    val size: Long,
    val sha256: String,
    val created_at: String
)

interface AropiApi {
    @GET("bundle/latest")
    suspend fun latestBundle(): BundlePointer

    @GET
    @Streaming
    suspend fun downloadBundle(@Url url: String): ResponseBody

    @GET("health")
    suspend fun health(): Map<String, Any>
}
```

### 5.5 Optional online fallback

If on-device composition returns `HeadNotFound` (e.g. the user's bundle
predates a newly added pronoun/verb), the app **may** fall back to
`POST /compose` when online. Response shape documented in §4.1.

---

## 6. Why this architecture

- **Bounded pre-computation.** Only `|subjects| × |verbs|` entries per
  language — trivial to regenerate when vocabulary grows.
- **Offline-first.** AAC users may not always have connectivity.
  Shipping a self-contained SQLite bundle means the app works indefinitely
  without the server.
- **No on-device LLM.** Composition is just two SQLite lookups plus
  string concatenation. Same Python logic, same Kotlin logic.
- **Deterministic outputs.** Same input → same sentence, so client and
  server always agree.
- **Graceful growth.** Adding new object/modifier pictograms does NOT
  require regenerating the conjugation DB — they are slotted in at
  composition time.

---

## 7. Repository layout (relevant files)

```
AropiBackend/
├── app.py                          # Flask server (API + bundle distribution)
├── batch_generate.py               # Pre-compute subject+verb conjugations
├── make_bundle.py                  # Package DBs + PNGs as versioned zip
├── pictogram_phrases.db            # Output of batch_generate.py
├── bundles/                        # Versioned offline bundles for clients
│   ├── latest.json                 # Pointer to newest bundle
│   └── <version>/
│       └── aropi-bundle-<ver>.zip  # pictos.db + phrases.db + png/
├── Modelfile_cat_translator        # Ollama Modelfile — EN → CA translator
├── Modelfile_cat_composer          # Ollama Modelfile — CA subject+verb composer
├── project/
│   ├── input/
│   │   ├── words.csv                     # Seed vocabulary
│   │   └── translation_overrides_ca.csv  # Manual CA translation fixes
│   ├── output/
│   │   ├── pictos.db                     # Pictogram catalogue
│   │   ├── png/                          # PNG pictograms
│   │   └── svg/                          # Original SVGs
│   └── python/
│       ├── build_db.py                   # Builds pictos.db from words.csv
│       ├── translate_pictos.py           # Populates word_ca / word_es
│       ├── combination_generator.py      # Yields (pronoun_en, verb_en) pairs
│       ├── composer.py                   # PhraseComposer (server runtime)
│       ├── db_utils.py                   # Schema helpers
│       ├── config.py                     # Paths + API secrets
│       └── opensymbols_api.py            # OpenSymbols client
├── ARCHITECTURE.md                 # This file
├── IMPLEMENTATION_PLAN.md          # Migration / build plan
└── README.md                       # Quick-start + run instructions
```
