# Aropi Backend

Pictogram → natural-language sentence service, optimised for Catalan
(Spanish coming later). No LLM inference at request time: a small
subject+verb conjugation database is pre-generated once, and runtime
composition appends object/modifier pictograms as-is in the target
language.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design and HTTP
API contract, and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for
the phased migration from the legacy LLM-at-request-time server.

---

## Quick start

```bash
# 1. Python deps (inside the repo's .venv or your env)
pip install flask ollama

# 2. Make sure Ollama is running with the two custom models:
ollama serve &
ollama create cat-translator -f Modelfile_cat_translator
ollama create cat-composer   -f Modelfile_cat_composer

# 3. Start the server
python app.py
# -> listens on http://0.0.0.0:5000
```

Then from another shell:

```bash
curl -s http://localhost:5000/health | jq
curl -s 'http://localhost:5000/pictos?language=ca' | jq '.count, .pictos[0]'
curl -s -X POST http://localhost:5000/compose \
  -H 'Content-Type: application/json' \
  -d '{"pictos":["I","eat","apple"],"language":"ca"}' | jq
```

---

## (Re-)building the databases

If you change the vocabulary or add pictos, rerun the pipeline in this
order. Each step is idempotent and resumable.

### 1. Build `pictos.db` from `project/input/words.csv`

```bash
cd project/python
python build_db.py
```

### 2. Populate Catalan translations (`word_ca`)

```bash
cd project/python
python translate_pictos.py --column word_ca --model cat-translator
```

- Uses `project/input/translation_overrides_ca.csv` for manual
  corrections (edit freely).
- Skips rows where `word_ca` is already non-null, so re-running only
  fills gaps.
- To force a full re-translation:
  ```bash
  sqlite3 project/output/pictos.db "UPDATE pictos SET word_ca = NULL;"
  ```

### 3. Pre-generate subject+verb conjugations

From the repo root:

```bash
systemd-inhibit --what=idle:sleep:handle-lid-switch \
  --why="batch conjugation" \
  python batch_generate.py
```

- ~410 jobs at current vocabulary → ~30 min.
- Safe to interrupt: resumes via `batch_checkpoint.json` and DB-level
  duplicate skip.
- To force a rebuild:
  ```bash
  rm -f pictogram_phrases.db batch_checkpoint.json
  ```

### 4. Start / restart the server

```bash
python app.py
```

The server loads both DBs into memory at startup. Restart it after
regenerating so caches are fresh.

### 5. (Optional) Package an offline bundle for the mobile app

Once `pictos.db` and `pictogram_phrases.db` are satisfactory, build a
bundle that the Kotlin app can download for offline use:

```bash
python make_bundle.py
```

- Produces `bundles/<version>/aropi-bundle-<version>.zip` (DBs + PNGs).
- Updates `bundles/latest.json` so `GET /bundle/latest` returns the new
  version.
- `<version>` defaults to the current UTC timestamp; override with
  `--version v2026.04.20` if you want semantic versioning.

The mobile app polls `/bundle/latest`, compares the `version` against
the one it has locally, and downloads the zip when newer. See
`ARCHITECTURE.md` §5 for the full offline-first mobile flow.

---

## HTTP endpoints (summary)

| Method | Path                         | Purpose                                   |
|:------ |:---------------------------- |:----------------------------------------- |
| GET    | `/`                          | Endpoint listing                          |
| GET    | `/health`                    | Liveness + head count                     |
| GET    | `/pictos?language=ca`        | Catalogue for picker (id, translation, image) |
| GET    | `/pictos/image/<file.png>`   | Serve pictogram PNG                       |
| GET    | `/bundle/latest`             | Pointer to newest offline bundle          |
| GET    | `/bundle/<version>/<file>`   | Download a bundle zip                     |
| POST   | `/compose`                   | Pictos → conjugated sentence (fallback)   |
| POST   | `/cache/clear`               | Clear the phrase cache                    |

See `ARCHITECTURE.md` §4 for request/response schemas.

---

## Editing translation quality

The pre-generated Catalan strings aren't perfect. Two places to fix
things:

- **Word-level translations** — edit
  `project/input/translation_overrides_ca.csv`, then:
  ```bash
  sqlite3 project/output/pictos.db \
    "UPDATE pictos SET word_ca = NULL WHERE word = 'brush';"
  python project/python/translate_pictos.py --column word_ca --model cat-translator
  ```
- **Conjugated heads** — edit directly:
  ```bash
  sqlite3 pictogram_phrases.db \
    "UPDATE phrases SET output='Jo bec.' WHERE pictos='I drink' AND language='ca';"
  ```
  Then either restart the server or `POST /cache/clear`.

---

## Project layout

```
AropiBackend/
├── app.py                          # Flask server (runtime composition)
├── batch_generate.py               # Pre-compute subject+verb conjugations
├── pictogram_phrases.db            # Output of batch_generate.py
├── Modelfile_cat_translator        # Ollama Modelfile — EN → CA word translator
├── Modelfile_cat_composer          # Ollama Modelfile — CA subject+verb composer
├── project/
│   ├── input/
│   │   ├── words.csv                     # Seed vocabulary
│   │   └── translation_overrides_ca.csv  # Manual CA translation fixes
│   ├── output/
│   │   ├── pictos.db                     # Pictogram catalogue
│   │   ├── png/                          # PNG pictograms served to the app
│   │   └── svg/                          # Original SVGs
│   └── python/
│       ├── build_db.py                   # Builds pictos.db from words.csv
│       ├── translate_pictos.py           # Populates word_ca / word_es
│       ├── combination_generator.py      # Yields (pronoun_en, verb_en) pairs
│       ├── composer.py                   # PhraseComposer (runtime)
│       ├── db_utils.py                   # Schema helpers
│       ├── config.py                     # Paths
│       └── opensymbols_api.py            # OpenSymbols client
├── ARCHITECTURE.md
├── IMPLEMENTATION_PLAN.md
└── README.md                       # This file
```

---

## Adding Spanish (later)

1. Create a `Modelfile_es_translator` (same shape as the Catalan one,
   system prompt in Spanish).
2. `ollama create es-translator -f Modelfile_es_translator`
3. `python translate_pictos.py --column word_es --model es-translator`
4. Create `Modelfile_es_composer` for Spanish conjugation.
5. Add `"es"` to `TARGET_LANGUAGES` in `batch_generate.py`, update
   `MODEL_NAME` selection per language, rerun.
