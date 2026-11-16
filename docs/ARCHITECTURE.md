# Architecture

## System overview

```
React + Vite dashboard (:5173)
        │  /api/* proxy (vite.config.js)
        ▼
FastAPI + Uvicorn backend (:8000)
        │
        ├── results/results.json, results.csv   (frozen 24-experiment evidence, read-only)
        ├── models/*.joblib                      (24 saved classifier+vectorizer artifacts)
        └── optional: Reddit search / CSV upload → VADER → Groq LLM → runtime/sessions.json
```

Two independent processes talk over HTTP only: the frontend never reads a model file, a dataset, or the results files directly. Everything it shows comes from a backend JSON response.

## Backend (`main.py`, `ml/`)

FastAPI app, Pydantic request/response models, read-only by design for ML:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status, Groq-configured flag, pipeline status |
| `GET /ml/datasets`, `GET /ml/models` | Static metadata for datasets/models |
| `GET /ml/results` | All 24 frozen result records |
| `GET /ml/summary` | Rankings, best-per-dataset, derived conclusions |
| `POST /ml/predict` | `{text, dataset, model}` → saved-model prediction |
| `POST /ml/train`, `/ml/train-all` | `403 Forbidden` by design — results are frozen |
| `POST /analyze` | Start a product report from Reddit |
| `POST /upload-csv` | Start a product report from an uploaded file |
| `GET /status`, `/latest`, `/history`, `/history/{id}` | Product-report progress/history |
| `GET /groq/check`, `GET /sample-csv` | Groq key test, bundled sample file |

**`ml/` package** holds the experiment pipeline, split into single-responsibility modules: `dataset_loader.py` / `prepare_datasets.py` (download + normalize), `split_manifest.py` (deterministic 80/20 stratified split), `preprocessing.py` (text cleaning, negation-safe), `feature_extraction.py` (TF-IDF, train-only fit), `trainers.py` / `training.py` (the six classifiers), `evaluator.py` (metrics), `predictor.py` (loads a saved artifact and classifies new text), `experiment_runner.py` (drives all 24 runs), `results_analysis.py` (produces the summary the API serves). `model_registry.py` and `config.py` hold shared constants (random seed, model/dataset identifiers).

A separate, clearly isolated group of modules (`modern_benchmark*.py`, `phase4_orchestrator.py`, `phase4_persistence.py`, `transformer_*.py`, `*_smoke.py`) implements the **unfinished Phase 4 transformer extension** (see `docs/PHASE4_BENCHMARK_EXECUTION.md`). These modules are never imported by `main.py`'s request path and have no effect on the frozen baseline or the running dashboard; their tests skip automatically when `torch` is absent.

`reddit_scraper.py`, `preprocessor.py`, `templates.py`, `json_synthesizer.py`, `rate_limiter.py`, and `aso_router.py` implement the optional product-report pipeline: Reddit JSON search → text cleanup/dedup → VADER scoring (`data_manager.py` persists sessions to `runtime/sessions.json`) → a Groq LLM call that only summarizes the already-scored data.

## Frontend (`src/`)

Single-page React app, one `App.jsx` owning top-level state (theme, backend health, the loaded study payload) and fetching through `src/api.js`. Everything below it is presentational, reading props:

| File | Responsibility |
|---|---|
| `components/TopBar.jsx` | Nav, section tabs, connection indicator, theme toggle |
| `components/Overview.jsx` | Headline numbers, per-dataset winner cards |
| `components/Results.jsx` | Accuracy heatmap + full sortable table, CSV export |
| `components/Compare.jsx` | Mean accuracy/F1 ranking, speed-vs-accuracy chart |
| `components/Predictor.jsx` | Single-model and six-model-vote live prediction UI |
| `components/Method.jsx` | Pipeline explanation / methodology walkthrough |
| `components/Equalizer.jsx` | Six-bar loading indicator (model-colored), shown while the vote or a product report is running |
| `components/Shell.jsx` | Shared chrome: loading skeleton, backend-down state, footer |
| `ProductReport.jsx` | The optional, separate product-review feature (own section, own API calls) |
| `lib.js` | Palette/theme helpers, `useRevealOnce` (scroll-triggered, once-only reveal animations), storage helpers |

Styling is plain CSS (`src/index.css`) against a documented design-token scale (`DESIGN.md`); Tailwind and Sass are installed but unused by the dashboard.

## Data flow for a prediction

1. User picks a dataset + model (or "let all 6 vote") and submits text.
2. Frontend calls `POST /ml/predict` (once, or six times in parallel for a vote).
3. Backend's `ml/predictor.py` loads `models/<dataset>_<model>.joblib`, which bundles the fitted classifier **and** its own TF-IDF vectorizer.
4. The saved vectorizer (not a new one) transforms the input text; the saved classifier predicts; the label is returned.
5. No training occurs on this path — if the artifact file is missing, the API returns a clear error instead of retraining.

## Why this shape

- **Strict separation of frozen evidence from live computation**: `results/` is never written to by the running app; only `ml/experiment_runner.py`, run manually and intentionally, can regenerate it.
- **One pipeline, six models**: feature extraction and splitting code paths are shared across all six classifiers per dataset, so differences in the results come from the models, not from inconsistent preprocessing.
- **Backend-driven truth**: the frontend contains no hardcoded metrics; every number is fetched, so there is only one place (the backend's stored results) that can ever be wrong or right.
- **Optional feature isolation**: the product-report code path is additive — removing a Groq key degrades only that feature, never the academic dashboard or predictor.
