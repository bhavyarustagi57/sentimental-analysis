# Sentiment Analysis — Comparative Machine Learning Study

## Project overview

This college minor project compares six classical supervised-learning models for binary sentiment classification. It presents a reproducible, frozen academic baseline of **24 real experiments** across four public datasets, with a FastAPI backend and a React dashboard for comparison and saved-model prediction.

The optional VADER/Groq product-review workflow is separate from this academic comparison and is not counted as an ML classifier in the study.

## Research objective

Compare Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Machine, Random Forest, Decision Tree, and K-Nearest Neighbors across IMDb, Sentiment140, Amazon Polarity, and Yelp Polarity using predictive quality and computational-efficiency metrics.

## Datasets

Each prepared dataset contains up to 10,000 genuine, deduplicated examples: 5,000 negative and 5,000 positive reviews/posts. The normalized schema is `text,label`, where `0 = negative` and `1 = positive`.

| Dataset | Domain | Prepared samples |
|---|---|---:|
| IMDb Movie Reviews | Movie reviews | 10,000 |
| Sentiment140 | Social-media posts | 10,000 |
| Amazon Polarity | Product reviews | 10,000 |
| Yelp Polarity | Restaurant reviews | 10,000 |

Source revisions, deterministic sampling, and provenance are recorded in `datasets/manifest.json` when datasets are prepared.

## Models

- **Logistic Regression** — efficient linear probabilistic classifier for sparse text.
- **Multinomial Naive Bayes** — fast frequency-based baseline with a conditional-independence assumption.
- **Linear SVM** — maximum-margin linear classifier for high-dimensional feature spaces.
- **Random Forest** — nonlinear tree ensemble with greater computational cost.
- **Decision Tree** — interpretable tree model that can overfit sparse data.
- **KNN** — lazy nearest-neighbour model with low fitting cost and potentially expensive inference.

## Methodology

```text
Prepared data → deduplication → 80/20 stratified split → text cleaning
→ train-only TF-IDF → classifier fitting → test prediction → evaluation
→ artifact and result persistence
```

- Random state: `42`
- Split: 80% train / 20% test, stratified where possible
- TF-IDF: up to 20,000 features, unigrams/bigrams, `min_df=2`, negation-safe stop words
- Precision, Recall, F1: weighted average with `zero_division=0`
- Timings: `perf_counter()` around classifier fitting and test-set prediction only

TF-IDF is fitted only on training text. Deduplication happens before splitting, and normalized train/test text overlap is checked to prevent leakage.

## Frozen results

The canonical 4 × 6 baseline is stored in:

- `results/results.json`
- `results/results.csv`
- `results/EXPERIMENT_METADATA.md`

Do not rerun or overwrite these results for normal dashboard use. The dashboard and `GET /ml/summary` derive their conclusions directly from those records.

| Dataset | Best model | Accuracy | F1 |
|---|---|---:|---:|
| IMDb Movie Reviews | Linear SVM | 90.25% | 90.25% |
| Sentiment140 | Logistic Regression | 71.40% | 71.38% |
| Amazon Polarity | Logistic Regression | 84.25% | 84.25% |
| Yelp Polarity | Linear SVM | 90.35% | 90.35% |

Linear models have the strongest average predictive results in this setup. KNN has minimal fitting cost but substantially slower inference and weak Sentiment140/Yelp outcomes. Random Forest has a much higher training cost on these sparse TF-IDF features. See [docs/ACADEMIC_RESULTS.md](docs/ACADEMIC_RESULTS.md) for the complete analysis.

## Limitations and future scope

This study uses classical models, binary labels, deterministic 10,000-sample subsets, and no hyperparameter optimization. TF-IDF does not capture deeper contextual semantics, and timing depends on the hardware and system load. Future work could evaluate tuned models, cross-validation, contextual embeddings/transformers, multilingual or aspect-based sentiment, full datasets, and model explainability.

## Run locally

Use **Python 3.11 or 3.12** (Python 3.12 was validated for this frozen release) and **Node.js 20.19+ or 22.12+**. Older versions are not supported by the pinned pandas/scikit-learn or Vite dependencies.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
```

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Start the frontend in another terminal:

```powershell
npm run dev
```

Open `http://localhost:5173`. The dashboard loads saved results without retraining. `POST /ml/predict` uses the matching stored vectorizer and model. If an artifact is absent, result visualizations still work and prediction returns a clear regeneration message.

## Dataset preparation and experiment regeneration

The canonical results are already included for submission. Only regenerate datasets/artifacts when intentionally rebuilding the academic baseline:

```powershell
python -m ml.prepare_datasets
python -m ml.experiment_runner --all
```

Raw source downloads are optional and can be recreated by the preparation command. The prepared datasets and `.joblib` artifacts are retained because they are required for the frozen dashboard and real prediction flow.

## Submission ZIP checklist

Include the source files, `requirements.txt`, `package.json`, `package-lock.json`, `settings.json`, `.env.example`, `docs/`, `results/`, `datasets/manifest.json`, all four files in `datasets/processed/`, and all 24 `models/*.joblib` files. These prepared datasets and artifacts are required to view the frozen evidence and make predictions without retraining.

Do **not** include `.env`, `.venv/`, `node_modules/`, `dist/`, `runtime/`, caches, or `datasets/raw/`. A Groq key is optional; it enables only the separate product-analysis feature, not the academic dashboard or prediction endpoint.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
npm run lint
npm run build
```

For presentation and viva preparation, see [docs/ACADEMIC_RESULTS.md](docs/ACADEMIC_RESULTS.md) and [docs/VIVA_GUIDE.md](docs/VIVA_GUIDE.md).
