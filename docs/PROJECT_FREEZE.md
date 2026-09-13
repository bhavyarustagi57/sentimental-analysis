# Project Freeze Record

**Freeze date:** 2026-08-21  
**Status:** audited, reproducible, and submission-ready.

## Academic objective

Compare six classical machine-learning classifiers for binary sentiment classification across four real datasets using one fixed TF-IDF evaluation pipeline.

## Frozen configuration

- **Datasets (4):** IMDb Movie Reviews, Sentiment140, Amazon Polarity, Yelp Polarity
- **Models (6):** Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest, Decision Tree, KNN
- **Metrics (6):** Accuracy, weighted Precision, weighted Recall, weighted F1, training time, inference time
- **Task/labels:** binary sentiment; `0 = negative`, `1 = positive`
- **Samples:** 10,000 prepared rows per dataset, targeted at 5,000 per class
- **Split/seed:** 80/20 stratified split, random state 42
- **Features:** train-only TF-IDF, 20,000 maximum uni/bi-gram features

## Frozen evidence

- **Completion:** 24/24 unique dataset × model experiments
- **Canonical results:** `results/results.csv`, `results/results.json`
- **Methodology record:** `results/EXPERIMENT_METADATA.md`
- **Artifacts:** 24 expected local `.joblib` artifacts; validated during final audit
- **Leakage controls:** pre-split deduplication, disjoint normalized train/test text, train-only vectorizer fitting

The result CSV, JSON, metadata, prepared datasets, and model binaries are submission evidence and must not be overwritten during ordinary dashboard use. Canonical regeneration is explicit rather than automatic.

## Local commands

```powershell
# Install and run
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
npm run dev

# Recreate processed data only when intentionally rebuilding the baseline
.\.venv\Scripts\python.exe -m ml.prepare_datasets

# Regenerate the full academic baseline only when explicitly required
.\.venv\Scripts\python.exe -m ml.experiment_runner --all

# Validation
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
npm run lint
npm run build
```

For interpretation and viva preparation, use `docs/ACADEMIC_RESULTS.md` and `docs/VIVA_GUIDE.md`.
