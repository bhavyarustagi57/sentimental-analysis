# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary:** university examiners at a 7th-semester minor-project viva. A student demos the dashboard live on a laptop or projector; examiners need to grasp the study, its results and the working prediction within minutes and judge whether the work is real and understood.
- **Secondary:** GitHub / live-link visitors (classmates, recruiters, faculty) opening the project later on their own laptops and phones, with no presenter beside them.

## Product Purpose

A comparative machine-learning study: six classical classifiers (Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest, Decision Tree, KNN) trained on four public sentiment datasets (IMDb, Sentiment140, Amazon Polarity, Yelp Polarity) under one leak-free TF-IDF pipeline, giving 24 frozen experiments. The dashboard presents the evidence and lets anyone classify their own text with the saved models. Success = an examiner leaves knowing which models won, why, and that the predictor really works.

## Positioning

A controlled, honest comparison rather than a single-model demo: identical data preparation, features and split for every model, every result kept (including KNN's chance-level 50% on Yelp), and a live predictor backed by the actual 24 saved artifacts.

## Operating Context

- Run locally: FastAPI backend (port 8000) + React/Vite frontend (port 5173); `run-dev.ps1` starts both.
- Used during a viva (presenter scrolls and clicks while explaining) and browsed independently from GitHub.
- An optional, separate "product opinion report" (Reddit or uploaded reviews -> VADER -> Groq LLM report) needs a Groq API key; it is explicitly not part of the academic comparison. Reddit often blocks automated requests, so file upload is the reliable path.

## Capabilities and Constraints

- Results are frozen and read-only in the web app (training endpoints return 403).
- Prediction: `POST /ml/predict` with dataset + model; any of the 24 combinations.
- Endpoints: `/ml/datasets`, `/ml/models`, `/ml/results`, `/ml/summary`, `/ml/predict`, `/health`, `/status`, `/analyze`, `/upload-csv`, `/latest`, `/history`, `/history/{id}`, `/groq/check`, `/sample-csv`.
- Stack: React 19, Vite, Recharts, plain CSS (Tailwind present but not used for the dashboard).

## Brand Commitments

- Every metric, number and finding sentence must stay exactly as the backend reports it, including the "frozen results" and "not a universally best classifier" honesty statements (user confirmed).
- The product report stays clearly secondary to the academic study.

## Evidence on Hand

- `results/results.json` / `results.csv`: the 24 canonical experiment records.
- `models/*.joblib`: the 24 saved classifiers.
- `docs/ACADEMIC_RESULTS.md`, `docs/VIVA_GUIDE.md`, `viva-prep/preparation.md`.
- Future work (started): transformer extension, 3 of 24 runs complete, not part of the frozen baseline — see `docs/PHASE4_BENCHMARK_EXECUTION.md`.
- Student names, roll numbers, college and guide are NOT provided: use clearly marked placeholders (user confirmed); never invent them.

## Product Principles

1. Evidence first: the study's real numbers lead; decoration never obscures them.
2. Honest comparison: show weak results as plainly as strong ones.
3. Demo-ready: everything works and reads well live, on a projector, without explanation.
4. Self-explaining for remote visitors: method and caveats visible in the app.
