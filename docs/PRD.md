# Product Requirements Document

## 1. Overview

**Project:** Sentiment Analysis — Comparative Machine Learning Study
**Type:** 7th-semester minor project (academic)
**One-line summary:** A reproducible comparison of six classical ML classifiers across four public sentiment datasets, presented through a live dashboard with a working predictor, plus an optional separate product-review report feature.

## 2. Problem statement

Sentiment classification is commonly demonstrated with a single model and a single dataset, which makes it hard to tell whether a result reflects the model, the data, or chance. This project instead asks: **under one identical, leak-free pipeline, which classical model performs best, and does that answer change by domain?** It answers this with 24 real experiments (6 models × 4 datasets) rather than a single headline number.

## 3. Goals

- Produce a frozen, reproducible 24-experiment baseline comparing predictive quality and computational cost.
- Make every result — including weak ones — visible and explained, not cherry-picked.
- Let a viewer (examiner, visitor) verify the predictor is real by classifying their own text with the actual saved models, not a canned demo.
- Keep the academic comparison fully separate from the optional LLM-based product-report feature.

## 4. Non-goals

- Production-grade MLOps (model registry, retraining pipelines, monitoring) — out of scope for a minor project.
- State-of-the-art accuracy — the study intentionally uses classical models and TF-IDF, not transformers, for the frozen baseline (see "Future work" below).
- Multi-class or aspect-based sentiment — scope is binary (positive/negative) only.
- Authentication, multi-tenancy, or production deployment hardening.

## 5. Users

| User | Need |
|---|---|
| Viva examiner | Grasp the study, its results, and verify the predictor works — within minutes, on a projector |
| GitHub/live-link visitor | Understand the project unaided, without a presenter, on desktop or mobile |
| The student (presenter) | A dashboard that reads clearly live and survives follow-up questions |

## 6. Functional requirements

### 6.1 Academic comparison (core)
- FR1: Prepare four datasets (IMDb, Sentiment140, Amazon Polarity, Yelp Polarity), each balanced at 10,000 rows (5,000/5,000), deduplicated.
- FR2: Train six classifiers (Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest, Decision Tree, KNN) on one shared TF-IDF feature pipeline per dataset, fitted on training data only.
- FR3: Persist each of the 24 trained classifiers with its own fitted vectorizer as a single `.joblib` artifact.
- FR4: Record accuracy, weighted precision/recall/F1, training time, and inference time for every experiment in `results/results.json` / `.csv`.
- FR5: Serve the frozen results read-only via API (`GET /ml/results`, `GET /ml/summary`); training endpoints return `403` by design.
- FR6: Dashboard presents an overview, a model×dataset accuracy heatmap/table, and a speed-vs-accuracy comparison — all derived from the same stored records, with the actual best-per-dataset model marked (never a styling choice).

### 6.2 Live predictor
- FR7: `POST /ml/predict` accepts `{text, dataset, model}` and returns the saved model's prediction using its own saved vectorizer — no retraining.
- FR8: The dashboard lets a user run one model, or all six models for a dataset at once ("vote"); a tie is shown explicitly as a split decision.
- FR9: If a requested artifact is missing on disk, the API returns a clear error; it must never silently retrain or fall back to a different model.

### 6.3 Optional product report (separate feature)
- FR10: Accept a product name (→ Reddit search) or an uploaded CSV/XLSX of reviews.
- FR11: Score every collected post/review with VADER (rule-based, no training).
- FR12: Summarize the VADER-scored data into a structured report using a Groq-hosted LLM; the LLM never re-labels sentiment and is not one of the six compared models.
- FR13: This feature must work, and be clearly marked as optional/separate, even with no Groq key configured (the academic comparison and predictor must not depend on it).

## 7. Non-functional requirements

- **Reproducibility:** fixed random seed (42) for sampling, splitting, and all applicable classifiers.
- **No data leakage:** deduplication before splitting; train/test text-overlap check; TF-IDF fitted on training data only.
- **Integrity:** results are frozen read-only in the running app; nothing a visitor does can alter the submitted numbers.
- **Responsiveness:** dashboard must be usable on a projector (light mode default) and on a phone (tested at 375px width).
- **Accessibility:** motion (chart reveals, loading placeholders) must fully disable under `prefers-reduced-motion`.
- **Resilience:** the dashboard must show a clear, non-crashing state if the backend is down, loading, or a Groq key is absent.

## 8. Success criteria

- All 24 experiments present and internally consistent across `results.json`, `results.csv`, and the dashboard.
- `pytest`, `pip check`, `npm run lint`, and `npm run build` all pass cleanly.
- An examiner can, unprompted, classify arbitrary text with any of the 24 saved models and get an instant, correct-looking response.
- Every number shown in the UI traces exactly back to the backend's own stored records — no hardcoded or decorative figures.

## 9. Constraints

- Runs on a single development machine (no GPU required for the frozen baseline).
- Groq API key is optional and only gates the product-report feature.
- Reddit's public search is sometimes rate-limited/blocked from shared IPs; the CSV-upload path must always remain a working alternative.

## 10. Future work (explicitly out of scope for this submission)

The repository contains a **started, unfinished** Phase 4 extension that fine-tunes modern transformer models (EuroBERT, Gemma3, Qwen3, etc.) on a free Colab GPU. Only 3 of 24 planned runs are complete. It is not part of the frozen baseline, is not used by the dashboard, and is documented as future work in `docs/PHASE4_BENCHMARK_EXECUTION.md` and the README.
