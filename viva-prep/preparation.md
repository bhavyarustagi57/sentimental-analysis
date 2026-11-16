# Sentiment Analysis: Viva Preparation Guide

**Project:** Comparative Machine Learning Study for Binary Sentiment Classification
**Level:** 7th semester minor project
**Stack in one line:** Python + scikit-learn + FastAPI on the back end, React + Vite + Recharts on the front end.

---

## 1. The project in 30 seconds

- It answers one question: **which classical ML classifier is best at deciding if a piece of text is positive or negative?**
- It compares **6 models** on **4 public datasets**. That makes **24 experiments**, all run under one identical, leak-free pipeline.
- Results are **frozen** (never retrained by the web app) and shown in a **React dashboard** with charts, tables and rankings.
- The dashboard also has a **live predictor**. You type a sentence, pick a dataset and a model, and the saved model says *positive* or *negative*.
- There is also an **optional side feature**. It scrapes Reddit for a product, scores sentiment with VADER, and asks the Groq LLM to write a report. It is *not* part of the academic comparison.

---

## 2. Aim and objectives

**Aim:** to compare classical supervised-learning classifiers fairly for sentiment classification, in both predictive quality and computational cost.

**Objectives**

- Prepare four balanced, deduplicated datasets from different domains: movies, tweets, product reviews and restaurant reviews.
- Build **one** preprocessing and TF-IDF pipeline so every model sees identical features.
- Train and evaluate Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest, Decision Tree and KNN.
- Measure **Accuracy, Precision, Recall, F1, training time and inference time**.
- Prevent **data leakage** (the model must never see test information).
- Present the evidence in an interactive dashboard, and let users try the saved models.

**What the study concludes:** no single model wins everywhere. Linear models (Linear SVM and Logistic Regression) are the best overall. Naive Bayes is the fastest strong baseline. Trees and KNN have clear weaknesses.

---

## 3. Tech stack: what, how it is used, and why it was chosen

| Layer | Technology | How it is used here | Why it was chosen |
|---|---|---|---|
| Language | **Python 3.12** | All ML, the API and the scripts | The standard language for ML, with rich libraries |
| ML | **scikit-learn** | TF-IDF, the 6 classifiers, metrics, stratified split | Simple, consistent `fit`/`predict` API, well tested |
| Data | **pandas, NumPy** | Load, clean and sample the CSV datasets | Fast tabular handling |
| Datasets | **Hugging Face Hub** (`huggingface-hub`, `pyarrow`) | Downloads the parquet sources for IMDb, Sentiment140, Amazon and Yelp | Reproducible, versioned public datasets (revision hashes are saved) |
| Model storage | **joblib** | Saves each classifier together with its fitted TF-IDF vectorizer | Efficient for large NumPy/sparse objects |
| Back end | **FastAPI + Uvicorn** | REST API for results, prediction and the optional product report | Fast, automatic validation (Pydantic), async, easy to read |
| Validation | **Pydantic** | Request and response schemas, with a strict schema for the LLM output | Catches bad input early |
| Front end | **React 19 + Vite** | Single-page dashboard | Component model, instant dev server |
| Charts | **Recharts** | Accuracy, F1 and timing charts | Declarative React charts |
| Styling | **Plain CSS** (`src/index.css`, a documented design-token scale) | UI layout and theme, light mode default / dark optional | Full control over the "Counting Night" visual system; Tailwind and Sass are installed but unused for the dashboard itself |
| Sentiment lexicon | **NLTK VADER** | Rule-based scoring of scraped product reviews | Needs no training, works on informal text |
| LLM | **Groq API** (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) | Competitor discovery and the final report | Free tier and very fast inference |
| Config | **python-dotenv, settings.json** | The API key is read from `.env` | Keeps secrets out of the code |
| Testing | **pytest, ESLint** | Unit tests for the ML code, linting for the front end | Quality checks |
| Storage | **JSON files** | `results.json` and `results.csv` for the frozen results, `runtime/sessions.json` for product reports | A database is unnecessary for a prototype |

**Design choice worth remembering:** the web app is **read-only for ML**. `POST /ml/train` returns HTTP 403 on purpose, so the submitted results cannot be changed by accident.

---

## 4. Datasets

| Dataset | Domain | Source (Hugging Face) | Prepared rows |
|---|---|---|---|
| **IMDb** | Movie reviews | `stanfordnlp/imdb` | 10,000 |
| **Sentiment140** | Tweets (short, informal) | `bdanko/sentiment140` (original from Stanford) | 10,000 |
| **Amazon Polarity** | Product reviews | `fancyzhx/amazon_polarity` | 10,000 |
| **Yelp Polarity** | Restaurant reviews | `fancyzhx/yelp_polarity` | 10,000 |

- Each dataset is **balanced**: 5,000 negative (label `0`) and 5,000 positive (label `1`).
- Sampling is **deterministic** (seed 42), and duplicates and empty rows are removed. The dataset manifest records the source revision hashes.
- Output schema is a simple CSV: `text,label`.
- Each prepared file has 10,000 rows, so each split gives **8,000 train / 2,000 test**.

---

## 5. How it works: the ML pipeline

```
Public datasets -> clean + dedupe -> 10,000 balanced rows per dataset
        -> 80/20 stratified split (seed 42)
        -> text cleaning (lowercase, strip HTML/URLs/mentions, keep negations)
        -> TF-IDF fitted on TRAIN only (20,000 features, 1-2 grams)
        -> train classifier -> predict on TEST -> metrics + timings
        -> save classifier + vectorizer (.joblib) and results (.json/.csv)
```

**Step by step**

1. **Prepare data** (`ml/prepare_datasets.py`). Download, normalise to `text,label`, deduplicate, sample 5,000 per class.
2. **Split** (`ml/split_manifest.py`). 80/20, stratified, seed 42. The split is saved so it can be reproduced.
3. **Clean text** (`ml/preprocessing.py`). Lowercase, remove HTML, URLs, @mentions and repeated punctuation, and normalise whitespace. **Negations such as "not" and "never" are kept**, because "not good" must not become "good".
4. **Extract features** (`ml/feature_extraction.py`). TF-IDF with `max_features=20000`, `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`. The vectorizer is fitted on **training text only**.
5. **Train and time** (`ml/training.py`, `ml/trainers.py`). `perf_counter()` wraps only `fit` and only `predict`.
6. **Evaluate** (`ml/evaluator.py`). Accuracy plus weighted Precision, Recall and F1.
7. **Persist** (`ml/utils.py`). 24 `.joblib` artifacts in `models/`, and the canonical results in `results/`.

**Leakage controls (a favourite viva topic)**

- Duplicates are removed **before** the split.
- Train and test text are checked to be **disjoint**.
- TF-IDF is **fitted on training data only**, then used to transform the test data.

---

## 6. The six models

| Model | Idea in one line | Settings used | Strength | Weakness |
|---|---|---|---|---|
| **Logistic Regression** | Weighted linear score passed through a sigmoid to give a probability | `max_iter=1000` | Strong, fast, gives probabilities | Linear boundary only |
| **Multinomial Naive Bayes** | Bayes' theorem with a "words are independent" assumption | `alpha=1.0` | Extremely fast (about 10 ms to train) | The independence assumption is unrealistic |
| **Linear SVM** | Finds the widest-margin separating hyperplane | `LinearSVC` defaults | Best on sparse, high-dimensional text | No probability output |
| **Random Forest** | 100 decision trees vote | `n_estimators=100` | Handles non-linear patterns | Slow to train (about 20 s) on sparse text |
| **Decision Tree** | A sequence of if/else splits | defaults | Interpretable | Overfits sparse features |
| **KNN** | Majority vote of the 5 nearest training points | `k=5` | No training cost | Slow prediction, and the curse of dimensionality |

All use `random_state=42` where applicable.

---

## 7. Evaluation metrics

- **Accuracy** = correct predictions / all predictions.
- **Precision** = of the texts predicted positive, how many really are. Weighted average across classes.
- **Recall** = of the truly positive texts, how many were found.
- **F1** = harmonic mean of Precision and Recall.
- **Weighted averaging** weights each class by its support. It is applied consistently, even though the classes are balanced.
- **Training time** is `fit` only. **Inference time** is `predict` on the whole test set (2,000 texts). Both depend on the hardware.

---

## 8. Results (frozen baseline, 24 experiments)

**Best model per dataset** (rule: Accuracy, then F1, then lower inference time)

| Dataset | Best model | Accuracy | F1 |
|---|---|---:|---:|
| IMDb | **Linear SVM** | 90.25% | 90.25% |
| Sentiment140 | **Logistic Regression** | 71.40% | 71.38% |
| Amazon Polarity | **Logistic Regression** | 84.25% | 84.25% |
| Yelp Polarity | **Linear SVM** | 90.35% | 90.35% |

**Overall ranking** (mean of the four datasets)

| Rank | Model | Mean accuracy | Mean train time | Mean inference time |
|---:|---|---:|---:|---:|
| 1 | Linear SVM | 83.69% | 73 ms | 0.99 ms |
| 2 | Logistic Regression | 83.68% | 116 ms | 1.07 ms |
| 3 | Multinomial NB | 82.58% | 10 ms | 2.00 ms |
| 4 | Random Forest | 80.30% | 20.5 s | 209 ms |
| 5 | Decision Tree | 69.80% | 7.4 s | 3.2 ms |
| 6 | KNN | 64.87% | 5 ms | 516 ms |

**Key observations**

- Longer, explicit reviews (IMDb, Yelp) reach about 90%. Short, slangy tweets (Sentiment140) only reach about 71%.
- **KNN on Yelp got exactly 50% accuracy**, which is chance level. Its weighted precision is 25% and F1 is 33.3%, which means it predicted a single class for every test text. This result is kept on purpose, to show the weakness honestly.
- Random Forest is about 200 times slower to train than Naive Bayes, for lower accuracy.

---

## 9. System architecture and API

```
React dashboard (Vite, :5173) --/api proxy--> FastAPI (Uvicorn, :8000)
                                               |-- results.json / results.csv   (frozen results)
                                               |-- models/*.joblib              (24 saved models)
                                               '-- optional: Reddit + VADER + Groq -> runtime/sessions.json
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status and whether a Groq key is configured |
| `GET /ml/datasets`, `GET /ml/models` | Describe the datasets and models |
| `GET /ml/results` | All 24 result records |
| `GET /ml/summary` | Rankings, best-per-dataset and conclusions, derived from the results |
| `POST /ml/predict` | `{text, dataset, model}` returns positive or negative |
| `POST /ml/train`, `/ml/train-all` | **403 Forbidden** by design (results are frozen) |
| `POST /analyze` | Start a product report from Reddit (needs the Groq key) |
| `POST /upload-csv` | Start a product report from an uploaded CSV/Excel file |
| `GET /status`, `/latest`, `/history` | Progress and stored reports |

**Prediction flow:** clean the text, load the matching `.joblib` file, transform with its **own saved vectorizer**, `predict`, and return the label. If the artifact file is missing, it returns a clear error and does *not* silently retrain.

---

## 10. The optional product-report feature (separate from the study)

1. User enters a product name (or uploads a spreadsheet: the first column is the target product, the other columns are competitors).
2. **Reddit scraper** uses Reddit's public JSON search, with no OAuth. A quality gate keeps only posts that mention the exact product and have the right length and language. The light Groq model (`openai/gpt-oss-20b`) discovers competitors.
3. **Preprocessor** strips URLs, HTML, emoji and Reddit markdown, truncates posts to 500 characters, and removes near-duplicates.
4. **VADER router** scores every post. A compound score of at least +0.05 is positive and at most -0.05 is negative, otherwise neutral. The score is mapped to a 0-10 scale.
5. **Groq synthesizer** (`openai/gpt-oss-120b`) writes a structured JSON report. Pydantic validates the JSON, with retries.
6. A **rate limiter** (30 requests per minute) protects the Groq free tier.
7. The report is saved to `runtime/sessions.json` and shown in the dashboard.

> It is *not* one of the six models. The comparison requires no API key.

---

## 11. Limitations and future scope

**Limitations**

- Classical models only, with binary labels (no neutral class).
- 10,000-sample subsets of much larger datasets.
- No hyperparameter tuning and no cross-validation (a single 80/20 split).
- TF-IDF ignores word order and deep context (sarcasm, for example).
- Timings depend on the hardware.

**Future scope**

- Cross-validation and tuning (GridSearch).
- Transformers and contextual embeddings (BERT-style models).
- Full datasets, multilingual and aspect-based sentiment, and explainability (LIME or SHAP).

> **Honest note on the extra "modern transformer" work:** the repository also contains an *unfinished* experiment (Phase 4) that fine-tunes modern transformers on a Colab GPU. Only 3 of its 24 planned runs finished (for example, EuroBERT reached 91.95% on IMDb). It is **not** part of the frozen baseline or the dashboard. If asked, call it *"future work started as an extension."*

---

## 12. Viva questions and answers

### Q1. What is the aim of your project, and what problem does it solve?
- Classify text sentiment as **positive or negative** and **compare six classical ML models** fairly.
- Sentiment analysis turns unstructured opinions (reviews, tweets) into a usable signal for feedback analysis.
- The main contribution is a **controlled comparison**: same data prep, same features and same split for every model, so differences come from the models alone.

### Q2. Why did you use four datasets and balanced 10,000-row samples?
- Four domains (movies, tweets, products, restaurants) show that the best model **depends on the domain**.
- 5,000 per class means accuracy is not inflated by a majority class.
- 10,000 rows keep all 24 experiments runnable on a normal laptop, and the seeded sampling makes the study reproducible.

### Q3. Explain TF-IDF. Why not just count words?
- **TF** (term frequency) is how often a word appears in a document. **IDF** (inverse document frequency) is a down-weight for words that appear in many documents, such as "the".
- The result highlights *distinctive* words and gives a sparse numeric vector the models can use.
- I used unigrams and bigrams (so "not good" is a feature), 20,000 features, `min_df=2` and sublinear TF.
- Raw counts overweight common words, and TF-IDF fixes that.

### Q4. What is data leakage, and how did you prevent it?
- Leakage means information from the test set influences training, which makes accuracy look better than it truly is.
- I (1) removed duplicates **before** splitting, (2) checked that train and test texts do not overlap, and (3) **fitted TF-IDF on training data only**.
- The IDF values and the vocabulary would otherwise carry test information into training.

### Q5. Why an 80/20 stratified split with seed 42?
- 80% gives enough data to learn from. The 20% (2,000 texts) is a large enough, untouched test set.
- **Stratified** keeps the positive/negative ratio the same in both parts.
- A fixed seed makes the split and the results **reproducible**.

### Q6. Why did you keep negation words like "not" in the text?
- Common stop-word lists remove "not", which can **flip the meaning**: "good" and "not good" are opposites.
- I use a negation-safe stop-word list, and bigrams capture phrases like "not good".

### Q7. Explain how Logistic Regression, Linear SVM and Naive Bayes work.
- **Logistic Regression:** a weighted sum of features goes through a sigmoid and gives P(positive). Training minimises log-loss.
- **Linear SVM:** finds the hyperplane with the **maximum margin** between the classes. Only the points near the boundary (support vectors) matter.
- **Multinomial NB:** applies Bayes' theorem with the naive assumption that words are independent given the class. It is just counting, so it is extremely fast.

### Q8. Which model was best, and why do linear models do well on text?
- **Linear SVM** has the best mean accuracy (83.69%). **Logistic Regression** is a very close second (83.68%) and won two datasets.
- TF-IDF vectors are **very high-dimensional and sparse** (20,000 features). In such spaces, classes are often close to linearly separable, so a linear boundary works well and does not overfit.
- This is an observed pattern. It is not a universal proof, so I say "no model is best everywhere".

### Q9. Why did KNN perform so badly, especially on Yelp (50%)?
- In 20,000 dimensions, **distances become less meaningful** (the curse of dimensionality), so neighbours are not truly "similar".
- Also, KNN does no learning. It stores the data and does all the work at prediction time, so inference took about 516 ms on average.
- On Yelp, precision of 25% and F1 of 33% show it effectively predicted **one class for everything**. I kept this result because a study should report all valid experiments and not cherry-pick.

### Q10. Why was Random Forest so slow, and why did Decision Tree do poorly?
- Random Forest trains **100 trees**, each searching many candidate splits among 20,000 features. That took about 20 s, versus about 100 ms for the linear models.
- A single Decision Tree **overfits** the sparse features (mean accuracy 69.8%). Averaging many trees (the forest) helps, but is still behind the linear models.

### Q11. What metrics did you use, and why weighted F1 on balanced data?
- Accuracy, Precision, Recall and F1, plus training and inference time.
- Accuracy alone can hide error types, so I also report Precision, Recall and F1.
- Weighted averaging is applied to **every** experiment so all 24 rows are directly comparable. The classes are balanced, so it is close to the macro average.
- Time metrics show the **cost-quality trade-off**, which matters for deployment.

### Q12. How are the models saved, and how does the prediction feature work?
- Each experiment saves one `.joblib` file containing the **classifier, its fitted vectorizer, and metadata** (dataset, model, label mapping).
- Prediction cleans the text, loads the right file, transforms with the **saved** vectorizer (never a new one), predicts, and returns "positive" or "negative".
- That is why prediction needs no retraining and runs in milliseconds.

### Q13. Explain the architecture. How are the front end and back end connected?
- **React + Vite** dashboard talks to a **FastAPI** server. Vite proxies `/api` to `localhost:8000`, which avoids CORS problems in development.
- FastAPI serves the frozen results (`/ml/results`, `/ml/summary`) and runs predictions (`/ml/predict`). Pydantic validates the input.
- Training endpoints return **403** on purpose, so the submitted results cannot be overwritten.

### Q14. Where do VADER and Groq fit in? Are they part of your comparison?
- **No.** They belong to a separate optional feature that analyses product opinions scraped from Reddit.
- **VADER** is a rule-based lexicon that scores each post (compound score between -1 and +1). **Groq's LLM** then writes a summary report from the scored posts.
- The academic comparison uses only the six trained classical models and needs no API key.

### Q15. What are the limitations, and what would you improve?
- Limitations: binary labels, 10,000-row subsets, no hyperparameter tuning, no cross-validation, and TF-IDF cannot understand context or sarcasm.
- Improvements: **k-fold cross-validation, GridSearch tuning, transformer models such as BERT, full datasets, multilingual and aspect-based sentiment, and explainability (LIME or SHAP).**
- I have started the transformer extension (3 of 24 runs done), and the classical baseline is the finished, frozen part.

### Q16. Why does the dashboard default to light mode, and what does yellow mean on it?
- Light mode is the default because it reads clearly on a projector during a viva; dark mode is an optional toggle for anyone browsing later.
- The accent yellow colour is deliberately scarce: it only marks a dataset's **actual winning cell** in the heatmap or results table, straight from the backend's `best_by_dataset` summary. It is never used as decoration, so a highlighted cell always means a real, declared result.

### Q17. What is the "let all 6 models vote" feature, and what happens on a tie?
- It sends one piece of text to all six saved models trained on the chosen dataset and tallies how many say positive vs. negative.
- A majority shows a winner with the count (e.g. "4 of 6 models say positive"). A 3-3 tie is shown honestly as a **"Split decision"** instead of being forced to a winner — that disagreement is itself a result worth showing, not a bug to hide.

### Q18. How does the product report use a language model, and is that part of the ML comparison?
- VADER (a rule-based lexicon) scores every collected post or review on a -1 to +1 scale; that scoring never involves a trained classifier.
- The Groq-hosted LLM (`openai/gpt-oss-120b`) only reads those VADER scores and writes a short narrative summary/estimate — it does not re-label sentiment and is not one of the six compared models. It stays clearly separate from the frozen 24-experiment study.

### Q19. What do the heatmap and the speed-vs-accuracy chart add beyond the results table?
- The heatmap turns 24 numbers into one glanceable grid (model × dataset, accuracy as colour intensity), with the winning cell per dataset ringed.
- The speed-vs-accuracy chart plots mean training/inference time against mean accuracy per model, which makes the cost/quality trade-off visible at a glance — e.g. Random Forest's accuracy isn't worth its ~20.49s training time, and KNN's low fitting cost comes with ~515.52ms inference time.

---

## 13. Quick cheat sheet (numbers to remember)

| Fact | Value |
|---|---|
| Datasets × models | 4 × 6 = **24 experiments** |
| Rows per dataset | **10,000** (5,000 positive and 5,000 negative) |
| Split | **80/20**, stratified, seed **42** (8,000 train / 2,000 test) |
| TF-IDF | **20,000** features, **1-2 grams**, `min_df=2`, sublinear TF |
| Best overall | **Linear SVM 83.69%** (Logistic Regression 83.68%) |
| Best per dataset | IMDb: SVM 90.25%, Yelp: SVM 90.35%, Amazon: LR 84.25%, Sentiment140: LR 71.40% |
| Fastest to train | Naive Bayes (about 10 ms) |
| Slowest to train | Random Forest (about 20 s) |
| Slowest to predict | KNN (about 516 ms) |
| Worst result | KNN on Yelp: 50% (chance level) |
| Model files | 24 `.joblib` files in `models/` |
| VADER thresholds | at least +0.05 positive, at most -0.05 negative |
| Groq models | `openai/gpt-oss-120b` (report), `openai/gpt-oss-20b` (light tasks) |
| Ports | front end **5173**, back end **8000** |

**Last tip:** if you do not know an answer, give the **reason and the trade-off** rather than a single fact. Examiners reward understanding of *why* far more than memorised numbers.
