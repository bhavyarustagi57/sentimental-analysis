# Frozen academic experiment configuration

- Task: binary sentiment classification (`0 = negative`, `1 = positive`)
- Datasets: IMDb Movie Reviews, Sentiment140, Amazon Polarity, Yelp Polarity
- Models: Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Machine, Random Forest, Decision Tree, K-Nearest Neighbors
- Prepared samples: up to 10,000 genuine, deduplicated rows per dataset (target 5,000 per class)
- Split: 80% training / 20% test, stratified where class counts permit
- Seed: 42
- Features: train-only TF-IDF, maximum 20,000 features, `(1, 2)` n-grams, `min_df=2`, negation-safe English stop words, sublinear term frequency
- Metrics: Accuracy, weighted Precision, weighted Recall, weighted F1 (`zero_division=0`), classifier training time, test-set inference time
- Timing: `time.perf_counter()` measures only `classifier.fit(...)` and `classifier.predict(...)`, respectively

The final CSV and JSON files are the canonical 4 x 6 baseline. Auxiliary Groq, VADER, Reddit, and product-analysis paths are excluded from this experiment.
