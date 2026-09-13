# Academic Results and Interpretation

## Experiment objective

This study evaluates six classical machine-learning classifiers for binary sentiment classification across four real review/post datasets. The evidence is the frozen 24-record baseline in `results/results.json` and `results/results.csv`; all statements below are derived from those records.

## Dataset and experimental setup

IMDb Movie Reviews, Sentiment140, Amazon Polarity, and Yelp Polarity were each prepared as balanced `text,label` datasets of 10,000 genuine examples (5,000 per class). Labels are `0 = negative` and `1 = positive`.

Each dataset used a deterministic seed of 42 and an 80/20 stratified train/test split. Text was cleaned independently, TF-IDF was fitted on training text only (20,000 maximum uni/bi-gram features), and the same fitted vectorizer transformed the test set. Results use Accuracy, weighted Precision, weighted Recall, weighted F1, classifier training time, and test-set inference time. Precision/Recall/F1 use weighted averaging with `zero_division=0`.

## Full 24-experiment results

| Dataset | Model | Accuracy | Precision | Recall | F1 | Train time | Inference time |
|---|---|---:|---:|---:|---:|---:|---:|
| Amazon Polarity | Decision Tree | 71.55% | 71.55% | 71.55% | 71.55% | 8.059 s | 2.858 ms |
| Amazon Polarity | KNN | 73.30% | 73.42% | 73.30% | 73.27% | 4.842 ms | 539.490 ms |
| Amazon Polarity | Linear SVM | 83.75% | 83.75% | 83.75% | 83.75% | 58.676 ms | 0.783 ms |
| Amazon Polarity | Logistic Regression | 84.25% | 84.25% | 84.25% | 84.25% | 96.814 ms | 1.113 ms |
| Amazon Polarity | Multinomial NB | 83.25% | 83.25% | 83.25% | 83.25% | 11.123 ms | 2.059 ms |
| Amazon Polarity | Random Forest | 81.50% | 81.50% | 81.50% | 81.50% | 22.354 s | 181.437 ms |
| IMDb | Decision Tree | 69.75% | 69.78% | 69.75% | 69.74% | 11.570 s | 3.616 ms |
| IMDb | KNN | 83.15% | 83.63% | 83.15% | 83.09% | 8.439 ms | 746.411 ms |
| IMDb | Linear SVM | 90.25% | 90.25% | 90.25% | 90.25% | 118.287 ms | 1.540 ms |
| IMDb | Logistic Regression | 89.00% | 89.00% | 89.00% | 89.00% | 215.905 ms | 1.352 ms |
| IMDb | Multinomial NB | 88.15% | 88.18% | 88.15% | 88.15% | 13.623 ms | 2.947 ms |
| IMDb | Random Forest | 84.60% | 84.66% | 84.60% | 84.59% | 20.359 s | 188.461 ms |
| Sentiment140 | Decision Tree | 65.55% | 65.56% | 65.55% | 65.55% | 3.526 s | 3.222 ms |
| Sentiment140 | KNN | 53.05% | 54.72% | 53.05% | 48.49% | 3.718 ms | 379.481 ms |
| Sentiment140 | Linear SVM | 70.40% | 70.41% | 70.40% | 70.39% | 34.473 ms | 0.653 ms |
| Sentiment140 | Logistic Regression | 71.40% | 71.45% | 71.40% | 71.38% | 43.798 ms | 0.564 ms |
| Sentiment140 | Multinomial NB | 70.30% | 70.30% | 70.30% | 70.30% | 7.846 ms | 1.220 ms |
| Sentiment140 | Random Forest | 69.45% | 69.52% | 69.45% | 69.42% | 19.356 s | 291.803 ms |
| Yelp Polarity | Decision Tree | 72.35% | 72.35% | 72.35% | 72.35% | 6.414 s | 2.973 ms |
| Yelp Polarity | KNN | 50.00% | 25.00% | 50.00% | 33.33% | 4.015 ms | 396.715 ms |
| Yelp Polarity | Linear SVM | 90.35% | 90.38% | 90.35% | 90.35% | 81.406 ms | 0.988 ms |
| Yelp Polarity | Logistic Regression | 90.05% | 90.06% | 90.05% | 90.05% | 108.150 ms | 1.234 ms |
| Yelp Polarity | Multinomial NB | 88.60% | 88.61% | 88.60% | 88.60% | 9.194 ms | 1.762 ms |
| Yelp Polarity | Random Forest | 85.65% | 85.75% | 85.65% | 85.64% | 19.908 s | 172.819 ms |

## Best model per dataset

The transparent selection rule is Accuracy, then F1, then lower inference time, then model key.

| Dataset | Best model | Accuracy | F1 |
|---|---|---:|---:|
| IMDb Movie Reviews | Linear SVM | 90.25% | 90.25% |
| Sentiment140 | Logistic Regression | 71.40% | 71.38% |
| Amazon Polarity | Logistic Regression | 84.25% | 84.25% |
| Yelp Polarity | Linear SVM | 90.35% | 90.35% |

## Overall model ranking and efficiency

| Rank by accuracy | Model | Mean accuracy | Mean F1 | Mean training time | Mean inference time |
|---:|---|---:|---:|---:|---:|
| 1 | Linear SVM | 83.69% | 83.69% | 73.210 ms | 0.991 ms |
| 2 | Logistic Regression | 83.68% | 83.67% | 116.167 ms | 1.066 ms |
| 3 | Multinomial NB | 82.58% | 82.57% | 10.447 ms | 1.997 ms |
| 4 | Random Forest | 80.30% | 80.29% | 20.494 s | 208.630 ms |
| 5 | Decision Tree | 69.80% | 69.80% | 7.392 s | 3.167 ms |
| 6 | KNN | 64.87% | 59.54% | 5.254 ms | 515.524 ms |

Linear SVM narrowly has the highest mean Accuracy/F1. Logistic Regression is essentially tied on aggregate quality and wins two datasets. Decision Tree has the smallest cross-dataset Accuracy standard deviation (about 2.63 percentage points), but at a substantially lower mean score; consistency alone is not a claim of highest predictive quality.

## Dataset-level observations

- IMDb and Yelp have the highest leading scores (90.25% and 90.35%). Their longer, explicit opinion language is consistent with sparse word/phrase features being informative, although this study does not prove that explanation.
- Sentiment140 is more difficult: its best accuracy is 71.40%. Short informal posts, slang, abbreviations, and domain-specific conventions plausibly make fixed TF-IDF representations less discriminative.
- Amazon Polarity is led by Logistic Regression at 84.25%, showing that strong performance is not restricted to movie or restaurant vocabulary.
- KNN falls to 53.05% on Sentiment140 and 50.00% on Yelp. The Yelp outcome is a meaningful chance-level observation and is intentionally retained rather than removed.

## Model-level observations

- **Linear SVM:** The strong IMDb/Yelp outcomes are consistent with the suitability of maximum-margin linear boundaries for high-dimensional sparse TF-IDF vectors. This does not establish universal superiority.
- **Logistic Regression:** It balances strong accuracy with low training/inference cost and leads Amazon and Sentiment140. Its linear probabilistic boundary is a practical match for sparse text features.
- **Multinomial Naive Bayes:** It trains very quickly and remains competitive, while its conditional-independence assumption may limit it relative to the leading linear models on some datasets.
- **Random Forest:** It has respectable predictive scores but is by far the slowest model to train in this experiment. That pattern is consistent with the difficulty tree ensembles can face on high-dimensional sparse features; it does not mean Random Forest is inherently unsuitable for NLP.
- **Decision Tree:** It is comparatively interpretable but trails the stronger models, plausibly because an individual tree can overfit sparse high-dimensional features.
- **KNN:** It has negligible fitting time because it defers work to prediction. Its much slower inference and poor outcomes on Sentiment140/Yelp illustrate sensitivity to high-dimensional distance calculations and the curse of dimensionality.

## Key findings and conclusion

The study demonstrates that model choice is domain-dependent even under one controlled TF-IDF pipeline. Linear SVM and Logistic Regression offer the strongest average predictive performance in these four datasets, while Multinomial NB is a fast competitive baseline. Tree models and KNN reveal important efficiency and generalization trade-offs. Therefore, this experiment supports selecting a model by the dataset and deployment constraint rather than asserting one universally best algorithm.

## Limitations and future scope

The work is limited to classical models, binary labels, deterministic 10,000-sample subsets, no hyperparameter optimization, and TF-IDF rather than contextual embeddings. Runtime measurements are hardware-sensitive. Future work may add cross-validation, tuned models, transformers/contextual embeddings, full datasets, multilingual and aspect-based sentiment, and explainability—without changing this frozen baseline.
