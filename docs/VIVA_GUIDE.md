# Viva Guide — Sentiment Analysis Comparative ML Study

Use these as short talking points, not scripts to recite word-for-word.

## Project basics

**What problem does this project solve?**  It classifies review/post sentiment as positive or negative and compares six classical ML models under the same reproducible pipeline.

**Why sentiment analysis?**  It turns unstructured user opinions into an interpretable signal that can support feedback analysis and product/content understanding.

**What is supervised learning?**  A model learns a mapping from labelled examples. Here, text is the input and the known sentiment label is the target.

**Why binary classification?**  The study deliberately freezes a clear academic task: negative (`0`) versus positive (`1`). It avoids mixing neutral sentiment or ratings with the comparison.

## Datasets and splitting

**Why four datasets?**  They cover movie, social-media, product, and restaurant language, so the comparison can show domain variation rather than relying on one corpus.

**Why 10,000 samples per dataset?**  It is a balanced, computationally practical deterministic subset that allows all 24 experiments to run on a college project machine.

**Why balance the classes?**  With 5,000 examples per class, accuracy and weighted metrics are not dominated by one sentiment class.

**Why remove duplicates before splitting?**  A duplicate review in train and test would let the model effectively see the same text twice, inflating evaluation performance.

**Why an 80/20 split and stratification?**  80/20 leaves substantial training data and an independent test set. Stratification preserves the positive/negative distribution in both sets.

**What is data leakage?**  Information from the test set influencing training. This project prevents it through pre-split deduplication, disjoint normalized text checks, and train-only TF-IDF fitting.

## Preprocessing and TF-IDF

**What preprocessing is performed?**  Text is normalized to lowercase; HTML, URLs, mentions, noise, and repeated punctuation are handled; whitespace is normalized. Negation terms such as “not” are preserved.

**Why preserve negation?**  Negation can reverse sentiment: “good” and “not good” should not be treated as the same signal.

**What is TF?**  Term Frequency measures how often a word or phrase occurs in a document.

**What is IDF?**  Inverse Document Frequency reduces the importance of terms that occur in many documents and gives more value to distinctive terms.

**Why TF-IDF?**  It is an effective, understandable sparse representation for classical text classifiers and makes a fair common feature space for all six models.

**Why fit TF-IDF only on training data?**  Fitting it on test text would reveal test-vocabulary/document-frequency information before evaluation, which is leakage.

**What are unigrams and bigrams?**  Unigrams are one-word features; bigrams are adjacent two-word features such as “not good”. This project uses both.

## Models

**How does Logistic Regression work?**  It learns a weighted linear decision boundary and estimates class probability through a logistic function. It was strong and efficient, winning Amazon (84.25%) and Sentiment140 (71.40%).

**How does Multinomial Naive Bayes work?**  It estimates class likelihoods from feature frequencies using a conditional-independence assumption. It trained very quickly and was competitive, but usually slightly behind the leading linear models.

**How does Linear SVM work?**  It finds a linear boundary with a large margin between classes. It performed best on IMDb (90.25%) and Yelp (90.35%), which is consistent with its suitability for sparse high-dimensional text.

**How does Random Forest work?**  It combines predictions from many decision trees. It can model nonlinear patterns, but its average training time was about 20.49 seconds here—far higher than the linear models.

**How does a Decision Tree work?**  It repeatedly splits on informative features. It is interpretable but can overfit high-dimensional sparse TF-IDF features; its mean accuracy was 69.80%.

**How does KNN work?**  It assigns a class from nearby stored training examples. It has little fitting work but expensive distance calculations at prediction time. Its mean inference time was about 515.52 ms and it performed weakly on Sentiment140/Yelp.

## Metrics

**What is Accuracy?**  The proportion of all predictions that are correct.

**What are Precision and Recall?**  Precision asks how reliable positive predictions are; Recall asks how many actual positives are found.

**What is F1?**  The harmonic mean of Precision and Recall, useful when both types of error matter.

**Why weighted averaging?**  It calculates class metrics weighted by support. The classes are balanced here, but using one consistent method makes every experiment directly comparable.

**What are training and inference time?**  Training time measures only `model.fit`; inference time measures only prediction on the test set. They are hardware-sensitive and not quality metrics by themselves.

## Results and interpretation

**Which model was best overall?**  Linear SVM narrowly had the highest mean Accuracy (83.69%) and F1 (83.69%), while Logistic Regression was extremely close (83.68% Accuracy) and won two datasets. The conclusion is not that either is universally best.

**Why did Linear SVM perform well?**  A plausible explanation is that maximum-margin linear classification works well with high-dimensional sparse TF-IDF vectors. The experiment shows a pattern, not a causal proof.

**Why did KNN perform poorly?**  Text TF-IDF spaces are high-dimensional, so distance can become less informative. KNN also defers work to inference. Yelp KNN reached 50.00% accuracy and 33.33% F1; this valid result is kept as evidence of the trade-off.

**Why was Random Forest slow?**  It trains 100 trees and evaluates many feature splits. In this sparse text setup that required much more fitting time than linear classifiers.

**Why not remove bad results?**  A comparative study should report all valid experiments. Poor performance reveals model limitations and prevents cherry-picking.

**Why did Sentiment140 score lower?**  Its short informal posts, slang, abbreviations, and domain-specific conventions plausibly make fixed word/phrase features harder to separate than longer explicit reviews.

## Engineering questions

**How are models saved?**  Each `.joblib` artifact stores the classifier, its fitted TF-IDF vectorizer, dataset/model identity, label mapping, preprocessing version, and split metadata.

**How does prediction work?**  The API cleans user text, loads the selected matching artifact, transforms text with its saved vectorizer, calls the classifier, and returns positive/negative.

**How are backend and frontend connected?**  React requests FastAPI endpoints through the Vite `/api` proxy. `/ml/results` serves canonical records and `/ml/summary` derives charts and rankings from them.

**Why is Groq not one of the six models?**  Groq/VADER belongs to an optional product-analysis feature. The academic comparison contains only the six frozen classical trained classifiers and requires no API key.

**What happens if an artifact is missing?**  The results dashboard still works from CSV/JSON. Prediction displays a clear message and documents the explicit regeneration command; it does not silently retrain.

## Limitations and future scope

**What are the main limitations?**  Classical models only; binary sentiment; 10,000-sample subsets; no hyperparameter tuning; TF-IDF lacks deep contextual semantics; timing depends on hardware.

**What would you do next?**  Add cross-validation and careful hyperparameter tuning, evaluate contextual embeddings/transformers, use larger datasets, explore multilingual/aspect-based sentiment, and add explainability. These are future work and do not alter the frozen baseline.

**What is the status of the transformer work in the repo?**  Future work (started): transformer extension, 3 of 24 runs complete, not part of the frozen baseline. It fine-tunes six modern transformer models on a free Colab GPU under `docs/PHASE4_BENCHMARK_EXECUTION.md`; only EuroBERT on IMDb (91.95% accuracy) finished validating. The code and its tests stay in the repo for continuation, but no transformer result is used anywhere in the dashboard or the frozen comparison.

## Dashboard design

**Why this visual style for the dashboard?**  Light mode is the default so results are legible on a projector in a viva room; dark mode is an optional toggle. The palette keeps a fixed colour per model across every chart (bars, heatmap, scatter, vote tally) so an examiner can track one model by eye without re-reading a legend each time.

**Why does one colour (the accent) get used so sparingly?**  It is reserved for declared results only — a dataset's actual winning cell in the heatmap, a win pill on a table row — never for decoration. If a cell is highlighted, it means `best_by_dataset` from the backend actually names it the winner; nothing is marked "best" by visual choice.

**What is the six-model vote and what does a split decision mean?**  "Let all 6 models vote" sends the same input text to every saved model for the chosen dataset and tallies how many call it positive vs. negative. A clear majority shows a winner with the count (e.g. "4 of 6 models say positive"); a 3–3 tie is shown explicitly as a "Split decision" rather than forced to a winner, because a tie is itself evidence that classical models disagree on genuinely ambiguous text.

**Why does yellow mark only declared results?**  The accent colour is treated as a scarce signal, not a theme colour: it only appears where the backend's own `best_by_dataset` summary names a winner (heatmap ring, win pill, win row). Anything else that looks important — chart titles, active tabs — uses a separate ink/indigo palette so a viewer never mistakes styling for a result.

**What do the heatmap and the speed-vs-accuracy chart show?**  The heatmap is model × dataset test accuracy on the real 2,000-example held-out set per cell, with each dataset's winning cell ringed. The speed-vs-accuracy chart plots mean training/inference time against mean accuracy per model, so the accuracy-vs-cost trade-off (e.g. Random Forest's ~20.49s training time, KNN's ~515.52ms inference time) is visible at a glance rather than buried in a table.

**How does the product report's language-model step relate to the academic study?**  The optional product report (Reddit posts or uploaded reviews) scores text with VADER, a rule-based lexicon, then asks a Groq-hosted LLM to write a short narrative summary of those VADER scores. The LLM never re-labels sentiment and is not one of the six compared classifiers; its output is an estimate/summary layer on top of VADER's numbers, kept clearly separate from the frozen 24-experiment comparison.
