"""Shared, deterministic configuration for the sentiment ML foundation."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
PROCESSED_DATASETS_DIR = DATASETS_DIR / "processed"
DATASET_MANIFEST_PATH = DATASETS_DIR / "manifest.json"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_SAMPLES_PER_DATASET = 10_000
TARGET_SAMPLES_PER_CLASS = MAX_SAMPLES_PER_DATASET // 2
MAX_FEATURES = 20_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MODEL_NAMES = (
    "logistic_regression",
    "multinomial_nb",
    "linear_svc",
    "random_forest",
    "decision_tree",
    "knn",
)
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "multinomial_nb": "Multinomial Naive Bayes",
    "linear_svc": "Linear Support Vector Machine",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
}

# These are the four fixed, public sources for the next academic experiment phase.
# Each processed file is always normalized to exactly: text,label where label is 0/1.
DATASETS = {
    "imdb": {
        "name": "IMDb Movie Reviews",
        "path": PROCESSED_DATASETS_DIR / "imdb.csv",
        "task": "binary_sentiment",
        "source_loader": "huggingface_parquet",
        "source_dataset": "stanfordnlp/imdb",
        "source_revision": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
        "source_split": "train",
        "source_files": ("plain_text/train-00000-of-00001.parquet",),
        "source_url": "https://huggingface.co/datasets/stanfordnlp/imdb",
        "source_train_examples": 25_000,
        "source_text_column": "text",
        "source_label_column": "label",
        "source_label_mapping": {"0": 0, "1": 1},
        "label_mapping": {"0": 0, "1": 1},
    },
    "sentiment140": {
        "name": "Sentiment140",
        "path": PROCESSED_DATASETS_DIR / "sentiment140.csv",
        "task": "binary_sentiment",
        # The original Stanford host can be intermittently unavailable. This
        # public parquet mirror exposes the same 1.6M Tweet corpus and labels.
        "source_loader": "huggingface_parquet",
        "source_dataset": "bdanko/sentiment140",
        "source_revision": "c8aaf1333e9d0cf5886f4eb9506ad917f1d8931c",
        "source_split": "train",
        "source_files": ("data/train-00000-of-00001.parquet",),
        "source_url": "https://huggingface.co/datasets/bdanko/sentiment140",
        "source_origin_url": "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
        "source_train_examples": 1_600_000,
        "source_text_column": "text",
        "source_label_column": "sentiment",
        "source_label_mapping": {"0": 0, "4": 1},
        "label_mapping": {"0": 0, "1": 1},
    },
    "amazon_polarity": {
        "name": "Amazon Polarity",
        "path": PROCESSED_DATASETS_DIR / "amazon_polarity.csv",
        "task": "binary_sentiment",
        "source_loader": "huggingface_parquet",
        "source_dataset": "fancyzhx/amazon_polarity",
        "source_revision": "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289",
        "source_split": "train",
        "source_files": (
            "amazon_polarity/train-00000-of-00004.parquet",
            "amazon_polarity/train-00001-of-00004.parquet",
            "amazon_polarity/train-00002-of-00004.parquet",
            "amazon_polarity/train-00003-of-00004.parquet",
        ),
        "source_url": "https://huggingface.co/datasets/fancyzhx/amazon_polarity",
        "source_train_examples": 3_600_000,
        "source_text_column": "content",
        "source_label_column": "label",
        "source_label_mapping": {"0": 0, "1": 1},
        "label_mapping": {"0": 0, "1": 1},
    },
    "yelp_polarity": {
        "name": "Yelp Polarity",
        "path": PROCESSED_DATASETS_DIR / "yelp_polarity.csv",
        "task": "binary_sentiment",
        "source_loader": "huggingface_parquet",
        "source_dataset": "fancyzhx/yelp_polarity",
        "source_revision": "bbf1c97a1f0cf005e5aded43839fd814654a1557",
        "source_split": "train",
        "source_files": ("plain_text/train-00000-of-00001.parquet",),
        "source_url": "https://huggingface.co/datasets/fancyzhx/yelp_polarity",
        "source_train_examples": 560_000,
        "source_text_column": "text",
        "source_label_column": "label",
        "source_label_mapping": {"0": 0, "1": 1},
        "label_mapping": {"0": 0, "1": 1},
    },
}
