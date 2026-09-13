"""Factories for the project's fixed six classical classifiers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .config import MODEL_DISPLAY_NAMES, MODEL_NAMES, RANDOM_STATE


def _logistic(_: int | None = None) -> LogisticRegression:
    return LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)


def _linear_svc(_: int | None = None) -> LinearSVC:
    return LinearSVC(random_state=RANDOM_STATE)


def _multinomial_nb(_: int | None = None) -> MultinomialNB:
    return MultinomialNB(alpha=1.0)


def _random_forest(_: int | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)


def _decision_tree(_: int | None = None) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=RANDOM_STATE)


def _knn(_: int | None = None) -> KNeighborsClassifier:
    # KNN supports sparse TF-IDF inputs directly; retaining them avoids a large,
    # unnecessary dense conversion for the 8,000-sample training split.
    return KNeighborsClassifier(n_neighbors=5)


MODEL_FACTORIES: dict[str, Callable[[int | None], Any]] = {
    "logistic_regression": _logistic,
    "multinomial_nb": _multinomial_nb,
    "linear_svc": _linear_svc,
    "random_forest": _random_forest,
    "decision_tree": _decision_tree,
    "knn": _knn,
}


def create_model(model_name: str, train_samples: int | None = None) -> Any:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. Supported models: {', '.join(MODEL_NAMES)}")
    return MODEL_FACTORIES[model_name](train_samples)


def describe_models() -> list[dict[str, str]]:
    return [{"key": key, "name": MODEL_DISPLAY_NAMES[key]} for key in MODEL_NAMES]
