"""Baseline oparty na EWOA feature selection i klasyfikatorze KNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize


@dataclass
class EWOAKNNConfig:
    """Konfiguracja eksperymentu EWOA + KNN."""

    n_whales: int = 18
    n_iterations: int = 25
    min_features: int = 3
    penalty_weight: float = 0.01
    k_neighbors: int = 5
    weights: str = "distance"
    metric: str = "minkowski"
    p: int = 2
    scoring: str = "accuracy"
    spiral_constant: float = 1.0
    random_state: int = 42


def _build_knn(config: EWOAKNNConfig) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=config.k_neighbors,
        weights=config.weights,
        metric=config.metric,
        p=config.p,
    )


def _mask_from_position(position: np.ndarray, min_features: int) -> np.ndarray:
    probs = np.clip(position, 0.0, 1.0)
    mask = probs >= 0.5
    if mask.sum() >= min_features:
        return mask

    top_indices = np.argsort(probs)[-min_features:]
    mask[top_indices] = True
    return mask


def _validation_score(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    if scoring == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if scoring == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError("scoring musi być równe 'accuracy' albo 'f1_macro'.")


def _evaluate_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mask: np.ndarray,
    config: EWOAKNNConfig,
) -> dict:
    knn = _build_knn(config)
    knn.fit(X_train[:, mask], y_train)
    y_pred = knn.predict(X_val[:, mask])
    score = _validation_score(y_val, y_pred, config.scoring)
    feature_ratio = float(mask.mean())
    fitness = (1.0 - score) + config.penalty_weight * feature_ratio

    return {
        "fitness": float(fitness),
        "score": float(score),
        "feature_count": int(mask.sum()),
        "y_pred": y_pred,
    }


def run_ewoa_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: EWOAKNNConfig,
) -> dict:
    """Uruchamia uproszczoną binarną wersję Whale Optimization do selekcji cech."""

    rng = np.random.default_rng(config.random_state)
    n_features = X_train.shape[1]
    whales = rng.random((config.n_whales, n_features), dtype=np.float32)

    best_position: np.ndarray | None = None
    best_mask: np.ndarray | None = None
    best_result: dict | None = None
    history: list[dict] = []

    for iteration in range(1, config.n_iterations + 1):
        for whale_idx in range(config.n_whales):
            current_mask = _mask_from_position(whales[whale_idx], config.min_features)
            current_result = _evaluate_subset(
                X_train, y_train, X_val, y_val, current_mask, config
            )

            if best_result is None or current_result["fitness"] < best_result["fitness"]:
                best_position = whales[whale_idx].copy()
                best_mask = current_mask.copy()
                best_result = current_result

        assert best_position is not None and best_mask is not None and best_result is not None
        history.append(
            {
                "iteration": iteration,
                "best_fitness": best_result["fitness"],
                "best_score": best_result["score"],
                "selected_features": best_result["feature_count"],
            }
        )

        a = 2.0 - 2.0 * (iteration / config.n_iterations)
        for whale_idx in range(config.n_whales):
            r1 = rng.random(n_features)
            r2 = rng.random(n_features)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = rng.random()
            l = rng.uniform(-1.0, 1.0)

            if p < 0.5:
                if np.mean(np.abs(A)) < 1.0:
                    reference = best_position
                else:
                    reference = whales[rng.integers(0, config.n_whales)]
                distance = np.abs(C * reference - whales[whale_idx])
                whales[whale_idx] = reference - A * distance
            else:
                distance_to_best = np.abs(best_position - whales[whale_idx])
                whales[whale_idx] = (
                    distance_to_best
                    * np.exp(config.spiral_constant * l)
                    * np.cos(2.0 * np.pi * l)
                    + best_position
                )

            whales[whale_idx] = np.clip(whales[whale_idx], 0.0, 1.0)

    selected_names = [name for name, keep in zip(feature_names, best_mask) if keep]
    return {
        "mask": best_mask,
        "selected_indices": np.flatnonzero(best_mask).tolist(),
        "selected_feature_names": selected_names,
        "best_fitness": best_result["fitness"],
        "best_score": best_result["score"],
        "history": history,
    }


def evaluate_knn_on_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_mask: np.ndarray,
    class_names: list[str],
    config: EWOAKNNConfig,
) -> dict:
    """Trenuje finalny KNN na train+val i liczy metryki na teście."""

    X_fit = np.vstack([X_train, X_val])[:, selected_mask]
    y_fit = np.concatenate([y_train, y_val])
    X_eval = X_test[:, selected_mask]

    knn = _build_knn(config)
    knn.fit(X_fit, y_fit)
    y_pred = knn.predict(X_eval)
    y_proba = knn.predict_proba(X_eval)

    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
    auc_ovr_macro = float(
        roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred) * 100.0),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0) * 100.0),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0) * 100.0),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0) * 100.0),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100.0),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100.0),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100.0),
        "auc_ovr_macro": auc_ovr_macro,
    }

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
    }


def run_ewoa_knn_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    config: EWOAKNNConfig,
) -> dict:
    """Pełny pipeline: selekcja cech EWOA, potem finalna ewaluacja KNN."""

    selection = run_ewoa_feature_selection(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        config=config,
    )
    evaluation = evaluate_knn_on_test(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        selected_mask=selection["mask"],
        class_names=class_names,
        config=config,
    )

    return {
        "selection": selection,
        "metrics": evaluation["metrics"],
        "y_pred": evaluation["y_pred"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "classification_report": evaluation["classification_report"],
        "history": selection["history"],
    }