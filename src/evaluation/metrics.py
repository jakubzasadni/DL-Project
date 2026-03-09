"""
Obliczanie metryk ewaluacyjnych dla klasyfikatora KNN z wyselekcjonowanymi cechami.

Pipeline: EWOA/WOA/PSO/GA → selekcja cech → KNN → metryki.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from src.utils.config import CLASS_NAMES, RESULTS_METRICS_DIR


def evaluate_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_features: list[int],
    n_neighbors: int = 5,
    algorithm_name: str = "EWOA",
) -> dict:
    """Ewaluacja KNN na wybranych cechach — zwraca pełny zestaw metryk.

    Args:
        X_train, y_train: dane treningowe (znormalizowane)
        X_test, y_test:   dane testowe
        selected_features: indeksy wybranych cech (wynik EWOA/WOA/...)
        n_neighbors:      k w KNN
        algorithm_name:   nazwa algorytmu (do zapisu)

    Returns:
        dict z metrykami
    """
    X_tr = X_train[:, selected_features]
    X_te = X_test[:, selected_features]

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_tr, y_train)
    y_pred = knn.predict(X_te)

    metrics = {
        "algorithm": algorithm_name,
        "n_features": len(selected_features),
        "selected_features": selected_features,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Wyświetl
    print(f"\n{'='*55}")
    print(f"  {algorithm_name} + KNN (k={n_neighbors})")
    print(f"{'='*55}")
    print(f"  Wybrane cechy:  {len(selected_features)} / 55")
    print(f"  Accuracy:       {metrics['accuracy']:.5f}")
    print(f"  F1-macro:       {metrics['f1_macro']:.5f}")
    print(f"  Precision:      {metrics['precision_macro']:.5f}")
    print(f"  Recall:         {metrics['recall_macro']:.5f}")
    print(f"{'='*55}")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Zapis do JSON
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_METRICS_DIR, f"{algorithm_name.lower()}_results.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Zapisano: {save_path}")

    return metrics


def compare_algorithms(results: list[dict]) -> pd.DataFrame:
    """Tworzy tabelę porównawczą wielu algorytmów.

    Args:
        results: lista dict-ów z evaluate_knn()

    Returns:
        DataFrame z porównaniem
    """
    rows = []
    for r in results:
        rows.append({
            "Algorytm": r["algorithm"],
            "Cechy": r["n_features"],
            "Accuracy": f"{r['accuracy']:.5f}",
            "F1-macro": f"{r['f1_macro']:.5f}",
            "Precision": f"{r['precision_macro']:.5f}",
            "Recall": f"{r['recall_macro']:.5f}",
        })

    df = pd.DataFrame(rows)
    print("\n=== PORÓWNANIE ALGORYTMÓW ===")
    print(df.to_string(index=False))

    # Zapis
    save_path = os.path.join(RESULTS_METRICS_DIR, "comparison.csv")
    df.to_csv(save_path, index=False)
    print(f"\nZapisano: {save_path}")

    return df
