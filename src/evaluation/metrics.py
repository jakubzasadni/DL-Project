"""
Obliczanie metryk ewaluacyjnych dla modeli detekcji malware.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, classification_report
)
import torch
from torch.utils.data import DataLoader

from src.utils.config import CLASS_NAMES, RESULTS_METRICS_DIR
import os


def get_reconstruction_errors(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Oblicza błędy rekonstrukcji dla wszystkich próbek z DataLoadera.
    
    Returns:
        errors: np.ndarray shape (N,) — MSE rekonstrukcji per próbka
        labels: np.ndarray shape (N,) — prawdziwe etykiety
    """
    model.eval()
    all_errors, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            errors = model.reconstruction_error(x).cpu().numpy()
            all_errors.append(errors)
            all_labels.append(y.numpy())
    return np.concatenate(all_errors), np.concatenate(all_labels)


def find_threshold(errors_benign: np.ndarray, percentile: float = 95.0) -> float:
    """Wyznacza próg detekcji anomalii na podstawie percentyla błędów benign.
    
    Args:
        errors_benign: Błędy rekonstrukcji próbek benign z zbioru walidacyjnego.
        percentile: Percentyl (domyślnie 95. — 5% FPR na danych benign).
    
    Returns:
        threshold: Próg — próbki z error > threshold → malware
    """
    return float(np.percentile(errors_benign, percentile))


def evaluate_anomaly_detection(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    experiment_name: str = "ae_anomaly",
) -> dict:
    """Ewaluacja modelu anomaly detection przy danym progu.
    
    Labels: 0=Benign, 1+=Malware (binaryzacja).
    """
    y_true_binary = (labels != 0).astype(int)
    y_pred_binary = (errors > threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "f1_macro": f1_score(y_true_binary, y_pred_binary, average="macro"),
        "precision": precision_score(y_true_binary, y_pred_binary),
        "recall": recall_score(y_true_binary, y_pred_binary),
        "auc_roc": roc_auc_score(y_true_binary, errors),
        "threshold": threshold,
    }

    print(f"\n=== {experiment_name} ===")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.6f}")

    # Zapis do CSV
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_METRICS_DIR, f"{experiment_name}_metrics.csv"), index=False
    )
    return metrics


def evaluate_classifier(
    model,
    loader: DataLoader,
    device: torch.device,
    experiment_name: str = "ae_classifier",
) -> dict:
    """Ewaluacja klasyfikatora wieloklasowego."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    print(f"\n=== {experiment_name} ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_METRICS_DIR, f"{experiment_name}_metrics.csv"), index=False
    )
    return metrics
