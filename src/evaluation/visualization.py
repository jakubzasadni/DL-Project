"""
Wizualizacje dla DL-Project — EWOA feature selection + KNN.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.config import CLASS_NAMES, RESULTS_PLOTS_DIR


def plot_convergence(
    convergence_dict: dict[str, list[float]],
    save_name: str = "convergence",
) -> None:
    """Krzywe zbieżności (fitness vs iteracja) dla wielu algorytmów.

    Args:
        convergence_dict: {"EWOA": [f0, f1, ...], "WOA": [...], ...}
    """
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c"]
    for (name, conv), color in zip(convergence_dict.items(), colors):
        ax.plot(range(len(conv)), conv, label=name, linewidth=2, color=color)

    ax.set_xlabel("Iteracja", fontsize=12)
    ax.set_ylabel("Wartość fitness", fontsize=12)
    ax.set_title("Krzywe zbieżności algorytmów", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_selection(
    results: dict[str, list[int]],
    feature_names: list[str] | None = None,
    save_name: str = "feature_selection",
) -> None:
    """Heatmapa wybranych cech per algorytm.

    Args:
        results: {"EWOA": [2, 5, 10], "WOA": [2, 3, 10, 15], ...}
        feature_names: lista nazw 55 cech (opcjonalna)
    """
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    algo_names = list(results.keys())
    n_features = 55

    matrix = np.zeros((len(algo_names), n_features), dtype=int)
    for i, name in enumerate(algo_names):
        for feat in results[name]:
            if feat < n_features:
                matrix[i, feat] = 1

    fig, ax = plt.subplots(figsize=(18, max(3, len(algo_names) * 0.8)))
    sns.heatmap(
        matrix, cmap="YlOrRd", cbar_kws={"label": "Wybrana (1) / Pominięta (0)"},
        yticklabels=algo_names, ax=ax, linewidths=0.3,
    )
    if feature_names:
        ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    else:
        ax.set_xlabel("Indeks cechy", fontsize=11)
    ax.set_title("Selekcja cech — porównanie algorytmów", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_accuracy_comparison(
    results: list[dict],
    paper_accuracy: float = 0.99987,
    save_name: str = "accuracy_comparison",
) -> None:
    """Wykres słupkowy: accuracy naszych algorytmów vs artykuł.

    Args:
        results:         lista dict-ów z evaluate_knn()
        paper_accuracy:  accuracy z artykułu (EWOA+KNN = 99.987%)
    """
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    names = [r["algorithm"] for r in results] + ["Artykuł\n(EWOA+KNN)"]
    accs = [r["accuracy"] for r in results] + [paper_accuracy]
    n_feats = [r["n_features"] for r in results] + [4]

    colors = ["#3498db"] * len(results) + ["#e74c3c"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, acc, nf in zip(bars, accs, n_feats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{acc:.4f}\n({nf} cech)", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Porównanie dokładności — nasze wyniki vs artykuł", fontsize=13, fontweight="bold")
    ax.set_ylim(min(accs) - 0.02, 1.005)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    algorithm_name: str = "EWOA",
    save_name: str = "confusion",
) -> None:
    """Macierz pomyłek dla klasyfikacji wieloklasowej."""
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0],
                linewidths=0.5)
    axes[0].set_title("Macierz pomyłek (liczby)", fontweight="bold")
    axes[0].set_ylabel("Prawdziwa klasa")
    axes[0].set_xlabel("Predykowana klasa")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1],
                linewidths=0.5, vmin=0, vmax=100)
    axes[1].set_title("Macierz pomyłek (%)", fontweight="bold")
    axes[1].set_ylabel("Prawdziwa klasa")
    axes[1].set_xlabel("Predykowana klasa")

    plt.suptitle(f"{algorithm_name} + KNN — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_n_features_comparison(
    results: list[dict],
    paper_n_features: float = 3.97,
    save_name: str = "n_features_comparison",
) -> None:
    """Wykres: ile cech wybrał każdy algorytm vs artykuł."""
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    names = [r["algorithm"] for r in results] + ["Artykuł\n(EWOA)"]
    n_feats = [r["n_features"] for r in results] + [paper_n_features]

    colors = ["#3498db"] * len(results) + ["#e74c3c"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, n_feats, color=colors, edgecolor="black", linewidth=0.5)

    for bar, nf in zip(bars, n_feats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{nf:.1f}" if isinstance(nf, float) else str(nf),
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Liczba cech", fontsize=12)
    ax.set_title("Redukcja cech — porównanie algorytmów", fontsize=13, fontweight="bold")
    ax.axhline(55, color="gray", linestyle="--", alpha=0.5, label="Wszystkie cechy (55)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches="tight")
    plt.show()
