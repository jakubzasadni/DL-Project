"""
Wizualizacje dla DL-Project.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.utils.config import CLASS_NAMES, RESULTS_PLOTS_DIR


def plot_training_history(history: dict, title: str = "Training History", save_name: str = "training") -> None:
    """Wykres straty treningowej i walidacyjnej."""
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}_loss.png"), dpi=150)
    plt.show()


def plot_reconstruction_error_distribution(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_name: str = "rec_error_dist",
) -> None:
    """Rozkład błędów rekonstrukcji: benign vs malware z zaznaczonym progiem."""
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))

    label_names = {0: "Benign", 1: "Spyware", 2: "Ransomware", 3: "Trojan"}
    colors = ["green", "red", "orange", "purple"]

    for cls_idx, color in zip(range(4), colors):
        mask = labels == cls_idx
        if mask.any():
            ax.hist(errors[mask], bins=80, alpha=0.5, label=label_names[cls_idx],
                   color=color, density=True)

    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.5f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution: Benign vs Malware")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}.png"), dpi=150)
    plt.show()


def plot_tsne_latent_space(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    save_name: str = "tsne_latent",
) -> None:
    """Wizualizacja przestrzeni latentnej t-SNE (2D)."""
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    print("Obliczam t-SNE... (może chwilę potrwać)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(latent_vectors)

    colors = ["green", "red", "orange", "purple"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = labels == cls_idx
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=name,
                  alpha=0.4, s=10)
    ax.set_title("t-SNE Visualization of Latent Space")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}.png"), dpi=150)
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_name: str = "confusion") -> None:
    """Macierz konfuzji dla klasyfikacji wieloklasowej."""
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, f"{save_name}.png"), dpi=150)
    plt.show()
