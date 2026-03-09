"""
Konfiguracja eksperymentów dla DL-Project.
Wszystkie hiperparametry i ścieżki definiowane tutaj.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

# --- Ścieżki ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_MODELS_DIR = os.path.join(ROOT_DIR, "results", "models")
RESULTS_PLOTS_DIR = os.path.join(ROOT_DIR, "results", "plots")
RESULTS_METRICS_DIR = os.path.join(ROOT_DIR, "results", "metrics")

# --- Klasy ---
CLASS_NAMES = ["Benign", "Spyware", "Ransomware", "Trojan"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
NUM_FEATURES = 55  # Liczba cech w CIC-MalMem-2022

# --- Reprodukowalność ---
SEED = 42


@dataclass
class AEConfig:
    """Konfiguracja podstawowego Autoencodera."""
    input_dim: int = NUM_FEATURES
    hidden_dims: list = field(default_factory=lambda: [128, 64, 32])
    latent_dim: int = 8
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    weight_decay: float = 1e-5


@dataclass
class VAEConfig:
    """Konfiguracja Variational Autoencodera."""
    input_dim: int = NUM_FEATURES
    hidden_dims: list = field(default_factory=lambda: [128, 64])
    latent_dim: int = 8
    beta: float = 1.0  # Waga KL-divergence
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    weight_decay: float = 1e-5


@dataclass
class ClassifierConfig:
    """Konfiguracja AE + głowica klasyfikacyjna."""
    latent_dim: int = 8
    num_classes: int = NUM_CLASSES
    hidden_dim: int = 64
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    freeze_encoder: bool = True
