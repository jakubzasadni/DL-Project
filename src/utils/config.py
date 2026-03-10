"""
Konfiguracja eksperymentów dla DL-Project.

Projekt: Reprodukcja artykułu EWOA (IJIES 2024).
Pipeline: EWOA selekcja cech → KNN klasyfikacja → detekcja malware.
Dataset:  CIC-MalMem-2022 (55 cech, 4 klasy / 2 klasy binarne).
"""
import os
from dataclasses import dataclass, field

# --- Ścieżki ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_MODELS_DIR = os.path.join(ROOT_DIR, "results", "models")
RESULTS_PLOTS_DIR = os.path.join(ROOT_DIR, "results", "plots")
RESULTS_METRICS_DIR = os.path.join(ROOT_DIR, "results", "metrics")

# --- Klasy (tryb binarny: Benign vs Malware — zgodne z artykułem) ---
CLASS_NAMES = ["Benign", "Malware"]
NUM_CLASSES = len(CLASS_NAMES)
NUM_FEATURES = 55  # Liczba cech w CIC-MalMem-2022

# --- Reprodukowalność ---
SEED = 42


# --- Algorytmy optymalizacji (z artykułu) ---
@dataclass
class WOAConfig:
    """Konfiguracja bazowego WOA (do porównań)."""
    n_whales: int = 20
    max_iter: int = 30
    n_neighbors: int = 5       # k w KNN
    alpha: float = 0.01        # waga kary za liczbę cech
    b: float = 1.0             # stała spirali


@dataclass
class EWOAConfig:
    """Konfiguracja Enhanced WOA (główny algorytm)."""
    n_whales: int = 20
    max_iter: int = 30
    n_neighbors: int = 5
    alpha: float = 0.01
    b: float = 1.0
    use_nss: bool = True       # Neighborhood Search Strategy


# --- Algorytmy porównawcze (metaheurystyki) ---
COMPARISON_ALGORITHMS = [
    "EWOA",   # Enhanced Whale Optimization (nasz główny)
    "WOA",    # bazowy Whale Optimization
    "PSO",    # Particle Swarm Optimization
    "GA",     # Genetic Algorithm
]
