"""
Ładowanie i preprocessing datasetu CIC-MalMem-2022.

Format pliku (Kaggle: dhoogla/cicmalmem2022):
  - 'Category': "Benign" lub pełna nazwa pliku malware ("Ransomware-Ako-hash-1.raw")
  - 'Class':    "Benign" lub "Malware"  — gotowa etykieta binarna
  - 55 kolumn cech numerycznych (wyniki analizy pamięci z Volatility)

Tryby preprocessingu:
  - 'binary'     → Benign=0, Malware=1           (używa kolumny Class)
  - 'multiclass' → Benign=0, Ransomware=1,
                   Spyware=2, Trojan=3            (wyciąga typ z kolumny Category)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import DATA_RAW_DIR, CLASS_NAMES, SEED


class MalMemDataset(Dataset):
    """Dataset dla CIC-MalMem-2022. Obsługuje tryb anomaly detection (tylko benign)
    oraz tryb klasyfikacji wieloklasowej."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Tablica cech (N, 55), znormalizowana Min-Max.
            labels:   Tablica etykiet (N,).
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def load_raw_data(data_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """Wczytuje surowe dane z katalogu data/raw/.

    Obsługuje:
      - Parquet (.parquet) — format z Kaggle
      - CSV (.csv)         — oryginalny format CIC

    Przeszukuje rekurencyjnie — możesz wrzucić cały wypakowany folder.
    """
    dfs = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.endswith(".parquet"):
                dfs.append(pd.read_parquet(fpath))
            elif fname.endswith(".csv") and not fname.startswith("."):
                dfs.append(pd.read_csv(fpath))

    if not dfs:
        raise FileNotFoundError(
            f"Brak plików w {data_dir}.\n"
            "Pobierz dataset:\n"
            "  Kaggle: https://www.kaggle.com/datasets/dhoogla/cicmalmem2022\n"
            "i wklej do data/raw/"
        )

    df = pd.concat(dfs, ignore_index=True)
    print(f"Wczytano {len(df)} wierszy z {len(dfs)} pliku/plików.")
    print(f"Kolumny ({len(df.columns)}): {list(df.columns[:5])} ...")
    return df


def _extract_type_from_category(category: str) -> str:
    """Wyciąga typ malware z kolumny Category.

    "Ransomware-Ako-abc123-1.raw" → "Ransomware"
    "Benign"                       → "Benign"
    """
    s = str(category).strip()
    if s == "Benign":
        return "Benign"
    return s.split("-")[0]


def preprocess(df: pd.DataFrame, mode: str = "multiclass") -> tuple[np.ndarray, np.ndarray]:
    """Preprocessing datasetu CIC-MalMem-2022.

    Args:
        df:   DataFrame z load_raw_data()
        mode: 'binary'     → Benign=0, Malware=1
              'multiclass' → Benign=0, Ransomware=1, Spyware=2, Trojan=3

    Returns:
        X: np.ndarray (N, 55) — znormalizowane cechy [0, 1]
        y: np.ndarray (N,)    — etykiety numeryczne
    """
    df = df.copy()

    # Kolumny meta — wyrzucamy je z cech
    skip_cols = {"Category", "category", "Class", "class",
                 "Label", "label", "Filename", "filename"}
    feature_cols = [c for c in df.columns if c not in skip_cols]

    if mode == "binary":
        # Kolumna 'Class': "Benign" / "Malware"
        col = next((c for c in df.columns if c.lower() == "class"), None)
        if col is None:
            raise ValueError("Brak kolumny 'Class' — wymagana do trybu binary.")
        df["_label"] = df[col].map({"Benign": 0, "Malware": 1})

    elif mode == "multiclass":
        # Kolumna 'Category': "Benign" lub "Ransomware-...-1.raw" itp.
        col = next((c for c in df.columns if c.lower() == "category"), None)
        if col is None:
            raise ValueError("Brak kolumny 'Category' — wymagana do trybu multiclass.")
        df["_type"] = df[col].apply(_extract_type_from_category)
        df["_label"] = df["_type"].map(
            {"Benign": 0, "Ransomware": 1, "Spyware": 2, "Trojan": 3}
        )

    else:
        raise ValueError(f"Nieznany tryb: '{mode}'. Użyj 'binary' lub 'multiclass'.")

    # Usuń wiersze z nierozpoznaną klasą
    before = len(df)
    df = df.dropna(subset=["_label"])
    dropped = before - len(df)
    if dropped:
        print(f"  [!] Usunięto {dropped} wierszy z nierozpoznaną klasą.")

    X = df[feature_cols].values.astype(np.float32)
    y = df["_label"].values.astype(np.int64)

    # Normalizacja Min-Max → każda cecha w [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    label_names = ["Benign", "Malware"] if mode == "binary" else CLASS_NAMES
    dist = {label_names[i]: int((y == i).sum()) for i in range(len(label_names)) if (y == i).any()}
    print(f"  {len(df)} próbek | {X.shape[1]} cech | tryb: {mode}")
    print(f"  Rozkład klas: {dist}")

    return X, y


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    anomaly_mode: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tworzy DataLoadery: train, val, test (80/10/10 split, stratified).
    
    W trybie anomaly_mode train/val zawiera TYLKO próbki benign (label=0).
    Test zawiera wszystkie klasy.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=SEED
    )

    if anomaly_mode:
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]
        X_val   = X_val[y_val == 0]
        y_val   = y_val[y_val == 0]
        print(f"  Anomaly mode: train={len(X_train)} benign | val={len(X_val)} benign | test={len(X_test)} wszystkich")

    train_loader = DataLoader(MalMemDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(MalMemDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(MalMemDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
