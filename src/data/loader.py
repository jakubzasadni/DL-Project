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

from src.utils.config import DATA_RAW_DIR, CLASS_NAMES, SEED


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


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Podział danych na train/test (stratified).

    Zgodne z artykułem: 80% train, 20% test.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    print(f"  Split: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test
