---
description: Project instructions for DL-Project — Deep Learning course
paths:
  - "src/**/*.py"
  - "notebooks/**/*.ipynb"
  - "*.py"
  - "*.ipynb"
---

# DL-Project — Instrukcje projektowe

## Kontekst projektu

Projekt realizowany na przedmiot Deep Learning. Zadaniem jest **reprodukcja artykułu**
(Base.pdf): system detekcji malware na memory dumpach (dataset CIC-MalMem-2022)
z użyciem **EWOA** (Enhanced Whale Optimization Algorithm) do selekcji cech
i klasyfikatora **KNN** (k=5).

Przed przystąpieniem do pracy zawsze zapoznaj się z:
1. `README.md` — pełna koncepcja projektu, architektura, plan realizacji
2. `.claude/rules/Base.pdf` — artykuł bazowy (EWOA, CIC-MalMem-2022, wyniki do reprodukcji)

## Informacje ogólne

- **Zespół**: 4 osoby
- **Czas realizacji**: ~1 miesiąc
- **Język**: Python 3.12+
- **Framework ML**: scikit-learn (KNN, metryki, scaler)
- **Algorytmy**: EWOA, WOA (numpy), PSO, GA
- **Dataset**: CIC-MalMem-2022 (55 cech, 4 klasy: benign/spyware/ransomware/trojan)

## Zasady kodowania

### 1. Styl i konwencje
- Stosuj **PEP 8** we wszystkich plikach Python
- Nazwy klas: `PascalCase` (np. `EWOA`, `WOA`)
- Nazwy funkcji/zmiennych: `snake_case` (np. `evaluate_knn`)
- Stałe konfiguracyjne: `UPPER_CASE` w `src/utils/config.py`
- Każdy moduł powinien mieć docstring na początku pliku

### 2. Algorytmy optymalizacji (src/algorithms/)
- `WOA` — bazowy Whale Optimization Algorithm
- `EWOA(WOA)` — Enhanced WOA z trzema ulepszeniami: OBL, mutacja, NSS
- Fitness = (1 - accuracy_knn) + alpha * (n_selected / n_features)
- Binarne pozycje wielorybów (sigmoid transfer function)
- Parametry: n_whales=20, max_iter=30, n_neighbors=5, alpha=0.01

### 3. Eksperymenty
- Konfiguracja przez dataclasses w `src/utils/config.py` (WOAConfig, EWOAConfig)
- Wszystkie hiperparametry muszą być konfigurowalne (nie hardkodowane)
- Wyniki zapisuj do `results/metrics/` jako JSON/CSV
- Używaj `SEED = 42` dla reprodukowalności

### 4. Ewaluacja
- Metryki: Accuracy, F1-macro, Precision, Recall, klasyfikacja per klasa
- Wykresy zapisuj do `results/plots/` w formacie PNG (dpi=150)
- Porównanie algorytmów: EWOA vs WOA vs PSO vs GA
- Porównanie z wynikami artykułu (accuracy 99.987%, avg 3.97 cech)

### 5. Notebooki Jupyter
- Notebooki służą do eksploracji i prezentacji — nie do produkcyjnego kodu
- Logika biznesowa trafia do `src/`, notebooki tylko ją importują i wizualizują
- Numeruj notebooki: `01_eda.ipynb`, `02_knn_baseline.ipynb`, itd.
- Na początku każdego notebooka: opis celu, importy, ładowanie danych

### 6. Efektywność
- Unikaj pętli Python tam, gdzie można użyć operacji wektorowych (numpy)
- Do fitness evaluation: subsample danych (domyślnie 3000 próbek)
- KNN: `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)`

## Dane — ważne zasady

- Dataset CIC-MalMem-2022 w formacie Parquet (Kaggle: dhoogla/cicmalmem2022)
- Surowe dane lądują w `data/raw/` — NIE commituj do gita (>100 MB)
- Dodaj `data/raw/` i `data/processed/` do `.gitignore`
- Podział: 80% train / 20% test (stratified split)
- Normalizacja: MinMaxScaler → [0, 1]

## Wyniki do reprodukcji (z artykułu Base.pdf)

| Metryka | EWOA+KNN (artykuł) | Nasz cel |
|---------|:---:|:---:|
| Accuracy | 0.99987 | ≥ 0.999 |
| Avg. features | 3.97 / 55 | ≤ 6 |
| Avg. time | 43.19 s | Porównywalny |

Komparacja z tymi wynikami jest OBOWIĄZKOWYM elementem projektu.

## Priorytety implementacji

1. **Najpierw**: podstawowy AE anomaly detection (to rdzeń projektu)
2. **Potem**: VAE + klasyfikacja wieloklasowa
3. **Opcjonalnie**: LSTM-AE (dane jako sekwencje 10 dumpów)
4. **Zawsze**: porównanie z baseline z artykułu


