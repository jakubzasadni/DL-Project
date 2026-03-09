# EWOA-Based Feature Selection for Malware Detection via Memory Analysis

> Projekt zaliczeniowy — przedmiot: Deep Learning  
> Zespół: 4 osoby | Czas realizacji: ~1 miesiąc  
> Dataset: [CIC-MalMem-2022](https://www.unb.ca/cic/datasets/malmem-2022.html)  
> Artykuł bazowy: _"EWOA-Based Feature Selection for Malware Detection Using Memory Analysis"_ (IJIES 2024)

---

## Cel projektu

Reprodukcja wyników artykułu bazowego (Base.pdf) — system detekcji malware oparty na:

1. **EWOA** (Enhanced Whale Optimization Algorithm) — metaheurystyczna selekcja cech
2. **KNN** (K-Nearest Neighbors, k=5) — klasyfikacja wieloklasowa

Porównanie EWOA z bazowym WOA, PSO oraz GA w zadaniu selekcji cech z datasetu CIC-MalMem-2022.

---

## Dataset: CIC-MalMem-2022

| Kategoria | Liczba próbek | Opis |
|-----------|:---:|------|
| Benign | 29 227 | Normalne procesy Windows |
| Ransomware | 9 815 | Malware szyfrujące pliki |
| Spyware | 9 529 | Oprogramowanie szpiegujące |
| Trojan | 9 487 | Konie trojańskie |

- **58 058** próbek łącznie
- **55 cech** numerycznych (Volatility framework)
- Format: Parquet (Kaggle: `dhoogla/cicmalmem2022`)

---

## Podejście: EWOA + KNN

### Pipeline

```
Dane surowe (55 cech)
  │
  ├─ MinMax normalizacja → [0, 1]
  │
  ├─ EWOA selekcja cech  → ~4 cech (z 55)
  │     ├─ OBL (Opposition-Based Learning) — lepsza inicjalizacja
  │     ├─ Mutacja bitowa — unikanie minimów lokalnych
  │     └─ NSS (Neighborhood Search Strategy) — eksploracja sąsiedztwa
  │
  └─ KNN (k=5) klasyfikacja → Benign / Ransomware / Spyware / Trojan
```

### Funkcja fitness (Eq. 12 z artykułu)

$$f(X) = \alpha \cdot E_R + (1 - \alpha) \cdot \frac{|S|}{|F|}$$

Gdzie:
- $E_R$ — błąd klasyfikacji (1 − accuracy z KNN)
- $|S|$ — liczba wybranych cech
- $|F|$ — łączna liczba cech (55)
- $\alpha = 0.99$ (domyślnie w artykule)

### EWOA — trzy ulepszenia vs WOA

| Ulepszenie | Opis |
|-----------|------|
| **OBL** | Inicjalizacja 2N populacji (oryginalna + opposition), wybranie N najlepszych |
| **Mutacja** | Bit-string mutation — wysoka (10-50%) w eksploracji, niska (1-9%) w eksploatacji |
| **NSS** | Ring-based Neighborhood Search Strategy — losowe przełączanie cech sąsiadów |

---

## Wyniki do reprodukcji (z artykułu)

| Metryka | EWOA+KNN (artykuł) | Nasz cel |
|---------|:---:|:---:|
| Accuracy | 99.987% | ≥ 99.9% |
| Avg. features | 3.97 / 55 | ≤ 6 |
| Avg. time | 43.19 s | Porównywalny |

---

## Struktura projektu

```
DL-Project/
├── README.md
├── data/
│   └── raw/                    # CIC-MalMem-2022 (.parquet)
├── notebooks/
│   ├── 01_eda.ipynb            # Eksploracja danych
│   ├── 02_knn_baseline.ipynb   # KNN na wszystkich 55 cechach
│   └── 03_ewoa_optimization.ipynb  # EWOA selekcja cech + wyniki
├── src/
│   ├── algorithms/
│   │   ├── woa.py              # Bazowy Whale Optimization Algorithm
│   │   └── ewoa.py             # Enhanced WOA (OBL + mutacja + NSS)
│   ├── data/
│   │   └── loader.py           # Ładowanie i preprocessing danych
│   ├── evaluation/
│   │   ├── metrics.py          # Ewaluacja KNN, porównanie algorytmów
│   │   └── visualization.py    # Wykresy: zbieżność, selekcja cech, CM
│   └── utils/
│       └── config.py           # Konfiguracja (WOAConfig, EWOAConfig)
├── experiments/
│   └── configs/                # Pliki YAML z konfiguracją
├── results/
│   ├── plots/                  # Wykresy
│   └── metrics/                # CSV/JSON z wynikami
├── requirements.txt
└── .claude/rules/
    ├── instructions.instructions.md
    └── Base.pdf
```

---

## Plan realizacji

### Tydzień 1 — Dane i baseline
- [x] Pobranie i eksploracja datasetu CIC-MalMem-2022
- [x] Preprocessing: normalizacja Min-Max
- [x] EDA notebook (rozkłady, korelacje, brakujące wartości)
- [ ] KNN baseline na wszystkich 55 cechach

### Tydzień 2 — Implementacja algorytmów
- [x] Implementacja WOA (bazowy Whale Optimization)
- [x] Implementacja EWOA (OBL + mutacja + NSS)
- [ ] Uruchomienie EWOA na pełnym datasecie
- [ ] Walidacja: porównanie z wynikami artykułu

### Tydzień 3 — Porównanie i analiza
- [ ] Implementacja PSO i GA do porównania
- [ ] Zestawienie: EWOA vs WOA vs PSO vs GA
- [ ] Analiza wybranych cech (interpretacja)
- [ ] Confusion matrix, classification report

### Tydzień 4 — Raport
- [ ] Wizualizacje: krzywe zbieżności, porównanie accuracy
- [ ] Porównanie z wynikami artykułu
- [ ] Przygotowanie raportu / prezentacji

---

## Technologie

| Narzędzie | Zastosowanie |
|-----------|-------------|
| Python 3.12 | Główny język |
| scikit-learn | KNN, MinMaxScaler, metryki |
| numpy / pandas | Przetwarzanie danych |
| matplotlib / seaborn | Wizualizacja |
| Jupyter Notebook | Eksploracja i prezentacja |

---

## Literatura

1. Alzubi, M. et al. *"EWOA-Based Feature Selection for Malware Detection Using Memory Analysis"*, IJIES, Vol. 17, No. 3, 2024.
2. Mirjalili, S., Lewis, A. *"The Whale Optimization Algorithm"*, Advances in Engineering Software, 2016.
3. CIC-MalMem-2022 Dataset: https://www.unb.ca/cic/datasets/malmem-2022.html
