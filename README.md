# Deep Autoencoder-Based Malware Detection via Memory Analysis

> Projekt zaliczeniowy — przedmiot: Deep Learning  
> Zespół: 4 osoby | Czas realizacji: ~1 miesiąc  
> Dataset: [CIC-MalMem-2022](https://www.unb.ca/cic/datasets/malmem-2022.html)  
> Inspiracja: _"EWOA-Based Feature Selection for Malware Detection Using Memory Analysis"_ (IJIES 2024)

---

## Motywacja i zagadnienie

Artykuł bazowy (Base.pdf) prezentuje system detekcji malware oparty na klasycznej selekcji cech (**EWOA**) i klasyfikatorze **KNN**, osiągając dokładność 99.987% na datasecie CIC-MalMem-2022. Autorzy sami wskazują jako kierunek przyszłych badań:

> _"It is also of great interest to investigate the efficacy of deep learning approaches such as recurrent neural networks, long- and short-term memory, and others in detecting malware."_

Nasz projekt odpowiada bezpośrednio na to wyzwanie — zastępujemy ręczną inżynierię cech (EWOA) **reprezentacją uczoną przez Autoencoder** i badamy skuteczność podejść opartych na wykrywaniu anomalii oraz klasyfikacji głębokiej.

---

## Cel projektu

1. Zbudować system detekcji malware oparty na **Autoencoderach (AE)** na memory dumpach z datasetu CIC-MalMem-2022.
2. Porównać skuteczność podejść AE z wynikami bazowymi z artykułu (EWOA + KNN).
3. Zbadać różne warianty AE jako metod uczenia reprezentacji i wykrywania anomalii.
4. Zwizualizować przestrzeń latentną i zrozumieć, czego sieć się uczy.

---

## Dataset: CIC-MalMem-2022

| Kategoria | Opis |
|-----------|------|
| Benign | Normalne procesy systemu Windows |
| Spyware | Oprogramowanie szpiegujące |
| Ransomware | Malware szyfrujące pliki |
| Trojan Horse | Konie trojańskie |

- **29 298** złośliwych memory dumpów + dane benign (50/50 po SMOTE)
- **55 cech** wyekstrahowanych z memory dumpów (Volatility framework)
- Podział: 80% train / 20% test

---

## Podejście: Autoencoder jako detektor anomalii

### Idea główna

Autoencoder trenowany **wyłącznie na próbkach benign** uczy się kompaktowej reprezentacji "normalnego" zachowania procesu. Podczas inferencji:

- Próbki **benign** → mały błąd rekonstrukcji ✓  
- Próbki **malware** → wysoki błąd rekonstrukcji ✗ → wykrycie anomalii

$$\mathcal{L}_{rec} = \|x - \hat{x}\|^2$$

Próg detekcji wyznaczany na zbiorze walidacyjnym.

### Warianty do zbadania

| Model | Opis | Cel |
|-------|------|-----|
| **AE (baseline)** | Fully-connected Autoencoder | Anomaly detection |
| **VAE** | Variational Autoencoder | Rozkład latentny, generatywność |
| **DAE** | Denoising Autoencoder | Robustność na szum w danych |
| **AE + klasyfikator** | Latent space → FC head | Klasyfikacja wieloklasowa (4 klasy) |
| **LSTM-AE** _(opcjonalnie)_ | Autoencoder sekwencyjny | Dane jako sekwencje 10 dumpów |

---

## Architektura bazowego Autoencodera

```
Input (55)
  │
Encoder:
  Dense(128, ReLU) → BatchNorm → Dropout(0.2)
  Dense(64, ReLU)  → BatchNorm
  Dense(32, ReLU)  → [Latent Space z]
  │
Decoder:
  Dense(64, ReLU)
  Dense(128, ReLU)
  Dense(55, Sigmoid) → Output
```

Dla VAE przestrzeń latentna modeluje parametry rozkładu normalnego:

$$\mathcal{L}_{VAE} = \mathcal{L}_{rec} + \beta \cdot D_{KL}(q(z|x) \| p(z))$$

---

## Plan realizacji

### Tydzień 1 — Dane i baseline
- [x] Pobranie i eksploracja datasetu CIC-MalMem-2022
- [x] Preprocessing: normalizacja Min-Max, obsługa brakujących wartości
- [x] Implementacja baseline: prosta sieć FC (75.3% acc, `02_baseline.ipynb`)
- [x] Wyznaczenie metryk baseline do porównania (`results/metrics/baseline_results.json`)

### Tydzień 2 — AE Anomaly Detection
- [ ] Implementacja podstawowego Autoencodera (FC)
- [ ] Trening na danych benign
- [ ] Wyznaczenie progu reconstruction error (ROC/percentyl)
- [ ] Ewaluacja: Accuracy, F1, AUC-ROC
- [ ] Wizualizacja: rozkład błędu rekonstrukcji (benign vs malware)

### Tydzień 3 — Warianty i klasyfikacja wieloklasowa
- [ ] Implementacja VAE
- [ ] Implementacja AE + głowica klasyfikacyjna (4 klasy)
- [ ] Opcjonalnie: Denoising AE, LSTM-AE
- [ ] Optymalizacja hiperparametrów (rozmiar latent space, dropout, lr)
- [ ] Ablation study: wpływ wymiaru przestrzeni latentnej

### Tydzień 4 — Ewaluacja i raport
- [ ] Zestawienie porównawcze wszystkich modeli
- [ ] Wizualizacja t-SNE przestrzeni latentnej
- [ ] Analiza błędów (confusion matrix, false positives analysis)
- [ ] Porównanie z wynikami EWOA+KNN z artykułu
- [ ] Przygotowanie raportu / prezentacji

---

## Metryki ewaluacji

| Metryka | Opis |
|---------|------|
| Accuracy | Ogólna dokładność |
| F1-score (macro) | Zbalansowana miara dla klas |
| AUC-ROC | Zdolność separacji klas |
| Reconstruction Error | Dla podejścia anomaly detection |
| Liczba parametrów | Porównanie z EWOA (55 → ~4 cechy) |
| Czas treningu | Porównanie efektywności |

---

## Struktura projektu

```
DL-Project/
├── README.md
├── data/
│   ├── raw/                  # Surowe dane CIC-MalMem-2022
│   └── processed/            # Przetworzone pliki .npy/.csv
├── notebooks/
│   ├── 01_eda.ipynb           # Eksploracja danych
│   ├── 02_baseline.ipynb      # Baseline FC classifier
│   ├── 03_autoencoder.ipynb   # AE anomaly detection
│   ├── 04_vae.ipynb           # Variational Autoencoder
│   └── 05_comparison.ipynb    # Porównanie wszystkich modeli
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Ładowanie i preprocessing danych
│   │   └── transforms.py      # Transformacje, augmentacja
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py     # Podstawowy AE
│   │   ├── vae.py             # Variational AE
│   │   ├── denoising_ae.py    # Denoising AE
│   │   └── classifier.py      # AE + głowica klasyfikacyjna
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Pętla treningowa
│   │   └── losses.py          # Funkcje straty (MSE, ELBO)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Obliczanie metryk
│   │   └── visualization.py   # Wykresy, t-SNE
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Konfiguracja eksperymentów
├── experiments/
│   └── configs/               # Pliki YAML z konfiguracją eksperymentów
├── results/
│   ├── models/                # Zapisane wagi modeli
│   ├── plots/                 # Wykresy i wizualizacje
│   └── metrics/               # CSV z wynikami eksperymentów
├── requirements.txt
└── .claude/
    └── rules/
        ├── instructions.instructions.md
        └── Base.pdf
```

---

## Oczekiwane wyniki i hipotezy

1. **AE anomaly detection** osiągnie porównywalną skuteczność do KNN z artykułu (~99%) przy braku potrzeby ręcznej selekcji cech.
2. **VAE** zapewni lepszą separację klas w przestrzeni latentnej dzięki regularyzacji KL.
3. **AE + klasyfikator** będzie skuteczniejszy w rozróżnianiu typów malware (Spyware/Ransomware/Trojan) niż podejście binarny anomaly detector.
4. Wymiar przestrzeni latentnej (~4-8) będzie zbliżony do liczby cech wybranych przez EWOA (~4 cechy), co potwierdza biologiczną zasadność redukcji wymiaru.

---

## Technologie

| Narzędzie | Zastosowanie |
|-----------|-------------|
| Python 3.10+ | Główny język |
| PyTorch | Implementacja modeli DL |
| scikit-learn | Preprocessing, metryki |
| pandas / numpy | Manipulacja danymi |
| matplotlib / seaborn | Wizualizacja |
| Jupyter Notebook | Eksploracja i prezentacja wyników |

---

## Literatura

1. Alzubi, M. et al. *"EWOA-Based Feature Selection for Malware Detection Using Memory Analysis"*, IJIES, Vol. 17, No. 3, 2024.
2. Goodfellow, I. et al. *Deep Learning*, MIT Press, 2016.
3. Kingma, D. P., Welling, M. *"Auto-Encoding Variational Bayes"*, ICLR 2014.
4. CIC-MalMem-2022 Dataset: https://www.unb.ca/cic/datasets/malmem-2022.html
5. An, J., Cho, S. *"Variational Autoencoder based Anomaly Detection using Reconstruction Probability"*, 2015.
