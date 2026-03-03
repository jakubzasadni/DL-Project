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

Projekt realizowany na przedmiot Deep Learning. Zadaniem jest budowa systemu detekcji
malware na memory dumpach (dataset CIC-MalMem-2022) przy użyciu metod opartych na
Autoencoderach (AE). Artykuł bazowy (Base.pdf) opisuje podejście EWOA+KNN — nasz projekt
bada głębokie alternatywy jako odpowiedź na "Future Work" z tego artykułu.

Przed przystąpieniem do pracy zawsze zapoznaj się z:
1. `README.md` — pełna koncepcja projektu, architektura, plan realizacji
2. `.claude/rules/Base.pdf` — artykuł inspiracyjny (EWOA, CIC-MalMem-2022, wyniki do porównania)

## Informacje ogólne

- **Zespół**: 4 osoby
- **Czas realizacji**: ~1 miesiąc
- **Język**: Python 3.10+
- **Framework DL**: PyTorch (preferowany)
- **Dataset**: CIC-MalMem-2022 (55 cech, 4 klasy: benign/spyware/ransomware/trojan)

## Zasady kodowania

### 1. Styl i konwencje
- Stosuj **PEP 8** we wszystkich plikach Python
- Nazwy klas: `PascalCase` (np. `VariationalAutoencoder`)
- Nazwy funkcji/zmiennych: `snake_case` (np. `reconstruction_loss`)
- Stałe konfiguracyjne: `UPPER_CASE` w `src/utils/config.py`
- Każdy moduł powinien mieć docstring na początku pliku

### 2. Modele PyTorch
- Wszystkie modele dziedziczą po `nn.Module`
- Metoda `forward()` musi być czytelna i zawierać komentarze dla każdego etapu
- Rozdziel enkoder i dekoder jako osobne `nn.Sequential` lub podmoduły
- Do VAE: osobne metody `encode()`, `reparameterize()`, `decode()`
- Używaj `BatchNorm1d` i `Dropout` dla regularyzacji

### 3. Trening
- Konfiguracja eksperymentów przez pliki YAML w `experiments/configs/`
- Wszystkie hiperparametry muszą być konfigurowalne (nie hardkodowane)
- Zapisuj checkpointy modeli do `results/models/`
- Loguj metryki per epoka (train loss, val loss, reconstruction error)
- Używaj `torch.manual_seed` dla reprodukowalności

### 4. Ewaluacja
- Zawsze obliczaj: Accuracy, F1-macro, AUC-ROC, Precision, Recall
- Dla anomaly detection: wyznaczaj próg na zbiorze walidacyjnym (nie testowym!)
- Wykresy zapisuj do `results/plots/` w formacie PNG (dpi=150)
- Wyniki metryk zapisuj do `results/metrics/` jako CSV

### 5. Notebooki Jupyter
- Notebooki służą do eksploracji i prezentacji — nie do produkcyjnego kodu
- Logika biznesowa trafia do `src/`, notebooki tylko ją importują i wizualizują
- Numeruj notebooki: `01_eda.ipynb`, `02_baseline.ipynb`, itd.
- Na początku każdego notebooka: opis celu, importy, ładowanie danych

### 6. Efektywność
- Używaj DataLoader z `num_workers > 0` na GPU
- Normalizacja danych w klasie Dataset, nie poza nią
- Unikaj pętli Python tam, gdzie można użyć operacji wektorowych (numpy/torch)

## Architektura modeli — wytyczne

### Podstawowy AE (src/models/autoencoder.py)
```
Encoder: 55 → 128 → 64 → 32 → latent_dim
Decoder: latent_dim → 32 → 64 → 128 → 55
Aktywacje: ReLU (encoder), Sigmoid (ostatnia warstwa dekodera)
```

### VAE (src/models/vae.py)
```
Encoder: 55 → 128 → 64 → (mu, log_var) [latent_dim każdy]
Reparametryzacja: z = mu + eps * exp(0.5 * log_var)
Decoder: identyczny jak w AE
Strata: MSE (rekonstrukcja) + beta * KL-divergence
```

### AE + Klasyfikator (src/models/classifier.py)
```
Zamrożony enkoder AE → latent_dim → 64 → 4 (softmax)
Trenuj w dwóch fazach: najpierw AE, potem fine-tune klasyfikatora
```

## Dane — ważne zasady

- Dataset CIC-MalMem-2022 pobieramy ze strony: https://www.unb.ca/cic/datasets/malmem-2022.html
- Surowe dane lądują w `data/raw/` — NIE commituj do gita (>100 MB)
- Przetworzone pliki `.npy` lub `.pkl` w `data/processed/`
- Dodaj `data/raw/` i `data/processed/` do `.gitignore`
- Dla anomaly detection AE: trening TYLKO na próbkach benign
- Podział: 80% train / 20% test (stratified split)

## Wyniki do pobicia (z artykułu Base.pdf)

| Metryka | EWOA+KNN (artykuł) | Nasz cel |
|---------|-------------------|----------|
| Accuracy | 0.99987 | ≥ 0.990 |
| Avg. features | 3.97 / 55 | Latent dim ≤ 8 |
| Avg. time | 43.19 s | Porównywalne |

Komparacja z tymi wynikami jest OBOWIĄZKOWYM elementem projektu.

## Podział pracy (orientacyjny)

| Osoba | Zakres |
|-------|--------|
| Osoba 1 | EDA, preprocessing, data pipeline (`src/data/`) |
| Osoba 2 | Modele AE, VAE (`src/models/`) |
| Osoba 3 | Trening, eksperymenty, YAML configs (`src/training/`) |
| Osoba 4 | Ewaluacja, wizualizacje, raport (`src/evaluation/`) |

## Priorytety implementacji

1. **Najpierw**: podstawowy AE anomaly detection (to rdzeń projektu)
2. **Potem**: VAE + klasyfikacja wieloklasowa
3. **Opcjonalnie**: LSTM-AE (dane jako sekwencje 10 dumpów)
4. **Zawsze**: porównanie z baseline z artykułu


