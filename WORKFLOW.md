# Ściąga projektowa — jak to wszystko działa

---

## 1. Gdzie wkleić dataset

**Pobierz dataset stąd:**  
https://www.kaggle.com/datasets/dhoogla/cicmalmem2022

**Po pobraniu wklej pliki CSV tutaj:**

```
DL-Project/
└── data/
    └── raw/         ← TU wklejasz plik
        ├── Obfuscated-MalMem2022.parquet
        └── ...
```

Kod sam je znajdzie — szuka pliku w folderze `data/raw/`.  
Folder `data/processed/` zostanie wypełniony automatycznie przez kod.

**Ważne:** `data/raw/` jest w `.gitignore` — pliki NIE trafią do GitHuba (za duże).

---

## 2. Wielkie pytanie: co w ogóle robimy?

### Problem
Mamy memory dumpy (zrzuty pamięci RAM) z komputerów Windows. Część pochodzi z normalnej pracy systemu (*benign*), część z komputerów zarażonych malware (*spyware, ransomware, trojan*). Chcemy nauczyć komputer odróżniać jedno od drugiego.

### Dane wejściowe
Każdy memory dump to **55 liczb** (cech) — np. liczba uruchomionych wątków, rodzaj wywołań systemowych itp. To już wyekstrahowane przez narzędzie Volatility, nie musimy tego robić sami.

### Nasz pomysł: Autoencoder jako detektor anomalii

Wyobraź sobie, że uczysz model jak wygląda **zdrowy** komputer. Potem pokazujesz mu zarażony — model nie potrafi go dobrze "zrozumieć", robi duży błąd. Ten błąd = alarm.

```
Próbka benign  → Autoencoder → Rekonstrukcja ≈ oryginał   (mały błąd → OK)
Próbka malware → Autoencoder → Rekonstrukcja ≠ oryginał   (duży błąd → ALARM)
```

---

## 3. Co to jest Autoencoder (AE)?

```
    Wejście (55 liczb)
         │
    ┌────▼────┐
    │ ENKODER │  ← "Ściska" dane do małej reprezentacji
    └────┬────┘
         │
    [8 liczb]   ← "Latent space" — esencja próbki
         │
    ┌────▼────┐
    │ DEKODER │  ← "Rozpycha" spowrotem do 55 liczb
    └────┬────┘
         │
    Wyjście (55 liczb) ← powinno być ≈ wejście
```

**Dlaczego 55 → 8 → 55?**  
Bo jeśli model potrafi skompresować dane do 8 liczb i z powrotem je odtworzyć — naprawdę rozumie strukturę danych. Ale nauczyliśmy go TYLKO na benign. Malware ma inną strukturę → duży błąd rekonstrukcji.

**Błąd rekonstrukcji (MSE):**
$$MSE = \frac{1}{55} \sum_{i=1}^{55} (x_i - \hat{x}_i)^2$$

To po prostu średnia kwadratów różnic między wejściem a wyjściem.

---

## 4. Co to jest VAE (Variational Autoencoder)?

Ulepszenie AE. Zamiast zapamiętywać jedną konkretną wartość (8 liczb), latent space staje się **rozkładem prawdopodobieństwa** (średnia + odchylenie standardowe).

**Po co?**  
- Przestrzeń latentna jest "ładniejsza" i bardziej regularna
- Lepiej separuje klasy (benign/spyware/ransomware/trojan)
- Można używać do generowania nowych próbek

**Strata VAE = błąd rekonstrukcji + kara za "chaos" w latent space (KL divergence)**

W praktyce: VAE trenuje się tak samo jak AE, ale zwraca lepszy latent space.

---

## 5. Mapa projektu — każdy plik wyjaśniony

### `src/utils/config.py` — Centralne ustawienia
**Co robi:** Zawiera wszystkie hiperparametry i ścieżki do folderów w jednym miejscu.  
**Kiedy edytować:** Gdy chcesz zmienić rozmiar latent space, liczbę epok, learning rate itp.  
**Kluczowe rzeczy:**
```python
NUM_FEATURES = 55        # liczba cech w datasecie — nie zmieniaj
AEConfig.latent_dim = 8  # rozmiar "ściśniętej" reprezentacji — możesz eksperymentować
AEConfig.epochs = 100    # ile razy model przejdzie przez wszystkie dane
AEConfig.learning_rate   # jak szybko model się uczy (za duże = niestabilny, za małe = wolny)
```

---

### `src/data/loader.py` — Ładowanie danych
**Co robi:** Wczytuje CSV, zamienia etykiety tekstowe na liczby, normalizuje dane, dzieli na train/val/test, pakuje do DataLoaderów.  
**Kiedy używać:** Na początku każdego notebooka/skryptu.  
**Kluczowe pojęcia:**

| Pojęcie | Co to znaczy |
|---------|--------------|
| **Normalizacja (Min-Max)** | Skaluje każdą cechę do przedziału [0, 1]. Bez tego cechy z dużymi wartościami dominują nad innymi. |
| **Train/Val/Test split** | Train = uczenie (80%), Val = strojenie (10%), Test = finalna ocena (10%). **Never** ucz się na test! |
| **DataLoader** | "Dozownik" danych — podaje modelowi paczki (batche) zamiast całego datasetu naraz. Oszczędza RAM. |
| **anomaly_mode=True** | Gdy True: train i val zawierają TYLKO próbki benign. Test zawiera wszystkie klasy. |
| **Batch size 256** | Model aktualizuje wagi co 256 próbek, nie po każdej. Szybciej i stabilniej. |

```python
# Typowe użycie:
from src.data.loader import load_raw_data, preprocess, make_dataloaders

df = load_raw_data()                          # wczytaj CSV z data/raw/
X, y = preprocess(df)                         # normalizacja, mapowanie etykiet
train_loader, val_loader, test_loader = make_dataloaders(X, y, anomaly_mode=True)
```

---

### `src/models/autoencoder.py` — Podstawowy Autoencoder
**Co robi:** Definiuje architekturę sieci neuronowej AE.  
**Kiedy używać:** Faza 1 projektu — anomaly detection.

```
Encoder: 55 → 128 → 64 → 32 → 8    (warstwy FC + BatchNorm + ReLU + Dropout)
Decoder:  8 → 32 → 64 → 128 → 55   (odwrotnie)
```

**Kluczowe metody:**
- `forward(x)` → zwraca `(rekonstrukcja, latent_vector)` — główna metoda sieci
- `reconstruction_error(x)` → zwraca MSE per próbka — używane do detekcji anomalii

**Co to jest BatchNorm?** Normalizuje wartości wewnątrz sieci — przyspiesza uczenie i stabilizuje.  
**Co to jest Dropout?** Losowo "wyłącza" neurony podczas treningu — zapobiega overfittingowi (zapamiętywaniu zamiast uczenia się).  
**Co to jest ReLU?** Funkcja aktywacji: `f(x) = max(0, x)` — wprowadza nieliniowość (bez niej sieć byłaby jak mnożenie macierzy).

---

### `src/models/vae.py` — Variational Autoencoder
**Co robi:** Ulepszona wersja AE z probabilistycznym latent space.  
**Kiedy używać:** Faza 2 — gdy chcemy lepszą separację klas i wizualizację t-SNE.

**Dodatkowe metody vs AE:**
- `encode(x)` → zwraca `(mu, log_var)` — parametry rozkładu latentnego
- `reparameterize(mu, log_var)` → losuje próbkę z rozkładu N(mu, sigma²)
- `loss(x, x_hat, mu, log_var)` → strata ELBO = MSE + beta * KL

**Dlaczego log_var zamiast var?**  
Bo log sprawia, że wartości mogą być ujemne (wariancja musi być > 0, log_var może być dowolne) — numerycznie stabilniejsze.

---

### `src/models/classifier.py` — AE + Klasyfikator
**Co robi:** Bierze zamrożony enkoder już wytrenowanego AE i dokłada głowicę klasyfikacyjną do rozróżniania 4 klas.  
**Kiedy używać:** Faza 3 — klasyfikacja wieloklasowa (nie tylko benign vs malware, ale który typ).

```
Wytrenowany Encoder (zamrożony) → 8 liczb → FC(64) → ReLU → FC(4) → Softmax
                                                               ↑
                                              [Benign, Spyware, Ransomware, Trojan]
```

**Dlaczego zamrożony?**  
Enkoder już dobrze reprezentuje dane. Nie chcemy go psuć — uczymy tylko głowicę na labeled data.  
Metoda `unfreeze_encoder()` pozwala potem fine-tunować całość end-to-end.

---

### `src/training/trainer.py` — Pętla treningowa
**Co robi:** Uruchamia trening — epoka po epoce, batch po batchu. Zapisuje najlepszy model.

**Workflow treningu (1 epoka):**
```
for każdy batch w train_loader:
    1. Przepuść batch przez model → oblicz rekonstrukcję
    2. Oblicz loss (błąd rekonstrukcji)
    3. loss.backward() → oblicz gradienty (jak zmienić wagi żeby błąd był mniejszy)
    4. optimizer.step() → zaktualizuj wagi
    
Potem walidacja (bez gradient, tylko sprawdzamy)
Jeśli val_loss najniższy → zapisz model do results/models/
```

**Kluczowe pojęcia:**

| Pojęcie | Co to znaczy |
|---------|--------------|
| **Epoka** | Jedno pełne przejście modelu przez cały dataset treningowy |
| **Gradient** | "Kierunek" w którym trzeba zmienić wagi żeby błąd malał |
| **Backpropagation** | Algorytm liczenia gradientów — propaguje błąd od wyjścia do wejścia |
| **Adam optimizer** | Adaptacyjny algorytm optymalizacji — sam dobiera kroki dla każdego parametru |
| **Overfitting** | Model zapamiętał dane treningowe zamiast nauczyć się reguł — train_loss mały, val_loss duży |
| **Checkpoint** | Zapis wag modelu — bierzemy ten z najniższym val_loss, nie ostatni |

---

### `src/evaluation/metrics.py` — Metryki
**Co robi:** Oblicza jak dobry jest model. Obsługuje dwa tryby.

**Tryb 1: Anomaly Detection**
```python
errors, labels = get_reconstruction_errors(model, test_loader, device)
# errors[i] = MSE rekonstrukcji dla próbki i
# labels[i] = 0 (benign) lub 1/2/3 (malware)

threshold = find_threshold(errors[labels == 0], percentile=95)
# Próg = 95. percentyl błędów benign z VAL setu
# Czyli: 95% próbek benign ma błąd PONIŻEJ progu

metrics = evaluate_anomaly_detection(errors, labels, threshold)
```

**Tryb 2: Klasyfikacja wieloklasowa**
```python
metrics = evaluate_classifier(classifier_model, test_loader, device)
```

**Kluczowe metryki:**

| Metryka | Co mierzy | Idealna wartość |
|---------|-----------|-----------------|
| **Accuracy** | % poprawnych predykcji | 1.0 (100%) |
| **F1-score (macro)** | Balans między precision i recall, uśredniony po klasach | 1.0 |
| **AUC-ROC** | Zdolność separacji klas przy różnych progach | 1.0 |
| **Precision** | Ze wszystkich alarmów, ile było prawdziwym malware | 1.0 |
| **Recall** | Ze wszystkich malware, ile złapaliśmy | 1.0 |

**Dlaczego nie tylko Accuracy?**  
Bo dataset może być niezbalansowany. Jeśli 90% próbek to benign, model mówiący zawsze "benign" ma 90% accuracy — ale jest bezużyteczny jako detektor.

---

### `src/evaluation/visualization.py` — Wykresy
**Co robi:** Generuje wykresy, zapisuje do `results/plots/`.

| Funkcja | Co rysuje | Kiedy używać |
|---------|-----------|--------------|
| `plot_training_history` | Krzywe train/val loss po epokach | Po każdym treningu — sprawdź czy nie ma overfittingu |
| `plot_reconstruction_error_distribution` | Histogram błędów benign vs malware | Weryfikacja czy AE separuje klasy |
| `plot_tsne_latent_space` | 2D wizualizacja przestrzeni latentnej | Zrozumienie co model się nauczył |
| `plot_confusion_matrix` | Macierz: przewidywane vs rzeczywiste klasy | Dla klasyfikatora wieloklasowego |

**Co to jest t-SNE?**  
Algorytm redukcji wymiarowości — bierze 8-wymiarowy latent space i "rzutuje" go na 2D tak żeby zachować podobieństwa. Używamy go żeby zobaczyć czy model dobrze oddziela klasy.

---

### `experiments/configs/*.yaml` — Konfiguracja eksperymentów
**Co robi:** Przechowuje hiperparametry w czytelnym formacie. Zamiast edytować kod — edytujemy YAML.

```yaml
# experiments/configs/ae_anomaly.yaml
model:
  latent_dim: 8        # ← zmień na 4, 16, 32 żeby zobaczyć co się stanie
training:
  epochs: 100          # ← zmień żeby trenować dłużej/krócej
  learning_rate: 0.001
```

---

## 6. Workflow projekt — krok po kroku

```
ETAP 1: Przygotowanie danych
━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Pobierz dataset → wklej do data/raw/
2. Odpal notebook 01_eda.ipynb
   → Sprawdź rozkład klas, czy są NaN, czy cechy są w sensownym zakresie
   → Wizualizacja korelacji cech

ETAP 2: Baseline (punkt odniesienia)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Odpal notebook 02_baseline.ipynb
   → Prosta sieć FC (bez AE) trenowana na labeled data
   → Cel: zobaczyć "dolną poprzeczkę" — co osiągamy bez autoencoder
   → Porównaj z wynikami z artykułu (EWOA+KNN = 99.987%)

ETAP 3: AE Anomaly Detection (główny pomysł)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Odpal notebook 03_autoencoder.ipynb
   → anomaly_mode=True: trenuj AE tylko na benign
   → Sprawdź krzywe loss (czy zbiegają, czy nie ma overfittingu)
   → Wyznacz próg na VAL secie (percentyl 95)
   → Oceń na TEST secie → wypisz metryki
   → Wizualizacja: rozkład błędów rekonstrukcji

ETAP 4: VAE
━━━━━━━━━━━
5. Odpal notebook 04_vae.ipynb
   → Analogicznie do AE, ale model = VAE
   → Dodatkowo: wizualizacja t-SNE latent space
   → Czy VAE separuje klasy lepiej niż AE?

ETAP 5: Klasyfikator wieloklasowy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Załaduj wytrenowany AE/VAE
   → Stwórz AEClassifier(ae)
   → Trenuj na LABELED data (wszystkie 4 klasy)
   → Czy reprezentacja latentna AE jest dobra dla klasyfikacji?

ETAP 6: Porównanie i raport
━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. Notebook 05_comparison.ipynb
   → Tabela: AE vs VAE vs Classifier vs Baseline vs EWOA+KNN (z artykułu)
   → Wnioski: kiedy AE jest lepszy, kiedy gorszy?
```

---

## 7. Typowe błędy i co oznaczają

| Błąd / Objaw | Co to znaczy | Co zrobić |
|---|---|---|
| `train_loss` spada, `val_loss` rośnie | Overfitting — model zapamiętuje | Zwiększ Dropout, zmniejsz model, więcej danych |
| Loss nie spada w ogóle | Za duże LR lub zła inicjalizacja | Zmniejsz learning_rate 10x |
| Loss = NaN | Eksplodujące gradienty lub za duże LR | Sprawdź dane (NaN?), zmniejsz LR |
| Accuracy = 50% (dla 2 klas) | Model losuje | Debug architecture, sprawdź dane |
| t-SNE — wszystko zmieszane | Latent space nie separuje | Zwiększ beta w VAE, trenuj dłużej |
| `FileNotFoundError` w loader.py | Brak plików CSV w `data/raw/` | Pobierz i wklej dataset |

---

## 8. Słowniczek

| Termin | Po ludzku |
|--------|-----------|
| **Sieć neuronowa** | Funkcja matematyczna z milionami parametrów (wag), które uczymy |
| **Wagi (weights)** | Liczby wewnątrz sieci — to właśnie "uczymy" przez trening |
| **Epoka** | Jedno pełne przejście przez dataset treningowy |
| **Batch** | Paczka danych — model widzi 256 próbek naraz, nie jedną |
| **Loss (strata)** | Liczba mówiąca jak bardzo model się myli — chcemy minimalizować |
| **Gradient descent** | Algorytm minimalizowania loss — "jedź w dół zbocza" |
| **Latent space** | Skompresowana reprezentacja danych — "co model myśli o próbce" |
| **Overfitting** | Model zapamiętał dane zamiast nauczyć się reguł |
| **Underfitting** | Model za prosty, nie potrafi uchwycić wzorców |
| **Hiperparametr** | Ustawienie modelu które ty wybierasz (LR, liczba epok) — nie uczony |
| **Parametr** | Wagi sieci — uczone automatycznie przez backprop |
| **Inference** | Użycie wytrenowanego modelu na nowych danych (bez uczenia) |
| **Checkpoint** | Zapisany stan wag modelu w danym momencie treningu |
| **Fine-tuning** | Dalsze uczenie już wytrenowanego modelu na nowym zadaniu |
