# Ściąga projektowa — jak to wszystko działa

---

## 1. Gdzie wkleić dataset

**Pobierz dataset stąd:**  
https://www.kaggle.com/datasets/dhoogla/cicmalmem2022

**Po pobraniu wklej plik tutaj:**

```
DL-Project/
└── data/
    └── raw/         ← TU wklejasz plik
        └── Obfuscated-MalMem2022.parquet
```

Format to **Parquet** — skompresowany format kolumnowy, dużo szybszy niż CSV.  
Kod sam go znajdzie — `load_raw_data()` rekurencyjnie szuka plików `.parquet` i `.csv` w folderze `data/raw/`.

**Ważne:** `data/raw/` jest w `.gitignore` — pliki NIE trafią do GitHuba (za duże).

---

## 2. Wielkie pytanie: co w ogóle robimy?

### Problem
Mamy memory dumpy (zrzuty pamięci RAM) z komputerów Windows. Część pochodzi z normalnej pracy systemu (*benign*), część z komputerów zarażonych malware (*spyware, ransomware, trojan*). Chcemy nauczyć komputer odróżniać jedno od drugiego.

### Dane wejściowe
Każdy memory dump to **55 liczb** (cech) — np. liczba uruchomionych wątków, rodzaj wywołań systemowych itp. To już wyekstrahowane przez narzędzie Volatility, nie musimy tego robić sami.

### Nasz pomysł: EWOA + KNN (reprodukcja artykułu)

Artykuł bazowy używa **EWOA** (Enhanced Whale Optimization Algorithm) żeby wybrać **kilka najważniejszych cech** z 55, a potem klasyfikuje próbki algorytmem **KNN** (K-Nearest Neighbors).

```
55 cech  →  EWOA (selekcja cech)  →  ~4 cechy  →  KNN (k=5)  →  Benign/Malware/typ
```

**Dlaczego selekcja cech?**  
- Mniej cech = szybsze obliczenia
- Usuwamy szum (nieistotne cechy)
- Artykuł osiąga **99.987% accuracy** przy zaledwie **~4 cechach** z 55!

---

## 3. Co to jest WOA (Whale Optimization Algorithm)?

Algorytm optymalizacji inspirowany zachowaniem **wielorybów humbaka** podczas polowania. Wieloryby tworzą "sieć bąbelkową" (bubble-net) żeby otoczyć zdobycz.

### Fazy algorytmu

```
1. Inicjalizacja: 20 wielorybów, każdy z losową pozycją (55-wymiarowy wektor)
   → Pozycja = binarna tablica: 1 = cecha wybrana, 0 = pominięta

2. Pętla przez 30 iteracji:
   a) Oblicz fitness każdego wieloryba (= jak dobre są wybrane cechy)
   b) Znajdź najlepszego wieloryba (najniższy fitness)
   c) Dla każdego wieloryba:
      → Losuj r ∈ [0,1]
      → Jeśli r < 0.5: ruch w stronę zdobyczy (best whale)
      → Jeśli r ≥ 0.5: ruch spiralny (bubble-net)
      → Zmniejszaj parametr a (2→0) — mniejsze "skoki"

3. Zwróć najlepszego wieloryba = najlepsza selekcja cech
```

### Fitness (jak oceniamy wieloryba)

$$f(\text{whale}) = (1 - \text{accuracy}_{KNN}) + \alpha \cdot \frac{\text{wybrane cechy}}{55}$$

- Niska accuracy → wysoki fitness (źle)
- Dużo cech → wyższy fitness (kara za zbyt wiele cech)
- `alpha = 0.01` — mała waga kary za cechy (accuracy ważniejsza)

### Konwersja ciągła → binarna

Wieloryby poruszają się w przestrzeni ciągłej (np. 0.73, -1.2, 0.01...).  
Żeby uzyskać binarną selekcję (0/1), stosujemy **sigmoid transfer function**:

$$S(x) = \frac{1}{1 + e^{-x}}$$

Potem losujemy: `cecha_wybrana = random() < S(x_i)`.

---

## 4. Co to jest EWOA (Enhanced WOA)?

EWOA = WOA + **3 ulepszenia** opisane w artykule:

### Ulepszenie 1: OBL (Opposition-Based Learning)

**Problem:** WOA losowo inicjalizuje populację — może zacząć daleko od optimum.  
**Rozwiązanie:** Generujemy 2× populację (oryginalna + "lustrzane odbicie"), bierzemy N najlepszych.

```
wieloryb[i]  = [0.3, 0.8, 0.1, ...]   (oryginalny)
opposite[i]  = [0.7, 0.2, 0.9, ...]   (1 - wieloryb[i])

→ Wybieramy 20 najlepszych z 40 kandydatów
```

### Ulepszenie 2: Mutacja bitowa

**Problem:** WOA może utknąć w minimum lokalnym.  
**Rozwiązanie:** Po każdym ruchu losowo "przełączamy" niektóre bity (0→1 lub 1→0).

- Faza eksploracji (a > 1): wysoka mutacja 10-50%
- Faza eksploatacji (a ≤ 1): niska mutacja 1-9%

### Ulepszenie 3: NSS (Neighborhood Search Strategy)

**Problem:** Wieloryby mogą przegapić dobre rozwiązania "w sąsiedztwie".  
**Rozwiązanie:** Dla każdego wieloryba sprawdzamy sąsiadów w topologii pierścienia i losowo kopiujemy ich cechy.

```
Sąsiedzi wieloryba i (ring topology):
  ← backward: wieloryb (i-1)
  → forward:  wieloryb (i+1)
  
Losowo kopiujemy 1–3 cechy od sąsiada
```

---

## 5. Co to jest KNN (K-Nearest Neighbors)?

Najprostszy klasyfikator — nie ma treningu, nie ma parametrów do uczenia!

**Jak działa:**
1. Dostaję nową próbkę do sklasyfikowania
2. Szukam **k=5** najbliższych sąsiadów w danych treningowych (wg odległości euklidesowej)
3. Głosowanie większościowe: jaka klasa dominuje wśród 5 sąsiadów?
4. Zwracam tę klasę

**Dlaczego k=5?** Artykuł używa k=5. Nieparzyste k unika remisów.

---

## 6. Mapa projektu — każdy plik wyjaśniony

### `src/utils/config.py` — Centralne ustawienia
**Co robi:** Zawiera wszystkie hiperparametry i ścieżki do folderów.  
**Kiedy edytować:** Gdy chcesz zmienić liczbę wielorybów, iteracji, k w KNN itp.

```python
NUM_FEATURES = 55         # liczba cech w datasecie — nie zmieniaj
SEED = 42                 # ziarno losowości — reprodukowalność

WOAConfig:
  n_whales = 20           # rozmiar populacji
  max_iter = 30           # liczba iteracji
  n_neighbors = 5         # k w KNN
  alpha = 0.01            # waga kary za liczbę cech

EWOAConfig (WOAConfig + ekstra):
  use_nss = True          # Neighborhood Search Strategy
```

---

### `src/data/loader.py` — Ładowanie danych
**Co robi:** Wczytuje Parquet/CSV, wyciąga etykiety, normalizuje, dzieli na train/test.

**Kluczowe pojęcia:**

| Pojęcie | Co to znaczy |
|---------|--------------|
| **Normalizacja (Min-Max)** | Skaluje każdą cechę do przedziału [0, 1]. Bez tego cechy z dużymi wartościami dominują. |
| **Train/Test split (80/20)** | Train = trenowanie KNN i optymalizacja (80%). Test = finalna ocena (20%). |
| **Stratified split** | Zachowuje proporcje klas w train i test. |

```python
# Typowe użycie:
from src.data.loader import load_raw_data, preprocess, make_splits

df = load_raw_data()                           # wczytaj .parquet z data/raw/
X, y = preprocess(df, mode="multiclass")       # normalizacja, mapowanie etykiet
X_train, X_test, y_train, y_test = make_splits(X, y)  # 80/20 stratified
```

**Struktura datasetu (kolumny w pliku):**

| Kolumna | Co zawiera |
|---------|------------|
| `Category` | `"Benign"` albo pełna nazwa jak `"Ransomware-Ako-abc123-1.raw"` |
| `Class` | `"Benign"` lub `"Malware"` (uproszczone) |
| pozostałe 55 kolumn | cechy numeryczne — to jest wejście do algorytmu |

**Dwa tryby etykietowania:**
- `mode="binary"` → używa kolumny `Class`: 0 = Benign, 1 = Malware
- `mode="multiclass"` → parsuje `Category`: 0 = Benign, 1 = Ransomware, 2 = Spyware, 3 = Trojan

---

### `src/algorithms/woa.py` — Bazowy WOA
**Co robi:** Implementacja Whale Optimization Algorithm do binarnej selekcji cech.

**Kluczowe metody:**
- `optimize(X_train, y_train)` → uruchamia optymalizację, zwraca dict z wynikami
- `_fitness(binary_position, X, y)` → oblicza fitness wieloryba (niska = lepsza)
- `_sigmoid(x)` → transfer function: ciągłe → prawdopodobieństwo
- `_to_binary(position)` → konwertuje pozycję ciągłą na binarną

**Zwraca dict:**
```python
{
    "selected_features": [2, 5, 10],    # indeksy wybranych cech
    "binary_mask": [0, 0, 1, 0, 0, 1, ...],  # 55-elementowa maska
    "best_fitness": 0.00123,
    "convergence": [0.5, 0.3, 0.1, ...],  # fitness per iteracja
    "n_selected": 3
}
```

---

### `src/algorithms/ewoa.py` — Enhanced WOA (główny algorytm)
**Co robi:** WOA + 3 ulepszenia (OBL, mutacja, NSS). To jest **algorytm z artykułu**.

**Dodatkowe metody vs WOA:**
- `_init_with_obl(X, y)` → inicjalizacja z Opposition-Based Learning
- `_mutation(binary_pos, a)` → mutacja bitowa zależna od fazy (eksploracja/eksploatacja)
- `_nss(binary_positions, fitness_values, idx)` → Neighborhood Search Strategy

---

### `src/evaluation/metrics.py` — Ewaluacja KNN
**Co robi:** Trenuje KNN na wybranych cechach, oblicza metryki, zapisuje wyniki.

```python
from src.evaluation.metrics import evaluate_knn, compare_algorithms

# Ewaluacja jednego algorytmu
result = evaluate_knn(X_train, y_train, X_test, y_test,
                      selected_features=[2, 5, 10],
                      n_neighbors=5, algorithm_name="EWOA")
# result = {"algorithm": "EWOA", "accuracy": 0.999, "f1_macro": 0.998, ...}

# Porównanie wielu algorytmów → tabela
df = compare_algorithms([ewoa_result, woa_result, baseline_result])
```

**Kluczowe metryki:**

| Metryka | Co mierzy | Idealna wartość |
|---------|-----------|-----------------|
| **Accuracy** | % poprawnych predykcji | 1.0 (100%) |
| **F1-score (macro)** | Balans precision/recall, uśredniony po klasach | 1.0 |
| **Precision** | Ze alarmów, ile było prawdziwym malware | 1.0 |
| **Recall** | Ze malware, ile złapaliśmy | 1.0 |
| **N features** | Ile cech wybrał algorytm | Im mniej, tym lepiej (artykuł: ~4) |

---

### `src/evaluation/visualization.py` — Wykresy
**Co robi:** Generuje wykresy, zapisuje do `results/plots/`.

| Funkcja | Co rysuje | Kiedy używać |
|---------|-----------|--------------|
| `plot_convergence` | Krzywe fitness vs iteracja (EWOA vs WOA) | Po optymalizacji — czy algorytm się zbiegł? |
| `plot_feature_selection` | Heatmapa wybranych cech per algorytm | Porównanie: jakie cechy wybrał EWOA vs WOA |
| `plot_accuracy_comparison` | Słupki accuracy + porównanie z artykułem | Grafik do raportu |
| `plot_confusion_matrix` | Macierz: prawdziwe vs przewidywane klasy | Szczegółowa analiza błędów |
| `plot_n_features_comparison` | Ile cech wybrał każdy algorytm | Efektywność selekcji |

---

### `experiments/configs/*.yaml` — Konfiguracja eksperymentów
**Co robi:** Przechowuje hiperparametry w czytelnym formacie.

```yaml
# experiments/configs/ewoa_config.yaml
algorithm: EWOA
ewoa:
  n_whales: 20       # ← zmień na 30 żeby zobaczyć co się stanie
  max_iter: 30       # ← zmień żeby optymalizować dłużej
  n_neighbors: 5     # ← k w KNN
  alpha: 0.01        # ← waga kary za liczbę cech
```

---

## 7. Workflow projekt — krok po kroku

```
ETAP 1: Przygotowanie danych
━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Pobierz dataset → wklej do data/raw/
2. Odpal notebook 01_eda.ipynb
   → Sprawdź rozkład klas, czy są NaN, korelacje cech
   → 58 058 próbek, 55 cech, 4 klasy

ETAP 2: Baseline (punkt odniesienia)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Odpal notebook 02_knn_baseline.ipynb
   → KNN (k=5) na WSZYSTKICH 55 cechach
   → Cel: zobaczyć accuracy bez selekcji cech
   → Sprawdź wpływ k na accuracy

ETAP 3: EWOA + KNN (główny eksperyment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Odpal notebook 03_ewoa_optimization.ipynb
   → Uruchom EWOA (20 wielorybów, 30 iteracji)
   → Uruchom WOA (bazowy, do porównania)
   → Porównaj: ile cech wybrał, jaki accuracy
   → Wizualizacje: krzywe zbieżności, heatmapa cech, confusion matrix

ETAP 4: Porównanie z artykułem
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Tabela: EWOA vs WOA vs KNN-all-55 vs artykuł
   → Accuracy, F1, liczba cech, czas
   → Czy osiągamy ~99.987%? Ile cech wybraliśmy?

ETAP 5: Raport i prezentacja
━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Podsumowanie wyników
   → Wykresy do raportu
   → Wnioski: czy reprodukcja się udała?
```

---

## 8. Typowe błędy i co oznaczają

| Błąd / Objaw | Co to znaczy | Co zrobić |
|---|---|---|
| Fitness nie spada | EWOA utknął w minimum lokalnym | Zwiększ `n_whales` lub `max_iter` |
| Accuracy = 100% na train, niska na test | Overfitting / za mało cech wybranych | Zwiększ `alpha` (kara za mało cech) |
| EWOA bardzo wolny | Za dużo ewaluacji fitness | Zmniejsz `fitness_sample_size` w `_prepare_fitness_data()` |
| Wybrał 0 cech | Algorytm się nie zainicjalizował | Sprawdź seed, sprawdź dane |
| `FileNotFoundError` w loader.py | Brak pliku `.parquet` w `data/raw/` | Pobierz z Kaggle (`dhoogla/cicmalmem2022`) |
| EWOA wybrał 40+ cech | Za niskie `alpha` lub za mało iteracji | Zwiększ `alpha` do 0.05, zwiększ `max_iter` |

---

## 9. Słowniczek

| Termin | Po ludzku |
|--------|-----------|
| **EWOA** | Enhanced Whale Optimization Algorithm — metaheurystyka do selekcji cech |
| **WOA** | Bazowy Whale Optimization Algorithm (bez ulepszeń) |
| **KNN** | K-Nearest Neighbors — klasyfikator szukający k najbliższych sąsiadów |
| **Feature selection** | Wybór podzbioru cech (z 55 → ~4) żeby poprawić/przyspieszyć klasyfikację |
| **Fitness** | Ocena jakości rozwiązania — niższa = lepsza |
| **Metaheurystyka** | Algorytm optymalizacji inspirowany naturą (wieloryby, rój, ewolucja) |
| **OBL** | Opposition-Based Learning — sprytna inicjalizacja populacji |
| **NSS** | Neighborhood Search Strategy — eksploracja sąsiedztwa rozwiązań |
| **Mutacja** | Losowa zmiana bitów w rozwiązaniu — unikanie minimów lokalnych |
| **Sigmoid** | Funkcja S-kształtna: dowolna liczba → (0, 1). Używana do konwersji ciągłe→binarne |
| **Min-Max normalizacja** | Skalowanie cech do [0, 1] |
| **Stratified split** | Podział danych zachowujący proporcje klas |
| **Confusion matrix** | Tabela: prawdziwe vs przewidywane klasy — widać dokładnie gdzie model się myli |
| **F1-score** | Średnia harmoniczna precision i recall — dobra miara dla niezbalansowanych danych |
| **Accuracy** | % poprawnych predykcji — prosta, ale może być myląca przy niezbalansowanych danych |
