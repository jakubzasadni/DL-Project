"""
Whale Optimization Algorithm (WOA) — bazowa wersja do porównań.

Implementacja algorytmu rojowego do binarnej selekcji cech.
Każdy wieloryb = wektor binarny (1 = cecha wybrana, 0 = pominięta).
Funkcja przystosowania = błąd KNN + kara za liczbę cech.

Referencja:
    Mirjalili, S., Lewis, A. "The Whale Optimization Algorithm",
    Advances in Engineering Software, 2016.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class WOA:
    """Bazowy Whale Optimization Algorithm dla binarnej selekcji cech.

    Parametry (zgodne z artykułem):
        n_whales    = 20   — liczba osobników w populacji
        max_iter    = 30   — liczba iteracji
        b           = 1.0  — stała kształtu spirali
        alpha       = 0.01 — waga kary za liczbę cech w funkcji fitness
        n_neighbors = 5    — k w KNN
    """

    def __init__(
        self,
        n_whales: int = 20,
        max_iter: int = 30,
        n_neighbors: int = 5,
        alpha: float = 0.01,
        b: float = 1.0,
        max_fitness_samples: int = 5000,
        seed: int = 42,
    ):
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.alpha = alpha          # waga liczby cech w funkcji fitness
        self.b = b                  # stała spirali bańkowej WOA
        self.max_fitness_samples = max_fitness_samples  # podzbiór do oceny fitness
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._fitness_cache: dict = {}

    # -----------------------------------------------------------------------
    # Funkcja transferowa: pozycja ciągła → wektor binarny
    # -----------------------------------------------------------------------
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoidalna funkcja transferowa, stabilna numerycznie."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _to_binary(self, position: np.ndarray) -> np.ndarray:
        """Konwertuje pozycję ciągłą na binarną przez próbkowanie z rozkładu Bernoulliego."""
        probs = self._sigmoid(position)
        return (self.rng.random(position.shape) < probs).astype(np.int8)

    # -----------------------------------------------------------------------
    # Funkcja fitness: error_rate + alpha * feature_ratio
    # -----------------------------------------------------------------------
    def _fitness(
        self,
        binary_pos: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Oblicza wartość funkcji fitness dla danego podzbioru cech.

        fitness = (1 - accuracy) + alpha * (n_selected / n_features)

        Niższa wartość = lepsze rozwiązanie.
        Kara alpha * feature_ratio promuje mniejsze podzbiory cech.
        """
        key = bytes(binary_pos)
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        selected = np.where(binary_pos == 1)[0]
        n_features = len(binary_pos)

        if len(selected) == 0:
            result = 1.0 + self.alpha
        else:
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=-1)
            knn.fit(X_train[:, selected], y_train)
            acc = knn.score(X_val[:, selected], y_val)
            error_rate = 1.0 - acc
            feature_ratio = len(selected) / n_features
            result = error_rate + self.alpha * feature_ratio

        self._fitness_cache[key] = result
        return result

    # -----------------------------------------------------------------------
    # Pomocnicze: przygotowanie podzbiorów do szybkiej oceny fitness
    # -----------------------------------------------------------------------
    def _prepare_fitness_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        """Ogranicza rozmiar zbiorów treningowego/walidacyjnego na czas optymalizacji.

        Używamy podzbiorów żeby każda ocena fitness była szybka (~milliseconds).
        Pełna ewaluacja następuje PO optymalizacji.
        """
        n_train = min(self.max_fitness_samples, len(X_train))
        n_val = min(self.max_fitness_samples // 5, len(X_val))

        idx_train = self.rng.choice(len(X_train), n_train, replace=False)
        idx_val = self.rng.choice(len(X_val), n_val, replace=False)

        return X_train[idx_train], y_train[idx_train], X_val[idx_val], y_val[idx_val]

    # -----------------------------------------------------------------------
    # Główna pętla optymalizacji WOA
    # -----------------------------------------------------------------------
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """Uruchamia WOA i zwraca optymalny podzbiór cech.

        Args:
            X_train, y_train: dane treningowe
            X_val, y_val:     dane walidacyjne (do oceny fitness)
            verbose:          czy wypisywać postęp

        Returns:
            dict z kluczami:
                selected_features: np.ndarray — indeksy wybranych cech
                binary_mask:       np.ndarray — maska binarna (55 wartości 0/1)
                best_fitness:      float      — wartość funkcji fitness
                convergence:       list       — historia najlepszego fitness
                n_selected:        int        — liczba wybranych cech
        """
        self._fitness_cache.clear()

        # Podzbiory danych do oceny fitness (szybsze iteracje)
        Xf_tr, yf_tr, Xf_val, yf_val = self._prepare_fitness_data(
            X_train, y_train, X_val, y_val
        )
        n_features = X_train.shape[1]

        # --- Inicjalizacja losowa ---
        positions = self.rng.uniform(-4, 4, (self.n_whales, n_features))
        binary_positions = np.array([self._to_binary(p) for p in positions])
        fitnesses = np.array([
            self._fitness(b, Xf_tr, yf_tr, Xf_val, yf_val) for b in binary_positions
        ])

        best_idx = int(np.argmin(fitnesses))
        best_pos_cont = positions[best_idx].copy()
        best_binary = binary_positions[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])

        convergence = [best_fitness]

        for t in range(1, self.max_iter + 1):
            # a liniowo maleje od 2 do 0
            a = 2.0 - 2.0 * t / self.max_iter

            for i in range(self.n_whales):
                r1 = self.rng.random()
                r2 = self.rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = self.rng.random()
                l = self.rng.uniform(-1, 1)

                if p < 0.5:
                    if abs(A) < 1:
                        # Faza eksploatacji: okrążanie zdobyczy (Eq. 1-2)
                        D = np.abs(C * best_pos_cont - positions[i])
                        positions[i] = best_pos_cont - A * D
                    else:
                        # Faza eksploracji: losowy osobnik (Eq. 8-9)
                        rand_idx = self.rng.integers(0, self.n_whales)
                        D = np.abs(C * positions[rand_idx] - positions[i])
                        positions[i] = positions[rand_idx] - A * D
                else:
                    # Aktualizacja spiralna (Eq. 5-6)
                    D_star = np.abs(best_pos_cont - positions[i])
                    positions[i] = (
                        D_star * np.exp(self.b * l) * np.cos(2 * np.pi * l)
                        + best_pos_cont
                    )

                # Ogranicz do przestrzeni [-4, 4]
                positions[i] = np.clip(positions[i], -4, 4)

                # Konwersja na binarny
                binary_positions[i] = self._to_binary(positions[i])

                # Ocena fitness i aktualizacja globalnego najlepszego
                fit = self._fitness(binary_positions[i], Xf_tr, yf_tr, Xf_val, yf_val)
                if fit < best_fitness:
                    best_fitness = fit
                    best_pos_cont = positions[i].copy()
                    best_binary = binary_positions[i].copy()

            convergence.append(best_fitness)

            if verbose:
                n_sel = int(best_binary.sum())
                est_acc = 1.0 - (best_fitness - self.alpha * n_sel / n_features)
                print(
                    f"  [{self.__class__.__name__}] "
                    f"iter {t:3d}/{self.max_iter} | "
                    f"fitness={best_fitness:.6f} | "
                    f"features={n_sel:2d}/{n_features} | "
                    f"~acc={est_acc:.4f}"
                )

        return {
            "selected_features": np.where(best_binary == 1)[0].tolist(),
            "binary_mask": best_binary.astype(int),
            "best_fitness": best_fitness,
            "convergence": convergence,
            "n_selected": int(best_binary.sum()),
        }
