"""
Enhanced Whale Optimization Algorithm (EWOA) — główny algorytm projektu.

Rozszerzenie bazowego WOA o trzy ulepszenia:
  1. OBL (Opposition-Based Learning) — lepsza inicjalizacja populacji
  2. Mutacja bitowa (bit-string mutation) — zwiększa różnorodność
  3. NSS (Neighborhood Search Strategy) — lokalne przeszukiwanie najlepszego

Pipeline:
    EWOA selektuje optymalny podzbiór cech (z 55) → klasyfikator KNN
    ocenia fitness na podstawie dokładności klasyfikacji + kary za liczbę cech.

Referencja:
    "EWOA-Based Feature Selection for Malware Detection Using Memory Analysis"
    IJIES, Vol.17, No.3, 2024.

Parametry (z artykułu):
    n_whales    = 20   — liczba wielorybów (osobników)
    max_iter    = 30   — maksymalna liczba iteracji
    alpha       = 0.01 — waga kary za liczbę cech
    n_neighbors = 5    — k w KNN
"""
import numpy as np
from .woa import WOA


class EWOA(WOA):
    """Enhanced WOA z OBL + mutacją bitową + NSS.

    Dziedziczy podstawową pętlę WOA z woa.py i nadpisuje:
        - inicjalizację (_init_with_obl)
        - pętlę optimize() z mutacją i NSS
    """

    def __init__(
        self,
        n_whales: int = 20,
        max_iter: int = 30,
        n_neighbors: int = 5,
        alpha: float = 0.01,
        b: float = 1.0,
        max_fitness_samples: int = 5000,
        use_nss: bool = True,
        seed: int = 42,
    ):
        super().__init__(
            n_whales=n_whales,
            max_iter=max_iter,
            n_neighbors=n_neighbors,
            alpha=alpha,
            b=b,
            max_fitness_samples=max_fitness_samples,
            seed=seed,
        )
        self.use_nss = use_nss

    # -----------------------------------------------------------------------
    # Ulepszenie 1: OBL — Opposition-Based Learning
    # -----------------------------------------------------------------------
    def _init_with_obl(self, n_features: int) -> np.ndarray:
        """Inicjalizacja z OBL: generuj n_whales pozycji + ich przeciwieństwa.

        Dla przestrzeni [a, b] = [-4, 4]:
            x_opposite = a + b - x = -4 + 4 - x = -x

        Z 2*n_whales kandydatów wybieramy n_whales najlepszych (ocena fitness
        następuje po konwersji do binarnej w optimize()).

        Returns:
            positions: ndarray (2*n_whales, n_features) — startowa populacja
        """
        pop = self.rng.uniform(-4, 4, (self.n_whales, n_features))
        opp = -pop   # a+b=0, więc opozycja = -x
        return np.vstack([pop, opp])

    # -----------------------------------------------------------------------
    # Ulepszenie 2: Mutacja bitowa (część Search Strategy)
    # -----------------------------------------------------------------------
    def _mutation(self, binary_pos: np.ndarray, iteration: int) -> np.ndarray:
        """Bitowa mutacja losowych cech (bit-string mutation).

        Rozmiar mutacji (z artykułu, Eq. 13):
            Eksploracja (iter ≤ max_iter/2): 10%–50% cech
            Eksploatacja (iter > max_iter/2): 1%–9% cech

        Mutacja zwiększa różnorodność i zapobiega ugrzęźnięciu w optimum lokalnym.
        """
        n = len(binary_pos)
        mutated = binary_pos.copy()

        # Dobór rozmiaru mutacji
        if iteration <= self.max_iter / 2:
            mut_pct = self.rng.integers(10, 51)      # 10%–50% eksploracja
        else:
            mut_pct = self.rng.integers(1, 10)       # 1%–9% eksploatacja

        mut_size = max(1, int(n * mut_pct / 100))
        indices = self.rng.choice(n, size=mut_size, replace=False)
        mutated[indices] = 1 - mutated[indices]      # flipuj wybrane bity
        return mutated

    # -----------------------------------------------------------------------
    # Ulepszenie 3: NSS — Neighborhood Search Strategy
    # -----------------------------------------------------------------------
    def _nss(
        self,
        binary_best: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Przeszukiwanie sąsiedztwa najlepszego rozwiązania (ring-based).

        Dla każdej pozycji i w pierścieniu (ostatnia cecha łączy z pierwszą):
            - forward:  flipuj cechę (i+1) % n
            - backward: flipuj cechę (i-1) % n
            - zachowaj najlepszą z trzech opcji

        NSS jest wywoływane TYLKO gdy mutacja poprawi globalne optimum.
        Cel: dogłębne przeszukanie lokalnego otoczenia znalezionego optimum.
        """
        n = len(binary_best)
        best_pos = binary_best.copy()
        best_fit = self._fitness(best_pos, X_train, y_train, X_val, y_val)

        for i in range(n):
            # Forward: przełącz cechę po prawej stronie (ring)
            fwd = best_pos.copy()
            fwd[(i + 1) % n] ^= 1     # XOR flip
            fwd_fit = self._fitness(fwd, X_train, y_train, X_val, y_val)

            # Backward: przełącz cechę po lewej stronie (ring)
            bwd = best_pos.copy()
            bwd[(i - 1) % n] ^= 1
            bwd_fit = self._fitness(bwd, X_train, y_train, X_val, y_val)

            # Zachowaj najlepsze z trzech
            if fwd_fit < best_fit and fwd_fit <= bwd_fit:
                best_pos, best_fit = fwd, fwd_fit
            elif bwd_fit < best_fit:
                best_pos, best_fit = bwd, bwd_fit

        return best_pos, best_fit

    # -----------------------------------------------------------------------
    # Główna pętla EWOA (nadpisuje WOA.optimize)
    # -----------------------------------------------------------------------
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """Uruchamia EWOA i zwraca optymalny podzbiór cech.

        Różnica vs WOA:
          1. Inicjalizacja przez OBL (2× populacja → top N)
          2. Po każdej aktualizacji wieloryba → mutacja bitowa
          3. Jeśli mutacja poprawia optimum → NSS

        Returns:
            dict z kluczami:
                selected_features: list  — indeksy wybranych cech
                binary_mask:       ndarray (n_features,) — maska 0/1
                best_fitness:      float
                convergence:       list — historia fitness per iteracja
                n_selected:        int
        """
        self._fitness_cache.clear()

        # Podzbiory do szybkiej oceny fitness
        Xf_tr, yf_tr, Xf_val, yf_val = self._prepare_fitness_data(
            X_train, y_train, X_val, y_val
        )
        n_features = X_train.shape[1]

        # --- Inicjalizacja z OBL (Ulepszenie 1) ---
        candidates_cont = self._init_with_obl(n_features)  # (2*n_whales, n_feat)
        candidates_bin = np.array([self._to_binary(c) for c in candidates_cont])
        fitnesses = np.array([
            self._fitness(b, Xf_tr, yf_tr, Xf_val, yf_val) for b in candidates_bin
        ])

        # Wybierz top n_whales z 2*n_whales kandydatów
        top_idx = np.argsort(fitnesses)[:self.n_whales]
        positions = candidates_cont[top_idx].copy()
        binary_positions = candidates_bin[top_idx].copy()
        fitnesses = fitnesses[top_idx].copy()

        best_idx = int(np.argmin(fitnesses))
        best_pos_cont = positions[best_idx].copy()
        best_binary = binary_positions[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])

        convergence = [best_fitness]

        for t in range(1, self.max_iter + 1):
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
                        D = np.abs(C * best_pos_cont - positions[i])
                        positions[i] = best_pos_cont - A * D
                    else:
                        rand_idx = self.rng.integers(0, self.n_whales)
                        D = np.abs(C * positions[rand_idx] - positions[i])
                        positions[i] = positions[rand_idx] - A * D
                else:
                    D_star = np.abs(best_pos_cont - positions[i])
                    positions[i] = (
                        D_star * np.exp(self.b * l) * np.cos(2 * np.pi * l)
                        + best_pos_cont
                    )

                positions[i] = np.clip(positions[i], -4, 4)
                binary_positions[i] = self._to_binary(positions[i])

                # --- Ulepszenie 2: Mutacja bitowa ---
                mutated = self._mutation(binary_positions[i], t)
                mut_fit = self._fitness(mutated, Xf_tr, yf_tr, Xf_val, yf_val)
                cur_fit = self._fitness(binary_positions[i], Xf_tr, yf_tr, Xf_val, yf_val)

                if mut_fit < cur_fit:
                    binary_positions[i] = mutated
                    cur_fit = mut_fit

                # Aktualizacja globalnego najlepszego
                if cur_fit < best_fitness:
                    best_fitness = cur_fit
                    best_pos_cont = positions[i].copy()
                    best_binary = binary_positions[i].copy()

                    # --- Ulepszenie 3: NSS gdy optimum się poprawia ---
                    if self.use_nss:
                        nss_pos, nss_fit = self._nss(
                            best_binary, Xf_tr, yf_tr, Xf_val, yf_val
                        )
                        if nss_fit < best_fitness:
                            best_fitness = nss_fit
                            best_binary = nss_pos
                            # Zaktualizuj pozycję ciągłą (aproksymacja)
                            best_pos_cont = positions[i].copy()

            convergence.append(best_fitness)

            if verbose:
                n_sel = int(best_binary.sum())
                est_acc = 1.0 - (best_fitness - self.alpha * n_sel / n_features)
                print(
                    f"  [EWOA] iter {t:3d}/{self.max_iter} | "
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
