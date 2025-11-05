# tsp_hc_fast.py
import numpy as np
import random
from typing import List, Tuple, Optional

class TSPHillClimberFast:
    """
    Fast Hill Climber for TSP using first-improvement 2-opt with candidate lists.
    - O(1) delta evaluation per move
    - Only a small number of neighbors per step
    - Optional epsilon-greedy seeding
    """

    def __init__(self, D: np.ndarray, seed: Optional[int] = 42, k_candidates: int = 30, neigh_limit: int = 10):
        D = np.asarray(D, dtype=float)
        assert D.shape[0] == D.shape[1], "Distance matrix must be square"
        self.D = D
        self.n = D.shape[0]
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.k = int(k_candidates)
        self.neigh_limit = int(neigh_limit)
        self.cand = self._build_candidate_lists(self.D, self.k)

    # ---------- utils ----------
    def tour_length(self, tour: List[int]) -> float:
        D, n = self.D, self.n
        return float(sum(D[tour[i], tour[(i + 1) % n]] for i in range(n)))

    def _build_pos(self, tour: List[int]) -> List[int]:
        pos = [0] * self.n
        for i, c in enumerate(tour):
            pos[c] = i
        return pos

    @staticmethod
    def _apply_2opt_inplace(tour: List[int], i: int, k: int, pos: Optional[List[int]] = None) -> None:
        tour[i:k+1] = tour[i:k+1][::-1]
        if pos is not None:
            for t in range(i, k + 1):
                pos[tour[t]] = t

    def _two_opt_delta(self, tour: List[int], i: int, k: int) -> float:
        """Change in length if we reverse tour[i:k+1]."""
        n = self.n
        a = tour[i]
        b = tour[(i + 1) % n]
        c = tour[k]
        d = tour[(k + 1) % n]
        return (self.D[a, c] + self.D[b, d]) - (self.D[a, b] + self.D[c, d])

    @staticmethod
    def _build_candidate_lists(D: np.ndarray, k: int) -> List[List[int]]:
        order = np.argsort(D, axis=1)
        # skip self at [i, i] (first is self if D has zeros on diagonal)
        return [order[i][1:k+1].tolist() for i in range(D.shape[0])]

    # ---------- seeding ----------
    def _nearest_neighbor(self, start: Optional[int] = None) -> List[int]:
        n, D, rng = self.n, self.D, self.rng
        if start is None:
            start = rng.randrange(n)
        unvisited = set(range(n))
        tour = [start]
        unvisited.remove(start)
        cur = start
        for _ in range(n - 1):
            nxt = min(unvisited, key=lambda j: D[cur, j])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    def _eps_greedy_seed(self, eps: float = 0.1) -> List[int]:
        """Your idea kept: NN with occasional random choices to diversify."""
        n, D, rng = self.n, self.D, self.rng
        unvisited = set(range(n))
        cur = rng.randrange(n)
        tour = [cur]
        unvisited.remove(cur)
        for _ in range(n - 1):
            row = list(unvisited)
            if rng.random() < eps:
                nxt = rng.choice(row)
            else:
                nxt = min(row, key=lambda j: D[cur, j])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    # ---------- core HC: first-improvement 2-opt with capped neighbors ----------
    def _first_improvement_step(self, tour: List[int], pos: List[int]) -> bool:
        """
        Try up to self.neigh_limit candidate 2-opt moves using k-NN lists.
        Return True if an improving move was applied.
        """
        n, rng = self.n, self.rng
        tried = 0
        # iterate i in random order, limit the number of i's we even look at
        for i in rng.sample(range(n - 1), k=min(n - 1, 2 * self.neigh_limit)):
            a, b = tour[i], tour[(i + 1) % n]
            # candidate pool from endpoints a and b
            pool = list(dict.fromkeys(self.cand[a] + self.cand[b]))
            rng.shuffle(pool)
            for c in pool:
                k = pos[c]
                # avoid adjacent/wrap pairs
                if k <= i + 1 or (i == 0 and k == n - 1):
                    continue
                delta = self._two_opt_delta(tour, i, k)
                tried += 1
                if delta < -1e-12:
                    self._apply_2opt_inplace(tour, i, k, pos=pos)
                    return True
                if tried >= self.neigh_limit:
                    return False
        return False

    def local_search(self, tour: List[int]) -> List[int]:
        pos = self._build_pos(tour)
        while self._first_improvement_step(tour, pos):
            pass
        return tour

    # ---------- public API ----------
    def solve(self, restarts: int = 30, seed_mode: str = "mixed", verbose: bool = False) -> Tuple[List[int], float]:
        """
        Run multiple restarts and return best (tour, length).
        seed_mode: "random" | "nn" | "eps" | "mixed"
        """
        best_tour, best_len = None, float("inf")
        for r in range(restarts):
            if seed_mode == "random":
                tour = list(range(self.n)); self.rng.shuffle(tour)
            elif seed_mode == "nn":
                tour = self._nearest_neighbor()
            elif seed_mode == "eps":
                tour = self._eps_greedy_seed(eps=0.1)
            else:  # "mixed"
                tour = self._nearest_neighbor() if (r % 2) else self._eps_greedy_seed(0.1)

            tour = self.local_search(tour)
            L = self.tour_length(tour)
            if L < best_len:
                best_tour, best_len = tour[:], L
        return best_tour, best_len


# ---------------------- script usage ----------------------

def solve_file(path: str, **kwargs):
    D = np.load(path)
    solver = TSPHillClimberFast(D, **kwargs)
    best_tour, best_len = solver.solve()
    print(f"{os.path.basename(path)}: best length = {best_len:.2f}")
    return best_tour, best_len


if __name__ == "__main__":
    import glob
    import os
    # Prefer Problems/ if present; otherwise local dir.
    files = sorted(glob.glob("Problems/problem_*.npy")) + ["Problems/test_problem.npy"]
    if not any(os.path.exists(f) for f in files):
        files = sorted(glob.glob("problem_*.npy")) + ["test_problem.npy"]

    for f in files:
        if os.path.exists(f):
            try:
                solve_file(f, seed=42, k_candidates=30, neigh_limit=10)
            except Exception as e:
                print(f"Error on {os.path.basename(f)}: {e}")
