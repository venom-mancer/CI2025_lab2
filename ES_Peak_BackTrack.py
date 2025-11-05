# tsp_peak_backtrack_es_adaptive.py
import numpy as np
import random, glob, os
from typing import List, Tuple, Optional
from math import exp

# -------------------- Global ES parameters --------------------
MU = 8
LAMBDA = 24
GENERATIONS = 200
K_CANDIDATES = 40
NEIGH_LIMIT = 20
SEED = 42   

# Strategy parameter learning rates (self-adaptation)
TAU = 0.15       # global learning rate for log-normal mutation
TAU_LOCAL = 0.05 # local rate

# -----------------------------------------------------------------

Move = Tuple[int, int]


# ---------- Utility functions ----------
def build_candidate_lists(D: np.ndarray, k: int = 20) -> List[List[int]]:
    n = D.shape[0]
    order = np.argsort(D, axis=1)
    return [order[i][1:k+1].tolist() for i in range(n)]


def tour_length(D: np.ndarray, tour: List[int]) -> float:
    return float(sum(D[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour))))


def two_opt_delta(D, tour, i, k):
    n = len(tour)
    a, b = tour[i], tour[(i + 1) % n]
    c, d = tour[k], tour[(k + 1) % n]
    return (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])


def apply_2opt_inplace(tour, i, k):
    tour[i:k+1] = tour[i:k+1][::-1]


def build_pos(tour):
    pos = [0] * len(tour)
    for i, c in enumerate(tour):
        pos[c] = i
    return pos


def nearest_neighbor(D, rng):
    n = D.shape[0]
    start = rng.randrange(n)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


# ---------- Local search: greedy climb with midpoint branch ----------
def sample_improving_move(D, tour, pos, cand, rng, forbidden, neigh_limit):
    n = len(tour)
    tried = 0
    for i in rng.sample(range(n - 1), k=min(n - 1, 2 * neigh_limit)):
        a, b = tour[i], tour[(i + 1) % n]
        pool = list(dict.fromkeys(cand[a] + cand[b]))
        rng.shuffle(pool)
        for c in pool:
            k = pos[c]
            if k <= i + 1 or (i == 0 and k == n - 1):
                continue
            mv = (i, k)
            if mv in forbidden:
                continue
            delta = two_opt_delta(D, tour, i, k)
            tried += 1
            if delta < -1e-12:
                return mv, delta
            if tried >= neigh_limit:
                return None, 0.0
    return None, 0.0


def greedy_climb_with_midpoint(D, start_tour, cand, rng):
    tour = start_tour[:]
    pos = build_pos(tour)
    moves = []
    while True:
        mv, delta = sample_improving_move(D, tour, pos, cand, rng, set(), NEIGH_LIMIT)
        if mv is None:
            break
        i, k = mv
        apply_2opt_inplace(tour, i, k)
        moves.append(mv)
    mid_snapshot, mid_next_move = None, None
    if moves:
        mid_idx = len(moves) // 2
        mid = start_tour[:]
        pos_mid = build_pos(mid)
        for idx, (i, k) in enumerate(moves):
            if idx == mid_idx:
                mid_snapshot = mid[:]
                mid_next_move = moves[mid_idx]
            apply_2opt_inplace(mid, i, k)
    return tour, moves, mid_snapshot, mid_next_move


def branch_from_midpoint(D, mid_state, original_next_move, cand, rng):
    if mid_state is None or original_next_move is None:
        return None
    tour = mid_state[:]
    pos = build_pos(tour)
    tabu = {original_next_move}
    mv, delta = sample_improving_move(D, tour, pos, cand, rng, tabu, NEIGH_LIMIT)
    if mv is None:
        return None
    i, k = mv
    apply_2opt_inplace(tour, i, k)
    # continue climbing
    while True:
        mv2, delta2 = sample_improving_move(D, tour, pos, cand, rng, set(), NEIGH_LIMIT)
        if mv2 is None:
            break
        i2, k2 = mv2
        apply_2opt_inplace(tour, i2, k2)
    return tour


def evaluate_with_branching(D, tour, cand, rng):
    peak, moves, mid_state, original_next_move = greedy_climb_with_midpoint(D, tour, cand, rng)
    best_tour, best_len = peak, tour_length(D, peak)
    branched = branch_from_midpoint(D, mid_state, original_next_move, cand, rng)
    if branched is not None:
        blen = tour_length(D, branched)
        if blen < best_len:
            best_tour, best_len = branched, blen
    return best_tour, best_len


# ---------- Self-adaptive Evolution Strategy ----------
def mutate_swap(t, rng):
    i, j = rng.sample(range(len(t)), 2)
    t[i], t[j] = t[j], t[i]


def mutate_invert(t, rng):
    a, b = sorted(rng.sample(range(len(t)), 2))
    t[a:b+1] = t[a:b+1][::-1]


def es_adaptive(D: np.ndarray,
                mu: int = MU,
                lam: int = LAMBDA,
                generations: int = GENERATIONS,
                seed: Optional[int] = SEED):
    """
    (μ+λ) ES with self-adaptive mutation rates on permutations (swap + inversion).
    Each individual carries [tour, p_swap, p_invert].
    """
    rng = random.Random(seed)
    n = D.shape[0]
    cand = build_candidate_lists(D, K_CANDIDATES)

    # --- Initialize population (half NN, half random) ---
    def make_individual(tour):
        # initial mutation rates ~ N(0.15, 0.05)
        return {
            "tour": tour,
            "p_swap": max(0.05, min(0.9, rng.gauss(0.15, 0.05))),
            "p_invert": max(0.05, min(0.9, rng.gauss(0.20, 0.05))),
            "fit": tour_length(D, tour)
        }

    parents = []
    for _ in range(max(1, mu // 2)):
        t = nearest_neighbor(D, rng)
        t, _ = evaluate_with_branching(D, t, cand, rng)
        parents.append(make_individual(t))
    while len(parents) < mu:
        t = list(range(n)); rng.shuffle(t)
        t, _ = evaluate_with_branching(D, t, cand, rng)
        parents.append(make_individual(t))

    def fitness(ind): return ind["fit"]

    best = min(parents, key=fitness)
    bestL = best["fit"]

    for gen in range(generations):

        offspring = []
        for _ in range(lam):
            p = rng.choice(parents)
            child = {
                "tour": p["tour"][:],
                "p_swap": p["p_swap"],
                "p_invert": p["p_invert"],
                "fit": p["fit"]
            }

            # --- self-adapt rates (log-normal) ---
            global_noise = rng.gauss(0, TAU)
            local_noise = rng.gauss(0, TAU_LOCAL)
            child["p_swap"] *= exp(global_noise + local_noise)
            child["p_invert"] *= exp(global_noise + local_noise)
            child["p_swap"] = max(0.02, min(0.9, child["p_swap"]))
            child["p_invert"] = max(0.02, min(0.9, child["p_invert"]))

            # --- apply permutation mutations ---
            if rng.random() < child["p_invert"]:
                mutate_invert(child["tour"], rng)
            if rng.random() < child["p_swap"]:
                mutate_swap(child["tour"], rng)

            # --- local search evaluation (branching hill climb) ---
            child["tour"], child["fit"] = evaluate_with_branching(D, child["tour"], cand, rng)
            offspring.append(child)

        # --- (μ+λ) deterministic selection ---
        combined = parents + offspring
        combined.sort(key=fitness)
        parents = combined[:mu]

        # track best
        if parents[0]["fit"] < bestL:
            best = parents[0]
            bestL = best["fit"]

    print(f"Final best length: {bestL:.2f}")
    return best["tour"], bestL


if __name__ == "__main__":
    files = sorted(glob.glob("Problems/problem_*.npy")) + ["Problems/test_problem.npy"]
    if not any(os.path.exists(f) for f in files):
        files = sorted(glob.glob("problem_*.npy")) + ["test_problem.npy"]

    for f in files:
        if os.path.exists(f):
            print(f"\n{os.path.basename(f)}:")
            try:
                D = np.load(f)
                best_tour, best_len = es_adaptive(D)
            except Exception as e:
                print(f"Error solving {os.path.basename(f)}: {e}")
