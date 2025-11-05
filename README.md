# CI2025 Lab 2: Traveling Salesperson Problem (TSP) Solvers

## SOLVING TSP using a made up algorithm(Peak mid BackTrack) + Evolution Strategy approache

This repository contains implementations of heuristic and metaheuristic algorithms for solving the Traveling Salesperson Problem (TSP), including Hill Climbing and Evolution Strategy (ES) approaches.

## Algorithms Overview that im using (made up algorithm):
1-Start from an initial solution and perform standard hill climbing until we reach a local maximum — a point where no neighboring move improves the solution.

2-Instead of restarting from scratch, we backtrack halfway along the path we used to climb to that peak.
For example, we backtrack to the mid point of the path.
This allows us to reuse part of the previous search effort that we think it lead us to a good point instead of throwing it away.

3-From this halfway point, we explore a different branch — meaning we take a different move than before.
We keep a memory (tabu-like list) of moves already tried from that point so we don’t repeat paths we already explored.

4-We continue this process to discover multiple local peaks, storing each peak in a list of candidate solutions.

5-After collecting several peaks, we compare them and select the best one as our current best solution.

6-Once the algorithm has explored all reasonable variations near these peaks, we allow a controlled downhill move or “jump” away from these areas (diversification).
This allows us to explore new regions of the solution space that are far from the current peaks, with the goal of finding a higher global peak.

7-We repeat this process — climb, partial backtrack, branch, save peaks, diversify — until convergence or a stopping condition is reached.

### 1. Local Search Subroutine: Midpoint-Branch 2-opt

A sophisticated local search technique that uses midpoint backtracking to escape local optima.

**Algorithm Steps:**

1. **Initialization**: Start from a tour (Nearest Neighbor or random initialization).

2. **Greedy Hill Climb**:
   - Repeatedly sample up to `NEIGH_LIMIT` 2-opt candidates using the city's candidate lists (k-NN).
   - Apply the first improving move found (first-improvement strategy).
   - Record all moves made during the climb.

3. **Midpoint Branching**:
   - When no sampled improvements remain, record this peak and the list of moves that led to it.
   - Rebuild the midpoint solution by applying the first half of the move sequence.
   - From the midpoint, forbid the original next move.
   - Take a different improving 2-opt move (if one exists) and climb again.

4. **Return**: Return the better tour (shorter length) of the two peaks.

---

### 2. (μ+λ) Self-Adaptive Evolution Strategy

A population-based metaheuristic that evolves a population of TSP tours using self-adaptive mutation rates.

**Algorithm Steps:**

1. **Initialization**:
   - Initialize μ parents (half using Nearest Neighbor, half random).
   - Locally improve each parent with the Midpoint-Branch 2-opt subroutine.
   - Assign initial mutation rates `p_swap` and `p_invert` (Gaussian-drawn, clipped to valid range).

2. **For each generation**:
   
   **Create λ offspring:**
   - Pick a parent at random.
   - **Self-adapt mutation rates**: 
     ```
     p ← p * exp(τ·N(0,1) + τ_local·N(0,1))
     ```
     Clip to [0.02, 0.9].
   - **Mutate the permutation**:
     - Apply inversion with probability `p_invert`.
     - Apply swap with probability `p_swap`.
   - **Improve the child** with the Midpoint-Branch 2-opt subroutine.
   - Set fitness = tour length.

   **Selection (μ+λ)**:
   - Combine parents & offspring.
   - Keep the μ shortest tours (deterministic selection).

3. **Termination**: After `GENERATIONS` iterations, output the best tour & length seen.

---

## Files

- **`hill_climbing.py`**: Multi-start Hill Climbing with 2-opt local search
- **`ES_Peak_BackTrack.py`**: Self-adaptive Evolution Strategy with Midpoint-Branch 2-opt
- **`TSP_Comparison.ipynb`**: Comparison notebook for evaluating both algorithms
