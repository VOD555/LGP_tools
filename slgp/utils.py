import random
import numpy as np
from tqdm import tqdm
import csv


def mutation_func(a, b, c, mutation, bounds, method='rand1', best=None):
    """
    Mutation function for differential evolution algorithm.

    Args:
        a (list): First parent.
        b (list): Second parent.
        c (list): Third parent.
        mutation (float): Mutation rate.
        bounds (list): Parameter bounds (tuples of low, high).
        method (str): Mutation method ('rand1' or 'current_to_best').
        best (list): Best solution so far (only for 'current_to_best').

    Returns:
        list: Mutated individual.
    """
    if method == 'rand1' or best is None:
        mutant = [
            a[i] + mutation * (b[i] - c[i])
            for i in range(len(a))
        ]
    elif method == 'current_to_best':
        mutant = [
            a[i] + mutation * (best[i] - a[i]) +
            mutation * (b[i] - c[i])
            for i in range(len(a))
        ]

    mutant = [
        min(max(mutant[i], bounds[i][0]), bounds[i][1])
        for i in range(len(mutant))
    ]

    return mutant


def crossover_func(mutant, parent, crossover):
    """
    Crossover function for differential evolution algorithm.

    Args:
        mutant (list): Mutated individual.
        parent (list): Parent individual.
        crossover (float): Crossover rate.

    Returns:
        list: Trial individual.
    """
    trial = parent
    while np.equal(np.array(trial), np.array(parent)).all():
        trial = [
            mutant[i] if random.random() < crossover else parent[i]
            for i in range(len(mutant))
        ]
    return trial


def compare(func, x1, x2, y2):
    """
    Compare the output of a function for two different inputs.

    Args:
        func (function): Function to be optimized.
        x1 (list): First input.
        x2 (list): Second input.
        y2 (float): Score of the second input.

    Returns:
        tuple: The better input and its score.
    """
    y1 = func(x1)
    if np.isnan(y2):
        y2 = float('inf')
    if np.isnan(y1):
        y1 = float('inf')
    if y1 < y2:
        return x1, y1
    return x2, y2

def differential_evolution(func, bounds, max_iter=100, pop_size=10,
                           mutation=0.5, crossover=0.7, tol=1e-6,
                           save_path=None):
    """
    Differential evolution algorithm with convergence check and CSV logging.

    Args:
        func (function): Function to be optimized.
        bounds (list): Parameter bounds (tuples of low, high).
        max_iter (int): Maximum number of iterations.
        pop_size (int): Population size.
        mutation (float): Mutation rate.
        crossover (float): Crossover rate.
        tol (float): Convergence tolerance.
        save_path (str): If given, path to save CSV with accepted samples.

    Returns:
        tuple: Best parameter set and corresponding function value.
    """
    pop = np.array([
        np.random.uniform(low, high, pop_size)
        for low, high in bounds
    ]).T
    scores = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(scores)
    best = pop[best_idx]
    best_score = scores[best_idx]

    writer = None
    csvfile = None

    if save_path:
        csvfile = open(save_path, 'w', newline='')
        writer = csv.writer(csvfile)
        header = [f"param_{i}" for i in range(len(bounds))] + ["score"]
        writer.writerow(header)

    for i in tqdm(range(max_iter), desc="Optimizing", dynamic_ncols=True):
        for j in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != j]
            a, b, c = pop[random.sample(idxs, 3)]
            mutant = mutation_func(a, b, c, mutation, bounds)
            trial = crossover_func(mutant, pop[j], crossover)

            old_individual = pop[j].copy()
            old_score = scores[j]

            pop[j], scores[j] = compare(func, trial, old_individual, old_score)

            if not np.array_equal(pop[j], old_individual) and writer:
                writer.writerow(pop[j].tolist() + [scores[j]])

        best_idx = np.argmin(scores)
        best = pop[best_idx]
        best_score = scores[best_idx]

        if pop_size > 1:
            second_best = np.partition(scores, 1)[1]
        else:
            second_best = best_score

        if i % max(1, max_iter // 100) == 0:
            diff = abs(best_score - second_best)
            print(f"Iteration {i}: Best score difference = {diff}")
            if diff < tol:
                print(f"Convergence reached at iteration {i}")
                break

    if csvfile:
        csvfile.close()

    return best, best_score