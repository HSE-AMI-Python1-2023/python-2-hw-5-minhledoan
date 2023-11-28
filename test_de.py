import pytest
import numpy as np
from differential_evolution import DifferentialEvolution
import json

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))

BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin

@pytest.fixture
def de_solver():
    return DifferentialEvolution(FOBJ, BOUNDS)


def test_info_coverage_json(de_solver):
    de_solver._init_population()
    filename = 'coverage.json'
    de_solver.iterate()
    de_solver._evaluate(rastrigin(de_solver.best), de_solver.best_idx)
    de_solver._mutation()
    de_solver._crossover()
    trial, trial_denorm = de_solver._recombination(de_solver.best_idx)

    with open(filename, "r") as file:
        data = json.loads(file.read())

    coverage_percentage = data['totals']['percent_covered']
    assert coverage_percentage >= 100, f"Coverage is {coverage_percentage}%, expected at least 100%"

def test_best_solution_found(de_solver):
    de_solver._init_population()
    initial_best = de_solver.best
    for _ in range(100):  # Run for 100 iterations
        de_solver.iterate()

    assert np.all(rastrigin(de_solver.best) <= rastrigin(initial_best))

def test_population_bounds(de_solver):
    de_solver._init_population()
    assert np.all(de_solver.population >= 0) and np.all(de_solver.population <= 1)
