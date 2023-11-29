import pytest
import numpy as np
from differential_evolution import DifferentialEvolution

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))

BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin

@pytest.fixture
def de_solver():
    return DifferentialEvolution(FOBJ, BOUNDS)

def test_initialization(de_solver):
    assert de_solver.population_size > 0
    assert de_solver.dimensions == len(BOUNDS)

def test_init_population(de_solver):
    de_solver._init_population()
    assert de_solver.population.shape == (de_solver.population_size, de_solver.dimensions)
    assert np.all((de_solver.population >= 0) & (de_solver.population <= 1))

def test_mutation(de_solver):
    de_solver._init_population()
    for idx in range(de_solver.population_size):
        de_solver.idxs = [i for i in range(de_solver.population_size) if i != idx]
        mutant = de_solver._mutation()
        assert len(mutant) == de_solver.dimensions
        assert np.all((mutant >= 0) & (mutant <= 1))
        # Add more mutation tests as needed

def test_crossover(de_solver):
    de_solver._init_population()
    de_solver.idxs = list(range(de_solver.population_size))
    de_solver._mutation()
    cross_points = de_solver._crossover()
    assert len(cross_points) == de_solver.dimensions
    assert np.all(np.logical_or(cross_points, ~cross_points))

def test_recombination_and_evaluation(de_solver):
    de_solver._init_population()
    for idx in range(de_solver.population_size):
        de_solver.idxs = [i for i in range(de_solver.population_size) if i != idx]
        de_solver._mutation()
        de_solver._crossover()
        trial, trial_denorm = de_solver._recombination(idx)
        assert len(trial) == de_solver.dimensions
        assert np.all((trial >= 0) & (trial <= 1))
        de_solver._evaluate(rastrigin(trial_denorm), idx)

def test_iteration(de_solver):
    de_solver._init_population()
    initial_best = de_solver.best
    de_solver.iterate()
    assert de_solver.best is not None
    assert np.any(de_solver.best != initial_best)

