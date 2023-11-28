import pytest
import numpy as np

from differential_evolution import DifferentialEvolution

BOUNDS_1 = np.array([[-20, 20], [-20, 20]])
BOUNDS_2 = np.array([[-10, 50], [-10, 60]])
BOUNDS_3 = np.array([[0, 110], [-42, 32]])

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))

FOBJ = rastrigin


@pytest.fixture
def de_solver_1():
    return DifferentialEvolution(FOBJ, BOUNDS_1)

@pytest.fixture
def de_solver_2():
    return DifferentialEvolution(FOBJ, BOUNDS_2)

@pytest.fixture
def de_solver_3():
    return DifferentialEvolution(FOBJ, BOUNDS_3)

def test_initialization(de_solver_1):
    assert de_solver_1.population_size > 0
    assert de_solver_1.dimensions == 2

def test_init_population(de_solver_1):
    de_solver_1._init_population()
    assert de_solver_1.population.shape == (de_solver_1.population_size, de_solver_1.dimensions)
    assert np.all(de_solver_1.population >= 0) and np.all(de_solver_1.population <= 1)

def test_mutation(de_solver_1):
    de_solver_1._init_population()
    for idx in range(de_solver_1.population_size):
        de_solver_1.idxs = [i for i in range(de_solver_1.population_size) if i != idx]
        mutant = de_solver_1._mutation()
        assert len(mutant) == de_solver_1.dimensions

def test_crossover(de_solver_1):
    de_solver_1._init_population()
    de_solver_1.idxs = list(range(de_solver_1.population_size))
    de_solver_1._mutation()
    cross_points = de_solver_1._crossover()
    assert len(cross_points) == de_solver_1.dimensions

def test_recombination_and_evaluation(de_solver_1):
    de_solver_1._init_population()
    for idx in range(de_solver_1.population_size):
        de_solver_1.idxs = [i for i in range(de_solver_1.population_size) if i != idx]
        de_solver_1._mutation()
        de_solver_1._crossover()
        trial, trial_denorm = de_solver_1._recombination(idx)
        assert len(trial) == de_solver_1.dimensions
        de_solver_1._evaluate(rastrigin(trial_denorm), idx)

def test_iteration(de_solver_1):
    de_solver_1._init_population()
    initial_best = de_solver_1.best
    de_solver_1.iterate()
    assert de_solver_1.best is not None
    assert de_solver_1.best is not initial_best or np.array_equal(de_solver_1.best, initial_best)

