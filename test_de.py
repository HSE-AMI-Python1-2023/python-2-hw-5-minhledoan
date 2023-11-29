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

def test_initial(de_solver):
    assert de_solver.population_size > 0
    assert de_solver.dimensions == 2

def test_init_population(de_solver):
    de_solver._init_population()
    assert de_solver.population.shape == (de_solver.population_size, de_solver.dimensions)
    assert np.all((0 <= de_solver.population) & (de_solver.population <= 1))


def test_mutation(de_solver):
    de_solver._init_population()
    for idx in range(de_solver.population_size):
        assert np.shape(de_solver._mutation()) == (de_solver.dimensions,)


def test_crossover(de_solver):
    de_solver._init_population()
    de_solver.idxs = list(range(de_solver.population_size))
    de_solver._mutation()
    cross_points = de_solver._crossover()
    assert cross_points.shape == (de_solver.dimensions,)


def test_recombination_and_evaluation(de_solver, FOBJ):
    de_solver._init_population()
    for idx in range(de_solver.population_size):
        de_solver.idxs = [i for i in range(de_solver.population_size) if i != idx]
        de_solver._mutation()
        de_solver._crossover()
        trial, trial_denorm = de_solver._recombination(idx)
        assert np.shape(trial) == (de_solver.dimensions,)
        result_of_evolution = FOBJ(trial_denorm)
        de_solver._evaluate(result_of_evolution, idx)


def test_iteration(de_solver):
    de_solver._init_population()
    initial_best_value = FOBJ(de_solver.best)

    de_solver.iterate()
    assert de_solver.best is not None

    new_best_value = FOBJ(de_solver.best)
    assert new_best_value < initial_best_value, "New best solution does not improve the objective function value."

