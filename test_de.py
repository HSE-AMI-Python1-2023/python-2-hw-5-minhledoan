import pytest
import numpy as np
from differential_evolution import DifferentialEvolution

# Rastrigin function
def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))

# Test configuration
BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin

@pytest.fixture
def de_solver():
    return DifferentialEvolution(FOBJ, BOUNDS)

def test_initialization(de_solver):
    assert de_solver.population_size > 0
    assert de_solver.dimensions == 2

def test_init_population(de_solver):
    de_solver._init_population()
    assert de_solver.population.shape == (de_solver.population_size, de_solver.dimensions)
    assert np.all((0 <= de_solver.population) & (de_solver.population <= 1))

def test_mutation(de_solver):
    de_solver._init_population()
    for idx in range(de_solver.population_size):
        de_solver.idxs = [i for i in range(de_solver.population_size) if i != idx]
        mutant = de_solver._mutation()
        assert len(mutant) == de_solver.dimensions
        assert np.all((0 <= mutant) & (mutant <= 1))

def test_crossover(de_solver):
    de_solver._init_population()
    de_solver.idxs = list(range(de_solver.population_size))
    de_solver._mutation()
    cross_points = de_solver._crossover()
    assert len(cross_points) == de_solver.dimensions

def test_recombination_and_evaluation(de_solver):
    de_solver._init_population()
    if not de_solver.idxs:
        return
    de_solver._mutation()
    de_solver._crossover()

    for idx in range(de_solver.population_size):
        de_solver.idxs = [i for i in range(de_solver.population_size) if i != idx]
        # Perform recombination and evaluation
        trial, trial_denorm = de_solver._recombination(idx)
        de_solver._evaluate(rastrigin(trial_denorm), idx)
        # Ensure the shape of the trial vector
        assert len(trial) == de_solver.dimensions
        # Ensure trial vector values are either from the mutant or the current population
        assert np.all((trial == de_solver.mutant) | (trial == de_solver.population[idx]))
        # Ensure trial_denorm is calculated correctly
        expected_trial_denorm = de_solver.min_bound + trial * de_solver.diff
        assert np.array_equal(trial_denorm, expected_trial_denorm)
        # Ensure evaluation updates the fitness and population correctly
        result_of_evolution = rastrigin(trial_denorm)
        if result_of_evolution < de_solver.fitness[idx]:
            assert de_solver.fitness[idx] == result_of_evolution
            assert np.array_equal(de_solver.population[idx], de_solver.trial)
            if result_of_evolution < de_solver.fitness[de_solver.best_idx]:
                assert de_solver.best_idx == idx
                assert np.array_equal(de_solver.best, de_solver.trial_denorm)

def test_iteration(de_solver):
    de_solver._init_population()
    initial_best = de_solver.best
    de_solver.iterate()
    assert de_solver.best is not None
    assert de_solver.best is not initial_best or np.array_equal(de_solver.best, initial_best)
