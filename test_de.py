import pytest 
import numpy as np

from differential_evolution import DifferentialEvolution

# CONSTANTS

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))

BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin


"""
Ваша задача добиться 100% покрытия тестами DifferentialEvolution
Различные этапы тестирования логики разделяйте на различные функции
Запуск команды тестирования:
pytest -s test_de.py --cov-report=json --cov
"""
def test_initialization():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    assert de_solver.fobj == FOBJ
    assert np.array_equal(de_solver.bounds, BOUNDS)
    assert de_solver.mutation_coefficient == 0.8
    assert de_solver.crossover_coefficient == 0.7
    assert de_solver.population_size == 20
    # Add more assertions based on your initialization logic

def test_init_population():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    de_solver._init_population()
    assert de_solver.population.shape == (20, 2)
    assert np.all((de_solver.population >= 0) & (de_solver.population <= 1))
    # Add more assertions based on your initialization logic

def test_mutation():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    de_solver._init_population()
    de_solver.idxs = [idx for idx in range(de_solver.population_size) if idx != 0]
    mutant = de_solver._mutation()
    assert mutant.shape == (2,)
    assert np.all((mutant >= 0) & (mutant <= 1))
    # Add more assertions based on your mutation logic

# Update the test_recombination function
def test_recombination():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    de_solver._init_population()
    de_solver._mutation()
    de_solver._crossover()
    trial, trial_denorm = de_solver._recombination(0)
    assert trial.shape == (2,)
    assert np.all((trial >= 0) & (trial <= 1))
    # Add more assertions based on your recombination logic

# Update the test_evaluate function
def test_evaluate():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    de_solver._init_population()
    de_solver._mutation()
    de_solver._crossover()
    de_solver._recombination(0)
    result_of_evolution = FOBJ(de_solver.trial_denorm)
    de_solver._evaluate(result_of_evolution, 0)
    assert result_of_evolution == de_solver.fitness[0]
    # Add more assertions based on your evaluation logic

def test_iterate():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    de_solver._init_population()
    de_solver.iterate()
    # Add assertions based on the expected behavior of the iterate method

# Add more test cases as needed
def test_recombination():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    de_solver._init_population()
    de_solver._mutation()
    de_solver._crossover()
    trial, trial_denorm = de_solver._recombination(0)
    assert trial.shape == (2,)
    assert np.all((trial >= 0) & (trial <= 1))
    # Add more assertions based on your recombination logic

# Update the test_evaluate function
def test_evaluate():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    de_solver._init_population()
    de_solver._mutation()
    de_solver._crossover()
    de_solver._recombination(0)
    result_of_evolution = FOBJ(de_solver.trial_denorm)
    de_solver._evaluate(result_of_evolution, 0)
    assert result_of_evolution == de_solver.fitness[0]
if __name__ == "__main__":
    pytest.main()