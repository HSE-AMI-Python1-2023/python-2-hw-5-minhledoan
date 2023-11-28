import logging 
import datetime



"""
Ваше задание:

1. Используя код из differential_evolution.py, напишите собственный логгер, который будет логгировать каждый запуск и каждый логический этап работы алгоритма
2. Ваш логгер должен сохранять логи с 1, 2, 3 уровнями логгирования в файл logging_de.log
3. Если результат отработки алгоритма больше 1e-3, то логгируем результат с уровнем ERROR. Если результат больше 1e-1, то CRITICAL. 
    Также лог должен в себе отражать параметры алгоритма, такие как начальная популяция, размер популяции, количество итераций и тд.
4. ERROR и CRITICAL должны сохранятся в файл errors.log
5. Напишите свой форматтер, который будет отражать: 
    a. Время логгирования в формате datetime
    б. Имя логгера
    в. Уровень логгирования
    г. Действие, которое было выполнено

ВАЖНО! 
Весь код требуется писать в данном файле, не трогая исходный differential_evolution.py

Удачи!
"""


import numpy as np

class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_time = datetime.datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"{log_time} - {record.name} - {record.levelname} - {record.msg}"
        return log_msg

# Создаем свой логгер
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Создаем обработчики для файлов
file_handler = logging.FileHandler('logging_de.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(CustomFormatter())

errors_file_handler = logging.FileHandler('errors.log')
errors_file_handler.setLevel(logging.ERROR)
errors_file_handler.setFormatter(CustomFormatter())

# Добавляем обработчики к логгеру
logger.addHandler(file_handler)
logger.addHandler(errors_file_handler)
class DifferentialEvolution:
    def __init__(self, fobj, bounds, mutation_coefficient=0.8, crossover_coefficient=0.7, population_size=20):

        self.fobj = fobj
        self.bounds = bounds
        self.mutation_coefficient = mutation_coefficient
        self.crossover_coefficient = crossover_coefficient
        self.population_size = population_size
        self.dimensions = len(self.bounds)

        self.a = None
        self.b = None
        self.c = None
        self.mutant = None
        self.population = None
        self.idxs = None
        self.fitness = []
        self.min_bound = None
        self.max_bound = None
        self.diff = None
        self.population_denorm = None
        self.best_idx = None
        self.best = None
        self.cross_points = None

    def _init_population(self):
        self.population = np.random.rand(self.population_size, self.dimensions)
        self.min_bound, self.max_bound = self.bounds.T

        self.diff = np.fabs(self.min_bound - self.max_bound)
        self.population_denorm = self.min_bound + self.population * self.diff
        self.fitness = np.asarray([self.fobj(ind) for ind in self.population_denorm])

        self.best_idx = np.argmin(self.fitness)
        self.best = self.population_denorm[self.best_idx]
    
    def _mutation(self):
        self.a, self.b, self.c = self.population[np.random.choice(self.idxs, 3, replace = False)]
        self.mutant = np.clip(self.a + self.mutation_coefficient * (self.b - self.c), 0, 1)
        return self.mutant
    
    def _crossover(self):
        cross_points = np.random.rand(self.dimensions) < self.crossover_coefficient
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        return cross_points

    def _recombination(self, population_index):

        trial = np.where(self.cross_points, self.mutant, self.population[population_index])
        trial_denorm = self.min_bound + trial * self.diff
        return trial, trial_denorm
    
    def _evaluate(self, result_of_evolution, population_index):
        if result_of_evolution < self.fitness[population_index]:
                self.fitness[population_index] = result_of_evolution
                self.population[population_index] = self.trial
                if result_of_evolution < self.fitness[self.best_idx]:
                    self.best_idx = population_index
                    self.best = self.trial_denorm

    def iterate(self):
    
        for population_index in range(self.population_size):
            self.idxs = [idx for idx in range(self.population_size) if idx != population_index]

            self.mutant = self._mutation()
            self.cross_points = self._crossover()

            self.trial, self.trial_denorm = self._recombination(population_index)
    
            result_of_evolution = self.fobj(self.trial_denorm)

            self._evaluate(result_of_evolution, population_index)


def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))


if __name__ == "__main__":

    function_obj = rastrigin
    bounds_array = np.array([[-20, 20], [-20, 20]]), np.array([[-10, 50], [-10, 60]]), np.array([[-0, 110], [-42, 32]])
    steps_array = [40, 100, 200]
    mutation_coefficient_array = [0.5, 0.6, 0.3]
    crossover_coefficient_array = [0.5, 0.6, 0.3]
    population_size_array = [20, 30, 40, 50, 60]

    for bounds in bounds_array:
        for steps in steps_array:
            for mutation_coefficient in mutation_coefficient_array:
                for crossover_coefficient in crossover_coefficient_array:
                    for population_size in population_size_array:

                        de_solver = DifferentialEvolution(function_obj, bounds, mutation_coefficient=mutation_coefficient, crossover_coefficient=crossover_coefficient, population_size=population_size)

                        de_solver._init_population()

                        logger.info(f"Bounds: {bounds}, Steps: {steps}, Mutation Coefficient: {mutation_coefficient}, Crossover Coefficient: {crossover_coefficient}, Population Size: {population_size}")

                        for _ in range(steps):
                            de_solver.iterate()

                        if np.any(de_solver.fitness > 1e-3):
                            logger.error(f"Result greater than 1e-3: {de_solver.fitness}")
                        elif np.any(de_solver.fitness > 1e-1):
                            logger.critical(f"Result greater than 1e-1: {de_solver.fitness}")