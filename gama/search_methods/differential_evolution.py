import logging
from typing import List, Union

import pandas as pd
import pygmo as pg
import numpy as np

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.pygmo.problems.pipeline_problem import PipelineProblem
from gama.search_methods.base_search import BaseSearch

log = logging.getLogger(__name__)

class DifferentialEvolution(BaseSearch):
    def __init__(self, primitive_set, dim, population_size):
        super().__init__()
        self._primitive_set = primitive_set
        self._dim = dim
        self._population_size = population_size
        self._nr_of_objectives = 1

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        differential_evolution(self, ops=operations, primitive_set=self._primitive_set, dim=self._dim,
                               nobj=self._nr_of_objectives, population_size=self._population_size)

    def dynamic_defaults(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], time_limit: float,
                         x_test: np.ndarray = None, y_test: np.ndarray = None) -> None:
        pass

    def get_is_multi_objective(self):
        # Returns that this is not a multiobjective optimization method, but rather a singleobjective one
        return self._nr_of_objectives >= 2

def differential_evolution(
    de: DifferentialEvolution,
    ops: OperatorSet,
    primitive_set,
    dim: int = 5,
    nobj: int = 1,
    population_size: int = 20,
):
    """
    Perform differential evolution with given operators.

    de: the DifferentialEvolution object from which this function was called
    ops: the used operators
    primitive_set: the set of primitives used to construct pipelines from
    dim: the number of dimensions in this problem (i.e. the number of components in the pipeline)
    nobj: the number of optimization objectives
    population_size: the maximum size of the population of individuals

    NOTE: TODO: This method does not incorporate using start candidates yet, and does not incorporate a maximum
          TODO: number of iterations
          TODO: Exclusion of primitives has also not been added yet

    This function modifies the BaseSearch's 'output' variable
    """

    # Create the problem
    pipeline_problem = PipelineProblem(dim, nobj, ops, primitive_set)
    prob = pg.problem(pipeline_problem)

    # Instantiate the algorithm
    algo = pg.algorithm(pg.de())

    # Instantiate the population
    pop = pg.population(prob, population_size)

    # Allow for (asynchronous) multiprocessing
    isl = pg.island(algo=algo, pop=pop, udi=pg.mp_island())

    # Run generations until the time has expired
    while True:
        # Evolve one generation
        isl.evolve()

        # Wait for the result, and add set the output
        isl.wait()
        de.output = isl.get_population().problem.extract(PipelineProblem).get_extra_info()
