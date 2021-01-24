from gama.genetic_programming.components import Individual
from typing import DefaultDict

from gama.genetic_programming.operations import create_expression_from_decision_vector
from gama.genetic_programming.operator_set import OperatorSet


class PipelineProblem:
    """
    A class denoting the problem of optimizing ML pipelines in GAMA

    dim: maximum number of components in the pipeline
    nobj: the number of fitness objectives
    ops: the used operators
    primitive_set: the used primitives

    USAGE: PipelineProblem()
    """
    def __init__(self, dim, nobj, ops: OperatorSet, primitive_set: DefaultDict):
        self._dim = dim
        if not (nobj == 1 or nobj == 2):
            raise ValueError('Fitness evaluation is not defined with not either nobj == 1 or == 2')
        self._nobj = nobj
        self._ops = ops
        self._primitive_set = primitive_set
        self._evaluatedIndividuals = []

    def fitness(self, x):
        # Create from the decision vector x (i.e. the chromosome) a pipeline
        expression = create_expression_from_decision_vector(decision_vector=x, primitive_set=self._primitive_set, use_non_component=True)
        individual = Individual(main_node=expression, to_pipeline=self._ops.get_compile())

        evaluation = self._ops.evaluate(individual)
        self._evaluatedIndividuals.append(individual)
        if self._nobj == 2:
            return [-evaluation.score[0], len(individual.primitives)]
        elif self._nobj == 1:
            return [-evaluation.score[0]]
        else:
            raise ValueError('Fitness evaluation is not defined with not either nobj == 1 or == 2')

    def get_bounds(self):
        return [0] * self._dim, [1] * self._dim

    def get_extra_info(self):
        return self._evaluatedIndividuals

    def get_nobj(self):
        return self._nobj

    def get_ncx(self):
        return self._dim
