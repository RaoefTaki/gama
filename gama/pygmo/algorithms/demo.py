import random
from typing import DefaultDict

import numpy as np
import random as rd
import copy

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operations import create_expression_from_decision_vector
from gama.genetic_programming.operator_set import OperatorSet
import pygmo as pg


class DEMO:
    """
    DEMO Algorithm: Differential Evolution for Multiobjective Optimization.
    """

    def __init__(self, nobj: int = 2, ops: OperatorSet = None, primitive_set: DefaultDict = None,
                 use_non_component: bool = True, f: float = 0.8):
        """
        Constructs a DEMO algorithm

        USAGE: DEMO()

        nobj: the number of dimensions in this multioptimization problem
        ops: the used operators
        primitive_set: the used primitives
        use_non_component: whether or not to use non-components (no primitive at a certain index)
        f: the scaling factor
        """
        # Define the 'private' data members
        self._nobj = nobj
        self._ops = ops
        self._primitive_set = primitive_set
        self._use_non_component = use_non_component
        self._f = f

    def evolve(self, pop):
        # If the population is empty (i.e. no individuals) nothing happens
        if len(pop) == 0:
            return pop

        # Only iterate over the original population's size, so not over newly created, and added, candidate vectors
        original_pop_size = len(pop)
        pop_copy = copy.deepcopy(pop)
        for i in range(original_pop_size):
            # Get the fitness value of the parent individual
            parent_decision_vector = pop.get_x()[i]
            parent_fitness = pop.get_f()[i] * -1  # -1, to make it easily comparable to the candidate's calculated fitness

            # Mutation: Create candidate
            # Randomly select 3 individuals != parent, so first create list of possible indices, and sample from those
            sample_indices = list(range(len(pop)))  # This includes newly created, and added, candidate vectors
            sample_indices.pop(i)
            random_individuals = rd.sample(sample_indices, 3)

            candidate_decision_vector = pop.get_x()[random_individuals[0]] + self._f * \
                                        np.subtract(pop.get_x()[random_individuals[1]],
                                                    pop.get_x()[random_individuals[2]])

            # Perform crossover
            candidate_decision_vector = pg.sbx_crossover(parent1=candidate_decision_vector,
                                                         parent2=parent_decision_vector,
                                                         bounds=[[0]*len(candidate_decision_vector), [1]*len(candidate_decision_vector)],
                                                         nix=0, p_cr=1, eta_c=10, seed=random.randrange(1000000))[0]
            # Fix nan values and out of bound values
            for j in range(len(candidate_decision_vector)):
                if np.isnan(candidate_decision_vector[j]) or candidate_decision_vector[j] < 0:
                    candidate_decision_vector[j] = 0
                elif candidate_decision_vector[j] > 0:
                    candidate_decision_vector[j] = 1

            # Evaluate the fitness of the candidate
            candidate_expression = create_expression_from_decision_vector(decision_vector=candidate_decision_vector,
                                                                          primitive_set=self._primitive_set,
                                                                          use_non_component=self._use_non_component)
            candidate_individual = Individual(main_node=candidate_expression, to_pipeline=self._ops.get_compile())
            candidate_fitness = self._ops.evaluate(candidate_individual).score

            # Check what individual dominates, the parent or the candidate. Pareto dominance: an individual is
            # better or equal in all fitness aspects, and strictly better in one aspect
            if self.dominates(candidate_fitness, parent_fitness):
                # Replace the parent by the candidate. It calculates the fitness automatically
                pop.set_x(i, candidate_decision_vector)
            elif self.dominates(parent_fitness, candidate_fitness):
                # Discard the candidate, i.e. don't add it to the population
                pass
            else:
                # If no-one dominates, add the candidate to the back of the population
                pop.push_back(candidate_decision_vector)

        # If the population size is too large, truncate it
        if len(pop) > original_pop_size:
            # Sort the individuals with non-dominated sorting
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(pop.get_f())

            # Repeatedly add the non-dominated fronts to the new population, until there is no space left for a full
            # front to be added
            new_population = []
            j = 0
            while len(new_population) + len(ndf[j]) <= original_pop_size:
                for id_value in ndf[j]:  # Add each id in non-dominated front j to the population
                    new_population.append(id_value)
                j += 1
            # Now we add the best id values from the remaining front, ndf[j], to the new population, based on the
            # crowding distance
            if len(new_population) < original_pop_size:
                ndf_fitness_values = [pop.get_f()[id_value] for id_value in ndf[j]]
                # Add more id values
                crowding_distances = pg.crowding_distance(ndf_fitness_values)
                indices_from_sorted = sorted(range(len(crowding_distances)), key=lambda k: crowding_distances[k])
                for index in indices_from_sorted:
                    new_population.append(ndf[j][index])

                    if len(new_population) >= original_pop_size:
                        break

            # Turn IDs into values, and remove others from pop
            k = 0
            for id_value in new_population:
                pop_copy.set_x(k, pop.get_x()[id_value])
                k += 1
            pop = pop_copy
        # Return the evolved population
        return pop

    def dominates(self, ind1_fitness, ind2_fitness):
        """
        Determines whether ind1, with fitness=ind1_fitness, dominates ind2, with fitness=ind2_fitness
        These fitness values are multi-dimensional (multiple aspects, e.g. performance, pipeline length, etc)
        """
        ind1_better_all_aspects = all(ind1_fitness[i] >= ind2_fitness[i] for i in range(self._nobj))
        ind1_strictly_better_one_aspect = any(ind1_fitness[i] > ind2_fitness[i] for i in range(self._nobj))
        return ind1_better_all_aspects and ind1_strictly_better_one_aspect
