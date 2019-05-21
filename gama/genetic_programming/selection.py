import random
import numpy

from gama.utilities.generic.paretofront import ParetoFront


def create_from_population(operator_shell, pop, n, cxpb, mutpb):
    """ Creates n new individuals based on the population. Can apply both crossover and mutation. """
    offspring = []
    for _ in range(n):
        ind1, ind2 = random.sample(pop, k=2)
        if random.random() < cxpb and ind1.can_mate_with(ind2):
            ind1 = operator_shell.mate(ind1, ind2)
        else:
            ind1 = operator_shell.mutate(ind1)
        offspring.append(ind1)
    return offspring


def select_from_pareto(population, select_n, pareto_fronts_n):
    pareto_fronts = []
    population_left = population
    while len(pareto_fronts) != pareto_fronts_n:
        pareto_fronts.append(ParetoFront(start_list=population_left, get_values_fn=lambda ind: ind.fitness.values))
        population_left = [ind for ind in population_left if ind not in pareto_fronts[-1]]
        if len(population_left) == 0 and len(pareto_fronts) != pareto_fronts_n:
            # log.debug("Desired amount of pareto fronts could not be constructed.")
            break

    selected_individuals = []
    for _ in range(select_n):
        if len(pareto_fronts) == 1:
            selected_front = pareto_fronts[0]
        elif len(pareto_fronts) == 2:
            selected_front = numpy.random.choice(pareto_fronts, p=[2/3, 1/3])
        elif len(pareto_fronts) > 2:
            n = len(pareto_fronts)
            selected_front = numpy.random.choice(pareto_fronts, p=[1/2**i for i in range(1, n)] + [1/2**(n-1)])
        else:
            raise RuntimeError("Did not create any pareto front.")

        selected_individuals.append(random.choice(selected_front))
    return selected_individuals


def eliminate_from_pareto(pop, n):
    # For now we only eliminate one at a time so this will do.
    if n != 1:
        raise NotImplemented("Currently only n=1 is supported.")

    def inverse_fitness(ind):
        return [-value for value in ind.fitness.values]

    pareto_worst = ParetoFront(pop, inverse_fitness)
    return [random.choice(pareto_worst)]
