import random
from math import floor
from typing import List

from gama.genetic_programming.components import (
    Primitive,
    Terminal,
    PrimitiveNode,
    DATA_TERMINAL,
)


def random_terminals_for_primitive(
    primitive_set: dict, primitive: Primitive
) -> List[Terminal]:
    """ Return a list with a random Terminal for each required input to Primitive. """
    return [random.choice(primitive_set[term_type]) for term_type in primitive.input]


def random_primitive_node(
    output_type: str, primitive_set: dict, exclude: Primitive = None
) -> PrimitiveNode:
    """ Create a PrimitiveNode with specified output_type and random terminals. """
    primitive = random.choice([p for p in primitive_set[output_type] if p != exclude])
    terminals = random_terminals_for_primitive(primitive_set, primitive)
    return PrimitiveNode(primitive, data_node=DATA_TERMINAL, terminals=terminals)


def create_random_expression(
    primitive_set: dict, min_length: int = 1, max_length: int = 3
) -> PrimitiveNode:
    """ Create at least min_length and at most max_length chained PrimitiveNodes. """
    individual_length = random.randint(min_length, max_length)
    learner_node = random_primitive_node(
        output_type="prediction", primitive_set=primitive_set
    )
    last_primitive_node = learner_node
    for _ in range(individual_length - 1):
        primitive_node = random_primitive_node(
            output_type=DATA_TERMINAL, primitive_set=primitive_set
        )
        last_primitive_node._data_node = primitive_node
        last_primitive_node = primitive_node

    return learner_node

def primitive_node_from_decision_value(
    output_type: str, decision_value: float, primitive_set: dict, use_non_component: bool = False, exclude: Primitive = None
) -> PrimitiveNode:
    """ Creates a primitive node from a decision value (float; between 0 and 1) of a decision vector/genome.
        This depends on the type of node: prediction or DATA_TERMINAL, used by PyGMO methods """
    # Get the maximum number of available components, from either
    max_components = sum([1 for p in primitive_set[output_type] if p != exclude])
    if use_non_component:
        max_components += 1  # Add one possibility for the non-component

    index_from_decision = max(min(floor(decision_value * max_components), max_components - 1), 0)  # Find out value + enforce bounds

    # Possibly return nothing, if it's a non-component
    if use_non_component:
        index_from_decision -= 1
        if index_from_decision == -1:
            return None  # Don't place a component here, and thus return None

    # Place a specific component here, with random terminals
    primitive = primitive_set[output_type][index_from_decision]
    terminals = random_terminals_for_primitive(primitive_set, primitive)
    return PrimitiveNode(primitive, data_node=DATA_TERMINAL, terminals=terminals)

def create_expression_from_decision_vector(
    decision_vector: List[float], primitive_set: dict, use_non_component: bool = False, exclude: Primitive = None
) -> PrimitiveNode:
    """ Creates an expression from a decision vector, used by PyGMO methods """
    # Create the learner node
    learner_node = primitive_node_from_decision_value(
        output_type="prediction", decision_value=decision_vector[len(decision_vector) - 1], primitive_set=primitive_set,
        use_non_component=False
    )
    last_primitive_node = learner_node

    # Create and add the remaining nodes: data nodes
    for i in reversed(range(len(decision_vector) - 1)):
        primitive_node = primitive_node_from_decision_value(output_type=DATA_TERMINAL,
                                                            decision_value=decision_vector[i],
                                                            primitive_set=primitive_set,
                                                            use_non_component=use_non_component)
        # Only add a node if it's not a non-component
        if primitive_node is not None:
            last_primitive_node._data_node = primitive_node
            last_primitive_node = primitive_node

    # Return the end result
    return learner_node
