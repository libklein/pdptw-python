# coding=utf-8
import random
from copy import copy, deepcopy
from typing import Protocol, Optional, Iterator

from order_dispatcher.models import Request
from order_dispatcher.models.solution import Solution


class DestroyOperator(Protocol):
    def destroy(self, solution: Solution) -> set[Request]:
        """
        Modifies the solution in-place.
        :param solution: The solution
        :return: The set of removed requests
        """
        ...


class RepairOperator(Protocol):
    def repair(self, solution: Solution, missing_requests: set[Request]):
        """
        Repairs the solution by inserting missing_requests sequentially at their respective best insertion points.
        Modifies the solution in-place.
        :param solution: The solution.
        :param missing_requests: The set of missing requests.
        """
        ...


class LargeNeighborhood:
    def __init__(self, destroy_operators: list[DestroyOperator], repair_operators: list[RepairOperator],
                 destroy_operator_weights: Optional[list[float]] = None,
                 repair_operator_weights: Optional[list[float]] = None):
        """
        :param destroy_operators: A list of destroy operators to define the neighborhood.
        :param repair_operators: A list of repair operators to define the neighborhood.
        :param destroy_operator_weights: Weights of the destroy operators. 1.0 if not provided.
        :param repair_operator_weights: Weights of the repair operators. 1.0 if not provided.
        """
        # Actually not strictly necessary right now, choices works with None as well. But I'd expect valid weights
        # to be an invariant of the class.
        if destroy_operator_weights is None:
            destroy_operator_weights = [1.0 for _ in destroy_operators]
        if repair_operator_weights is None:
            repair_operator_weights = [1.0 for _ in repair_operators]

        assert len(destroy_operators) == len(destroy_operator_weights)
        assert len(repair_operators) == len(repair_operator_weights)

        self._destroy_operators = destroy_operators
        self._repair_operators = repair_operators
        self._destroy_operator_weights = copy(destroy_operator_weights)
        self._repair_operator_weights = copy(repair_operator_weights)

    def explore_neighborhood_of(self, solution: Solution) -> Iterator[Solution]:
        """
        Explores solutions in the (large) neighborhood of solution as defined by this class. Yields each explored solution.
        :param solution: The solution that defines the initial position in the solution space.
        """
        while True:
            neighborhood_solution = deepcopy(solution)
            # First destroy
            destroy_operator = random.choices(self._destroy_operators, weights=self._destroy_operator_weights)[0]
            removed_requests = destroy_operator.destroy(neighborhood_solution)

            # Then repair
            repair_operator = random.choices(self._repair_operators, weights=self._repair_operator_weights)[0]
            repair_operator.repair(neighborhood_solution, removed_requests)

            yield neighborhood_solution
