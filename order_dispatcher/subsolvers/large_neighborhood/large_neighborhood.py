# coding=utf-8
import random
from copy import copy, deepcopy
from typing import Protocol, Optional, Iterator

from order_dispatcher.models import Request
from order_dispatcher.solution import Solution


class DestroyOperator(Protocol):
    def destroy(self, solution: Solution) -> set[Request]:
        ...


class RepairOperator(Protocol):
    def repair(self, solution: Solution, missing_requests: set[Request]):
        ...


class LargeNeighborhood:
    def __init__(self, ruin_operators: list[DestroyOperator], recreate_operators: list[RepairOperator],
                 ruin_operator_weights: Optional[list[float]] = None,
                 recreate_operator_weights: Optional[list[float]] = None):
        # Actually not strictly necessary right now, choices works with None as well. But I'd expect valid weights
        # to be an invariant of the class.
        if ruin_operator_weights is None:
            ruin_operator_weights = [1.0 for _ in ruin_operators]
        if recreate_operator_weights is None:
            recreate_operator_weights = [1.0 for _ in recreate_operators]

        assert len(ruin_operators) == len(ruin_operator_weights)
        assert len(recreate_operators) == len(recreate_operator_weights)

        self._ruin_operators = ruin_operators
        self._recreate_operators = recreate_operators
        self._ruin_operator_weights = copy(ruin_operator_weights)
        self._recreate_operator_weights = copy(recreate_operator_weights)

    def explore_neighborhood_of(self, solution: Solution) -> Iterator[Solution]:
        while True:
            neighborhood_solution = deepcopy(solution)
            # First ruin
            ruin_operator = random.choices(self._ruin_operators, weights=self._ruin_operator_weights)[0]
            removed_requests = ruin_operator.destroy(neighborhood_solution)

            # Then recreate
            recreate_operator = random.choices(self._recreate_operators, weights=self._recreate_operator_weights)[0]
            recreate_operator.repair(neighborhood_solution, removed_requests)

            yield neighborhood_solution
