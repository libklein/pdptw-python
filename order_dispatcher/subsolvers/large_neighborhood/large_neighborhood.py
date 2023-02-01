# coding=utf-8
import random
from copy import copy, deepcopy
from typing import Protocol, Optional, Iterator

from order_dispatcher.models import Instance, Request
from order_dispatcher.solution import Solution, Evaluation


class DestroyOperator(Protocol):
    def destroy(self, solution: Solution) -> set[Request]:
        ...


class RandomDestroyOperator:
    def __init__(self, instance: Instance, fraction_to_remove: float):
        self._instance = instance
        self._fraction_to_remove = fraction_to_remove

    def destroy(self, solution: Solution) -> set[Request]:
        num_requests = int(self._fraction_to_remove * solution.num_requests)
        requests_to_remove = random.sample(list(solution.requests), k=num_requests)
        for next_request in requests_to_remove:
            solution.remove_request(next_request)
        return set(requests_to_remove)


class RepairOperator(Protocol):
    def repair(self, solution: Solution, missing_requests: set[Request]):
        ...


class BestInsertionOperator:
    def __init__(self, evaluation: Evaluation):
        self._evaluation = evaluation

    def repair(self, solution: Solution, missing_requests: set[Request]):
        for next_request in missing_requests:
            moves = (move
                     for route in solution.routes
                     for at in range(1, len(route) + 1)
                     for move in self._evaluation.calculate_insertion(next_request, route, at))
            best_move = min(moves, key=lambda x: x.delta_cost)
            best_move.apply(solution)

        return solution


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
