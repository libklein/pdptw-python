# coding=utf-8
import itertools
from typing import Protocol, Iterable, Optional

from order_dispatcher.evaluation import Move
from order_dispatcher.models import Instance
from order_dispatcher.models.solution import PenaltyFactors, Solution


class Operator(Protocol):
    """
    A local search operator
    """

    def generate_moves(self, solution: Solution) -> Iterable[Move]:
        """
        Generates moves that could be applied to the given solution.
        """
        ...

    def reset_cache(self):
        ...


class LocalSearchSolver:
    def __init__(self, instance: Instance, penalty: PenaltyFactors, operators: list[Operator]):
        self._instance = instance
        self._penalty = penalty
        self._operators = operators

    def notify_penalty_updated(self):
        self._reset_cache()

    def _generate_moves(self, solution: Solution) -> Iterable[Move]:
        # You can switch to feasible moves only by filtering the move stream
        return itertools.chain(*(op.generate_moves(solution) for op in self._operators))

    def _reset_cache(self):
        for op in self._operators:
            op.reset_cache()

    def _find_first_improvement(self, solution: Solution) -> Optional[Move]:
        for move in self._generate_moves(solution):
            if not move.worthwhile:
                continue
            return move
        return None

    def optimize(self, solution: Solution):
        """
        Generates solutions found by exploring the neighborhood of solution as defined by this class. Modifies the
        solution in-place. Applies the first worthwhile move.
        :param solution: The solution the search originates from
        """
        self._reset_cache()
        solution.shuffle_route_order()
        while True:
            if (move := self._find_first_improvement(solution)) is None:
                return
            # Apply
            prev_cost = solution.get_objective(self._penalty)
            expected_cost = prev_cost + move.delta_cost
            move.apply()
            assert abs(expected_cost - solution.get_objective(self._penalty)) <= 0.01
            yield solution
            assert abs(expected_cost - solution.get_objective(self._penalty)) <= 0.01
