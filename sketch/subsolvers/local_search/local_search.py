# coding=utf-8
import itertools
from copy import copy
from typing import Protocol, Iterable, Optional

from models import Instance
from solution import PenaltyFactors, Solution


class Move(Protocol):
    @property
    def delta_cost(self) -> float:
        ...

    @property
    def feasible(self):
        ...

    @property
    def worthwhile(self) -> bool:
        ...

    def apply(self, solution: Solution):
        ...

    def update(self):
        ...


class Operator(Protocol):
    def generate_moves(self, solution: Solution) -> Iterable[Move]:
        ...

    def reset_cache(self):
        ...


class LocalSearchSolver:
    def __init__(self, instance: Instance, initial_penalty: PenaltyFactors, operators: list[Operator]):
        self._instance = instance
        self._penalty = copy(initial_penalty)
        self._operators = operators

    def _generate_moves(self, solution: Solution) -> Iterable[Move]:
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
        self._reset_cache()
        while True:
            if (move := self._find_first_improvement(solution)) is None:
                return
            # Apply
            prev_cost = solution.get_objective(self._penalty)
            expected_cost = prev_cost + move.delta_cost
            # print("Applying move", move.delta_cost, "to solution with cost", prev_cost)
            move.apply(solution)
            move.update()
            yield
            # print(f"New cost: {solution.get_objective(self._penalty)}, prev cost: {prev_cost}, expected: {expected_cost}, delta: {abs(prev_cost - expected_cost)}")
            assert abs(expected_cost - solution.get_objective(self._penalty)) <= 0.01
