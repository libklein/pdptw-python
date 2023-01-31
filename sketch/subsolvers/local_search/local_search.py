# coding=utf-8
import itertools
from copy import copy
from dataclasses import dataclass
from typing import Protocol, Iterable

from models import Instance
from solution import PenaltyFactors, Solution, Evaluation, RemovalMove, InsertionMove


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

class LocalSearchSolver:
    def __init__(self, instance: Instance, initial_penalty: PenaltyFactors, operators: list[Operator]):
        self._instance = instance
        self._penalty = copy(initial_penalty)
        self._operators = operators

    def _generate_moves(self, solution: Solution) -> Iterable[Move]:
        return itertools.chain(*(op.generate_moves(solution) for op in self._operators))

    def optimize(self, solution: Solution):
        while True:
            moves = self._generate_moves(solution)
            best_move = min(moves, key=lambda move: move.delta_cost)
            if best_move.delta_cost >= 0:
                return
            # Apply
            prev_cost = solution.get_objective(self._penalty)
            expected_cost = prev_cost + best_move.delta_cost
            print("Applying move", best_move.delta_cost, "to solution with cost", prev_cost)
            best_move.apply(solution)
            best_move.update()

            print(
                f"New cost: {solution.get_objective(self._penalty)}, prev cost: {prev_cost}, expected: {expected_cost}, delta: {abs(prev_cost - expected_cost)}")
