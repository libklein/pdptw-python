# coding=utf-8
from dataclasses import dataclass

from solution import RemovalMove, InsertionMove, Evaluation, Solution


@dataclass
class RelocateMove:
    removal_move: RemovalMove
    insertion_move: InsertionMove

    @property
    def delta_cost(self) -> float:
        return self.removal_move.delta_cost + self.insertion_move.delta_cost

    @property
    def feasible(self):
        return self.removal_move.feasible and self.insertion_move.feasible

    @property
    def worthwhile(self) -> bool:
        return self.delta_cost < 0

    def apply(self, solution: Solution):
        self.removal_move.apply(solution)
        self.insertion_move.apply(solution)

    def update(self):
        self.removal_move.update()
        self.insertion_move.update()


class RelocateOperator:
    def __init__(self, evaluation: Evaluation):
        self._evaluation = evaluation

    def generate_moves(self, solution: Solution):
        for origin_route in solution.routes:
            for moved_request in origin_route.requests:
                removal_move = self._evaluation.calculate_removal(moved_request, origin_route)
                for target_route in solution.routes:
                    if id(origin_route) == id(target_route):
                        continue
                    for insertion_pos in range(1, len(target_route.nodes) + 1):
                        for move in self._evaluation.calculate_insertion(moved_request, target_route, insertion_pos):
                            yield RelocateMove(removal_move=removal_move, insertion_move=move)
