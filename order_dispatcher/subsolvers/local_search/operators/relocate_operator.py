# coding=utf-8
import time
from dataclasses import dataclass

from order_dispatcher.models import Solution, Route
from order_dispatcher.evaluation import Evaluation, RemovalMove, InsertionMove


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


class RelocateOperator:
    def __init__(self, evaluation: Evaluation):
        self._evaluation = evaluation
        self.reset_cache()

    def reset_cache(self):
        self._last_route_evaluation_timestamps = {}

    def _can_skip(self, origin_route: Route, target_route: Route):
        return origin_route.last_modification_time < self._last_route_evaluation_timestamps.get(id(origin_route), -1) \
            and target_route.last_modification_time < self._last_route_evaluation_timestamps.get(id(target_route), -1)

    def generate_moves(self, solution: Solution):
        for origin_route in solution.routes:
            for moved_request in origin_route.requests:
                removal_move = None
                for target_route in solution.routes:
                    if self._can_skip(origin_route, target_route):
                        continue
                    if id(origin_route) == id(target_route):
                        continue
                    if removal_move is None:
                        removal_move = self._evaluation.calculate_removal(moved_request, origin_route)
                    for insertion_pos in range(1, len(target_route.nodes) + 1):
                        for move in self._evaluation.calculate_insertion(moved_request, target_route, insertion_pos):
                            yield RelocateMove(removal_move=removal_move, insertion_move=move)
            self._last_route_evaluation_timestamps[id(origin_route)] = time.time()
