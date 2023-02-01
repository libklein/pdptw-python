# coding=utf-8
from order_dispatcher.models import Request
from order_dispatcher.evaluation.evaluation import Evaluation
from order_dispatcher.models.solution import Solution


class BestInsertionOperator:
    def __init__(self, evaluation: Evaluation):
        self._evaluation = evaluation

    def insert(self, solution: Solution, request: Request):
        return self.repair(solution, {request})

    def repair(self, solution: Solution, missing_requests: set[Request]):
        for next_request in missing_requests:
            moves = (move
                     for route in solution.routes
                     for at in range(1, len(route) + 1)
                     for move in self._evaluation.calculate_insertion(next_request, route, at))
            best_move = min(moves, key=lambda x: x.delta_cost)
            best_move.apply()

        return solution
