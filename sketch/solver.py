from copy import copy

from models import Instance
from subsolvers.local_search.operators.relocate_operator import RelocateOperator
from solution import Solution, Evaluation, PenaltyFactors
from subsolvers import LocalSearchSolver
import random

class Solver:
    def __init__(self, instance: Instance):
        self._instance = instance
        # TODO Termination criterion
        # TODO Come up with fairness factor
        self._obj_factor = PenaltyFactors(10.,1.,1.,1000.)
        self._penalty = PenaltyFactors(1.,10000.,10000.,1000.)

        self._avg_requests_per_driver = len(instance.requests)/len(instance.vehicles)

        self._evaluation = Evaluation(self._instance, self._penalty, self._avg_requests_per_driver)

        self._local_search_solver = LocalSearchSolver(self._instance, self._penalty,
                                                      [RelocateOperator(self._evaluation)])

    def _construct_initial_solution(self) -> Solution:
        sol = Solution(self._instance)

        request_set = copy(self._instance.requests)
        random.shuffle(request_set)

        for next_request in request_set:
            moves = (move
                     for route in sol.routes
                     for at in range(1, len(route)+1)
                     for move in self._evaluation.calculate_insertion(next_request, route, at))
            best_move = min(moves, key=lambda x: x.delta_cost)
            best_move.apply(sol)
            best_move.update()

        assert len(set(x for route in sol.routes for x in route.requests)) == len(self._instance.requests)
        return sol

    def solve(self):
        sol = self._construct_initial_solution()
        # Improve with local search
        self._local_search_solver.optimize(sol)
        return sol
