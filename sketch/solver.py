import time
from copy import copy
from typing import Protocol

from models import Instance
from subsolvers.large_neighborhood.large_neighborhood import RandomDestroyOperator, BestInsertionOperator
from subsolvers.local_search.operators.relocate_operator import RelocateOperator
from solution import Solution, Evaluation, PenaltyFactors
from subsolvers import LocalSearchSolver, LargeNeighborhood
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
        self._large_neighborhood = LargeNeighborhood(ruin_operators=[RandomDestroyOperator(self._instance, fraction_to_remove=0.2)],
                                                     recreate_operators=[BestInsertionOperator(Evaluation(self._instance, self._obj_factor, self._avg_requests_per_driver))])

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

    def _should_terminate(self):
        return (time.time() - self._solver_start_time) >= 60

    def solve(self):
        best_solution = self._construct_initial_solution()
        yield best_solution
        best_feasible_solution = best_solution if best_solution.feasible else None
        # TODO Termination criterion
        # Use wall clock time
        self._solver_start_time = time.time()
        # TODO this does not actually terminate in 60 secs
        while True:
            # Generate new solution
            next_candidate_solution = next(self._large_neighborhood.explore_neighborhood_of(best_solution))
            # Improve with local search
            for _ in self._local_search_solver.optimize(next_candidate_solution):
                self._
                if (best_obj := best_solution.get_objective(self._obj_factor)) > (cand_obj := next_candidate_solution.get_objective(self._obj_factor)):
                    print(f'Improved solution from {best_obj=} to {cand_obj=}.')
                    best_solution = next_candidate_solution
                    yield best_solution
                if self._should_terminate():
                    return
