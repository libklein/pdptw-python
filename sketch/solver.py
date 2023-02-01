import time
from copy import copy, deepcopy
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

        self._best_solution = None
        self._best_feasible_solution = None

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

    def _generate_solutions(self):
        while True:
            # Generate new solution
            next_candidate_solution = next(self._large_neighborhood.explore_neighborhood_of(self._best_solution))
            # Improve with local search
            for _ in self._local_search_solver.optimize(next_candidate_solution):
                yield next_candidate_solution

    def solve(self):
        self._solver_start_time = time.time()
        self._best_solution = self._construct_initial_solution()
        self._best_feasible_solution = self._best_solution if self._best_solution.feasible else None
        # Use wall clock time
        for candidate_solution in self._generate_solutions():
            candidate_objective = candidate_solution.get_objective(self._obj_factor)
            if (best_obj := self._best_solution.get_objective(self._obj_factor)) > candidate_objective:
                print(f'Improved solution from {best_obj=} to {candidate_objective=}.')
                self._best_solution = deepcopy(candidate_solution)
                yield self._best_solution
            if candidate_solution.feasible and (self._best_feasible_solution is None or self._best_feasible_solution.get_objective(self._obj_factor) > candidate_objective):
                self._best_feasible_solution = deepcopy(candidate_solution)
                yield self._best_feasible_solution
                print(f'Found improving feasible solution: {self._best_feasible_solution.get_objective(self._obj_factor)}')

            if self._should_terminate():
                return
