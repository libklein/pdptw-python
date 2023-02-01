import random
import time
from copy import copy, deepcopy

from order_dispatcher.models import Instance, Duration
from order_dispatcher.solution import Solution, Evaluation, PenaltyFactors
from order_dispatcher.subsolvers import LocalSearchSolver, LargeNeighborhood
from order_dispatcher.subsolvers.large_neighborhood.large_neighborhood import RandomDestroyOperator, BestInsertionOperator
from order_dispatcher.subsolvers.local_search.operators.relocate_operator import RelocateOperator


class Solver:
    def __init__(self, instance: Instance, objective_function_factors: PenaltyFactors, time_limit_sec: Duration):
        self._instance = instance
        self._obj_factor = objective_function_factors
        self._time_limit = time_limit_sec
        self._penalty = PenaltyFactors(delay_factor=self._obj_factor.delay_factor, overtime_factor=10000.,
                                       overload_factor=10000., fairness_factor=self._obj_factor.fairness_factor)

        self._evaluation = Evaluation(self._instance, self._penalty, self._instance.avg_requests_per_driver)

        self._local_search_solver = LocalSearchSolver(self._instance, self._penalty,
                                                      [RelocateOperator(self._evaluation)])
        self._large_neighborhood = LargeNeighborhood(
            ruin_operators=[RandomDestroyOperator(self._instance, fraction_to_remove=0.2)],
            recreate_operators=[
                BestInsertionOperator(Evaluation(self._instance, self._obj_factor, self._instance.avg_requests_per_driver))])

        self._best_solution = None
        self._best_feasible_solution = None

    def _construct_initial_solution(self) -> Solution:
        """
        Constructs a solution by inserting requests at the best possible location into an initially empty solution in random order.
        """
        best_insertion_operator = BestInsertionOperator(self._evaluation)
        sol = Solution(self._instance)

        request_set = copy(self._instance._requests)
        # Randomize the order in which requests are considered for insertion.
        random.shuffle(request_set)

        # Insert requests one at a time.
        for next_request in request_set:
            best_insertion_operator.repair(sol, missing_requests={next_request})

        return sol

    def _should_terminate(self):
        return (time.time() - self._solver_start_time) >= self._time_limit

    def _generate_solutions(self):
        i = 0
        while True:
            print(f"Iteration {i}")
            i += 1
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
            if self._best_solution.get_objective(self._obj_factor) > candidate_objective:
                self._best_solution = deepcopy(candidate_solution)
                yield self._best_solution
            if candidate_solution.feasible and (
                    self._best_feasible_solution is None or self._best_feasible_solution.get_objective(
                    self._obj_factor) > candidate_objective):
                self._best_feasible_solution = deepcopy(candidate_solution)
                yield self._best_feasible_solution

            if self._should_terminate():
                return
