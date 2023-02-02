import random
import time
from copy import copy, deepcopy

from order_dispatcher.models import Instance, Duration
from order_dispatcher.evaluation import ConstantTimeEvaluation
from order_dispatcher.models.solution import Solution, PenaltyFactors
from order_dispatcher.subsolvers import LocalSearchSolver, LargeNeighborhood
from order_dispatcher.operators.best_insertion_operator import BestInsertionOperator
from order_dispatcher.operators.random_destroy_operator import RandomDestroyOperator
from order_dispatcher.subsolvers.local_search.operators.relocate_operator import RelocateOperator


class Solver:
    def __init__(self, instance: Instance, objective_function_factors: PenaltyFactors, time_limit_sec: Duration):
        """
        :param instance: The instance to solve.
        :param objective_function_factors: The factors of the objective function
        :param time_limit_sec: The time limit (wall clock time) in seconds
        """
        self._instance = instance
        self._obj_factor = objective_function_factors
        self._time_limit = time_limit_sec
        # The penalty used in the local search and by operators
        self._penalty = PenaltyFactors(delay_factor=self._obj_factor.delay_factor, overtime_factor=10000.,
                                       overload_factor=10000., fairness_factor=self._obj_factor.fairness_factor)

        self._evaluation = ConstantTimeEvaluation(self._instance, penalty_factors=self._penalty,
                                                  target_fairness=self._instance.avg_requests_per_driver)

        self._local_search_solver = LocalSearchSolver(self._instance, self._penalty,
                                                      [RelocateOperator(self._evaluation)])
        self._large_neighborhood = LargeNeighborhood(
            destroy_operators=[RandomDestroyOperator(fraction_to_remove=0.2)],
            repair_operators=[
                BestInsertionOperator(ConstantTimeEvaluation(self._instance, penalty_factors=self._obj_factor,
                                                             target_fairness=self._instance.avg_requests_per_driver))])

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
            best_insertion_operator.insert(sol, request=next_request)

        return sol

    def _should_terminate(self):
        return (time.time() - self._solver_start_time) >= self._time_limit

    def _update_penalty(self, solution: Solution):
        """
        Updates the factors of the generalized cost function based on the feasibility of solution
        """
        if solution.cost.has_overtime:
            self._penalty.overtime_factor *= 1.2
        else:
            self._penalty.overtime_factor *= 0.8
        if solution.cost.is_overloaded:
            self._penalty.overload_factor *= 1.2
        else:
            self._penalty.overload_factor *= 0.8
        # Propagate update to local search solver.
        self._local_search_solver.notify_penalty_updated()


    def _generate_solutions(self):
        """
        Generates solutions using LNS. Yields each explored solution.
        """
        while True:
            # Generate new solution
            next_candidate_solution = next(self._large_neighborhood.explore_neighborhood_of(self._best_solution))
            # Returns a copy as we will modify next_canidate_solution further
            yield deepcopy(next_candidate_solution)
            # Improve with local search
            for _ in self._local_search_solver.optimize(next_candidate_solution):
                yield deepcopy(next_candidate_solution)

            self._update_penalty(next_candidate_solution)

    def solve(self):
        """
        Solves the instance, reporting each improving solution. Terminates when exceeding the time limit.
        """
        self._solver_start_time = time.time()
        self._best_solution = self._construct_initial_solution()
        self._best_feasible_solution = self._best_solution if self._best_solution.feasible else None
        # Use wall clock time
        for candidate_solution in self._generate_solutions():
            candidate_objective = candidate_solution.get_objective(self._obj_factor)
            if self._best_solution.get_objective(self._obj_factor) > candidate_objective:
                self._best_solution = candidate_solution
                yield self._best_solution
            if candidate_solution.feasible and (
                    self._best_feasible_solution is None or self._best_feasible_solution.get_objective(
                    self._obj_factor) > candidate_objective):
                self._best_feasible_solution = candidate_solution
                yield self._best_feasible_solution

            if self._should_terminate():
                return
