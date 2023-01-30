from copy import copy
from dataclasses import dataclass

from models import Instance, Vehicle, Request, Vertex
from solution import Solution, Route, Evaluation, InsertionMove, RemovalMove, ExactEvaluation, PenaltyFactors
import random

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

class Solver:
    def __init__(self, instance: Instance):
        self._instance = instance
        self._obj_factor = PenaltyFactors(1.,1.,1.,100000.)
        self._penalty = copy(self._obj_factor)

        self._avg_requests_per_driver = len(instance.requests)/len(instance.vehicles)

        self._evaluation = Evaluation(self._instance, self._penalty, self._avg_requests_per_driver)

    def _try_relocate_request(self, solution: Solution):
        for origin_route in solution.routes:
            for moved_request in origin_route.requests:
                removal_move = self._evaluation.calculate_removal(moved_request, origin_route)
                for target_route in solution.routes:
                    if id(origin_route) == id(target_route):
                        continue
                    for insertion_pos in range(1, len(target_route.nodes)+1):
                        for move in self._evaluation.calculate_insertion(moved_request, target_route, insertion_pos):
                            yield RelocateMove(removal_move=removal_move, insertion_move=move)
        # TODO Intra route relocate?


    def _local_search(self, solution: Solution):
        while True:
            moves = self._try_relocate_request(solution)
            best_move = min(moves, key=lambda move: move.delta_cost)
            if best_move.delta_cost >= 0:
                return
            # Apply
            prev_cost = solution.get_objective(self._penalty)
            expected_cost = prev_cost + best_move.delta_cost
            print("Applying move", best_move.delta_cost, "to solution with cost", prev_cost)
            best_move.apply(solution)
            best_move.update()

            req_set = set()
            for r in solution.routes:
                for req in r.requests:
                    if req in req_set:
                        raise ValueError
                    req_set.add(req)

            print(f"New cost: {solution.get_objective(self._penalty)}, prev cost: {prev_cost}, expected: {expected_cost}, delta: {abs(prev_cost-expected_cost)}")

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
        print(sol)
        # Improve with local search
        self._local_search(sol)
        print(sol)
        return sol
