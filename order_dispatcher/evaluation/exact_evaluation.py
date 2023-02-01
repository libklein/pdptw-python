# coding=utf-8
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from .move import InsertionMove, RemovalMove
from order_dispatcher.models import Instance, PenaltyFactors, Request, Route
from order_dispatcher.models.solution import Cost


class ExactEvaluation:
    """
    Evaluates moves by copying the route, modifying the copy according to the requested move, and calculating
    the resulting objective value.
    """
    def __init__(self, instance: Instance, penalty_factors: PenaltyFactors, target_fairness: float):
        self._instance = instance
        self._penalty_factors = penalty_factors
        self._target_fairness = target_fairness

    def compute_cost(self, cost: Cost) -> float:
        return cost * self._penalty_factors

    def calculate_removal(self, of: Request, from_route: Route) -> RemovalMove:
        prev_cost = self.compute_cost(from_route.cost)

        tmp_route = deepcopy(from_route)
        tmp_route.remove(of)

        return RemovalMove(
            delta_cost=self.compute_cost(tmp_route.cost) - prev_cost,
            feasible=tmp_route.feasible,
            request=of, route=from_route)

    def calculate_insertion(self, of: Request, into_route: Route, at: int) -> Iterable[InsertionMove]:
        assert of not in into_route.requests
        prev_cost = self.compute_cost(into_route.cost)
        # Find best insertion spot
        assert at > 0
        for succ_idx in range(at, len(into_route) + 1):
            tmp_route = deepcopy(into_route)
            tmp_route.insert(of, at, succ_idx)

            yield InsertionMove(delta_cost=self.compute_cost(tmp_route.cost) - prev_cost,
                                feasible=tmp_route.feasible,
                                pickup_insertion_point=at,
                                dropoff_insertion_point=succ_idx,
                                request=of,
                                route=into_route)
