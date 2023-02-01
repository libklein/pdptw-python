# coding=utf-8
from __future__ import annotations

from copy import copy
from typing import Iterable

from order_dispatcher.evaluation.evaluation import RemovalMove, InsertionMove
from order_dispatcher.models import Instance, PenaltyFactors, Request, Route
from order_dispatcher.models.solution import Cost, concatenate, Label


class ConstantTimeEvaluation:

    def __init__(self, instance: Instance, penalty_factors: PenaltyFactors, target_fairness: float):
        self._instance = instance
        self._penalty_factors = copy(penalty_factors)
        self._target_fairness = target_fairness

    def compute_cost(self, cost: Cost) -> float:
        return cost * self._penalty_factors

    def calculate_removal(self, of: Request, from_route: Route) -> RemovalMove:
        prev_cost = self.compute_cost(from_route.cost)
        pickup_idx, dropoff_idx = from_route.get_idx_of_request(of)
        assert pickup_idx > 0
        fairness_violation = abs((len(from_route.requests) - 1) - self._target_fairness)
        prev_node = from_route.nodes[pickup_idx - 1]
        label = prev_node.forward_label
        for i in range(pickup_idx + 1, dropoff_idx):
            cur_node = from_route.nodes[i]
            label = concatenate(label, Label.FromVertex(cur_node.vertex),
                                self._instance.get_travel_time(prev_node.vertex, cur_node.vertex))
            prev_node = cur_node

        if len(from_route.nodes) > dropoff_idx + 1:
            target_node = from_route.nodes[dropoff_idx + 1]
            label = concatenate(label, target_node.backward_label,
                                self._instance.get_travel_time(prev_node.vertex, target_node.vertex))

        cost = label.get_cost(from_route._vehicle.capacity, from_route._vehicle.end_time, fairness_violation)
        return RemovalMove(delta_cost=self.compute_cost(cost) - prev_cost,
                           feasible=cost.feasible,
                           request=of, route=from_route)

    def calculate_insertion(self, of: Request, into_route: Route, at: int) -> Iterable[InsertionMove]:
        prev_cost = self.compute_cost(into_route.cost)
        # Find best insertion spot
        assert at > 0
        fairness_violation = abs((len(into_route.requests) + 1) - self._target_fairness)
        capacity = into_route._vehicle.capacity
        pred_vertex = into_route.nodes[at - 1].vertex
        label = concatenate(into_route.nodes[at - 1].forward_label, Label.FromVertex(of.pickup),
                            self._instance.get_travel_time(pred_vertex, of.pickup))
        pred_vertex = of.pickup
        for succ_idx in range(at, len(into_route)):
            succ_node = into_route.nodes[succ_idx]
            # Compute cost of inserting here
            inserted_label = concatenate(label, Label.FromVertex(of.dropoff),
                                         self._instance.get_travel_time(pred_vertex, of.dropoff))
            inserted_label = concatenate(inserted_label, succ_node.backward_label,
                                         self._instance.get_travel_time(of.dropoff, succ_node.vertex))
            insertion_cost = inserted_label.get_cost(capacity, into_route._vehicle.end_time, fairness_violation)
            yield InsertionMove(delta_cost=self.compute_cost(insertion_cost) - prev_cost,
                                feasible=insertion_cost.feasible,
                                pickup_insertion_point=at,
                                dropoff_insertion_point=succ_idx,
                                request=of,
                                route=into_route)

            # Compute next
            label = concatenate(label, Label.FromVertex(succ_node.vertex),
                                self._instance.get_travel_time(pred_vertex, succ_node.vertex))
            pred_vertex = succ_node.vertex

        assert pred_vertex == into_route.nodes[-1].vertex or pred_vertex == of.pickup

        label = concatenate(label, Label.FromVertex(of.dropoff),
                            self._instance.get_travel_time(pred_vertex, of.dropoff))
        cost = label.get_cost(capacity, into_route._vehicle.end_time, fairness_violation)
        yield InsertionMove(delta_cost=self.compute_cost(cost) - prev_cost,
                            feasible=cost.feasible,
                            pickup_insertion_point=at, dropoff_insertion_point=len(into_route),
                            request=of, route=into_route)
