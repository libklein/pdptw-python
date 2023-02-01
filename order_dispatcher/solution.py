from __future__ import annotations

import itertools
import time
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Iterable

from order_dispatcher.models import Instance, Vertex, Request, Vehicle


@dataclass(slots=True)
class PenaltyFactors:
    delay_factor: float
    overtime_factor: float
    overload_factor: float
    fairness_factor: float


@dataclass(slots=True, frozen=True)
class Cost:
    travel_time: float
    delay: float
    overtime: float
    overload: int
    fairness_violation: float

    def __post_init__(self):
        assert self.travel_time >= 0 and self.delay >= 0 and self.overtime >= 0 and self.fairness_violation >= 0.0

    def __str__(self):
        return f"{self.travel_time=} {self.delay=} {self.overtime=} {self.overload=} {self.fairness_violation=} ({self.feasible=})"

    @property
    def feasible(self):
        return self.overtime == 0 and self.overload == 0

    @property
    def is_overloaded(self):
        return self.overload > 0

    @property
    def has_overtime(self):
        return self.overtime > 0

    def __mul__(self, other):
        assert isinstance(other, PenaltyFactors)
        return self.travel_time + self.delay * other.delay_factor + self.overtime * other.overtime_factor \
            + self.overload * other.overload_factor + self.fairness_violation * other.fairness_factor

    def __add__(self, other):
        assert isinstance(other, Cost)
        return Cost(self.travel_time + other.travel_time, self.delay + other.delay,
                    self.overtime + other.overtime, self.overload + other.overload,
                    self.fairness_violation + other.fairness_violation)


@dataclass(slots=True, frozen=True)
class Label:
    cum_time: float
    cum_travel_time: float
    cum_load: int

    activity_start_time: float
    earliest_arrival_time: float
    latest_arrival_time: float

    cum_delay: float
    max_load: int

    def get_cost(self, capacity: int, shift_duration: float, fairness_violation: float):
        return Cost(travel_time=self.cum_travel_time, delay=self.cum_delay,
                    overtime=max(0., self.cum_time - shift_duration), overload=max(0, self.max_load - capacity),
                    fairness_violation=fairness_violation)

    @staticmethod
    def FromVertex(vertex: Vertex):
        return Label(cum_time=0., cum_travel_time=0., cum_load=vertex.items, activity_start_time=vertex.tw_start,
                     earliest_arrival_time=vertex.tw_start,
                     latest_arrival_time=vertex.tw_end, cum_delay=0., max_load=0)


def concatenate(prefix: Label, postfix: Label, travel_time: float) -> Label:
    load = prefix.cum_load + postfix.cum_load
    delta = prefix.cum_time - prefix.cum_delay + travel_time
    waiting_time = max(postfix.earliest_arrival_time - delta - prefix.latest_arrival_time, 0.)
    delay = max(prefix.earliest_arrival_time + delta - postfix.latest_arrival_time, 0.)
    return Label(
        cum_time=prefix.cum_time + postfix.cum_time + travel_time + waiting_time,
        cum_travel_time=prefix.cum_travel_time + postfix.cum_travel_time + travel_time,
        cum_load=load,
        activity_start_time=max(prefix.activity_start_time + travel_time, postfix.earliest_arrival_time),
        earliest_arrival_time=max(postfix.earliest_arrival_time - delta, prefix.earliest_arrival_time) - waiting_time,
        latest_arrival_time=min(postfix.latest_arrival_time - delta, prefix.latest_arrival_time) + delay,
        cum_delay=prefix.cum_delay + postfix.cum_delay + delay,
        max_load=max(prefix.max_load, prefix.cum_load + postfix.max_load)
    )


@dataclass(slots=True)
class Node:
    vertex: Vertex
    forward_label: Label
    backward_label: Label

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        node = copy(self)
        node.forward_label = deepcopy(self.forward_label, memodict)
        node.backward_label = deepcopy(self.backward_label, memodict)
        return node

    def __str__(self):
        return str(self.vertex)

    @staticmethod
    def FromVertex(vertex: Vertex):
        return Node(vertex, Label.FromVertex(vertex), Label.FromVertex(vertex))


class Route:
    def __init__(self, instance: Instance, vehicle: Vehicle):
        self._instance = instance
        self._vehicle = vehicle
        self._nodes: list[Node] = [Node.FromVertex(self._vehicle.start)]
        self._requests: list[Request] = []
        self._last_modified_timestamp = time.time()
        # TODO Add route_id, adapt moves and solution accordingly

    @property
    def last_modification_time(self):
        return self._last_modified_timestamp

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        route = copy(self)
        route._nodes = deepcopy(self._nodes, memodict)
        self._requests = copy(self._requests)
        return route

    def _handle_change(self):
        for prev_node, next_node in itertools.pairwise(self._nodes):
            next_node.forward_label = concatenate(prev_node.forward_label, Label.FromVertex(next_node.vertex),
                                                  self._instance.get_travel_time(prev_node.vertex, next_node.vertex))
            assert next_node.forward_label.activity_start_time >= prev_node.vertex.tw_start
            assert next_node.forward_label.activity_start_time >= next_node.vertex.tw_start
        # Backwards node may have changes
        self._nodes[-1].backward_label = Label.FromVertex(self._nodes[-1].vertex)
        # prev_node is before next_node viewed from the first node, i.e., in regular and not reversed order:
        # [first_node, ..., prev_node, next_node, ..., last_node]
        for next_node, prev_node in itertools.pairwise(reversed(self._nodes)):
            # We use the "forward" (prev, next) travel time here. This avoids issues with asymmetric distance matrices. (real world routing)
            prev_node.backward_label = concatenate(Label.FromVertex(prev_node.vertex), next_node.backward_label,
                                                   self._instance.get_travel_time(prev_node.vertex, next_node.vertex))

        self._last_modified_timestamp = time.time()

    def insert(self, request: Request, pickup_at: int, dropoff_at: int):
        assert pickup_at <= dropoff_at
        assert pickup_at > 0
        self._nodes.insert(pickup_at, Node.FromVertex(request.pickup))
        self._nodes.insert(dropoff_at + 1, Node.FromVertex(request.dropoff))
        self._requests.append(request)
        self._handle_change()


    def remove(self, request: Request):
        pickup_pos, dropoff_pos = self.get_idx_of_request(request)
        del self._nodes[dropoff_pos]
        del self._nodes[pickup_pos]
        self._requests.remove(request)

        self._handle_change()

    def append(self, request: Request):
        self.insert(request, len(self._nodes), len(self._nodes))

    def __str__(self):
        return "-".join(map(str, self._nodes)) + " | " + str(self.cost)

    @property
    def travel_time(self) -> float:
        return self.cost.travel_time

    @property
    def start(self) -> Vertex:
        return self._vehicle.start

    @property
    def nodes(self):
        return self._nodes

    @property
    def requests(self):
        return self._requests

    def get_idx_of_vertex(self, vertex: Vertex):
        return next(i for i, x in enumerate(self._nodes) if x.vertex.vertex_id == vertex.vertex_id)

    def get_idx_of_request(self, request: Request):
        pickup_idx = self.get_idx_of_vertex(request.pickup)
        for i in range(pickup_idx + 1, len(self._nodes)):
            if self._nodes[i].vertex.vertex_id == request.dropoff.vertex_id:
                return pickup_idx, i
        raise ValueError

    def get_node_of(self, vertex: Vertex):
        return next(x for x in self._nodes if x.vertex.vertex_id == vertex.vertex_id)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, item: Vertex | Request):
        if isinstance(item, Request):
            return item in self._requests
        else:
            for n in self._nodes:
                if n.vertex.vertex_id == item.vertex_id:
                    return True
        return False

    @property
    def feasible(self):
        return self.cost.feasible

    @property
    def cost(self):
        return self.label.get_cost(self._vehicle.capacity, self._vehicle.end_time,
                                   abs(self._instance.avg_requests_per_driver - len(self.requests)))

    @property
    def label(self):
        return self._nodes[-1].forward_label

    def get_objective(self, factors: PenaltyFactors):
        return self.cost * factors


@dataclass(slots=True, frozen=True)
class InsertionMove:
    delta_cost: float
    feasible: bool
    pickup_insertion_point: int
    dropoff_insertion_point: int
    request: Request
    route: Route

    def __post_init__(self):
        assert self.request not in self.route.requests, f'Found {self.request} in {self.route.requests}'
        assert self.pickup_insertion_point >= 1
        assert self.dropoff_insertion_point >= self.pickup_insertion_point
        assert self.dropoff_insertion_point <= len(self.route.nodes)

    def apply(self, solution: Solution):
        assert any(id(x) == id(self.route) for x in solution.routes)
        self.route.insert(self.request, self.pickup_insertion_point, self.dropoff_insertion_point)


@dataclass(slots=True, frozen=True)
class RemovalMove:
    delta_cost: float
    feasible: bool
    request: Request
    route: Route

    def __post_init__(self):
        assert self.request in self.route.requests

    def apply(self, solution: Solution):
        assert any(id(x) == id(self.route) for x in solution.routes)
        self.route.remove(self.request)


class Evaluation:

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


class ExactEvaluation:

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


class Solution:

    def __init__(self, instance: Instance):
        self._instance = instance
        self.routes: list[Route] = [
            Route(self._instance, vehicle) for vehicle in self._instance.vehicles
        ]

    # TODO Replace with copy?
    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        sol = copy(self)
        sol.routes = deepcopy(self.routes, memodict)
        return sol

    def find_route(self, of: Vertex | Request) -> Route:
        return next(r for r in self.routes if of in r)

    def __str__(self):
        return f'Solution with cost {self.cost} {self.feasible}:\n' + '\n\t'.join(map(str, self.routes))

    @property
    def cost(self) -> Cost:
        return sum((r.cost for r in self.routes), Cost(0., 0., 0., 0, 0.))

    @property
    def feasible(self):
        return all(r.feasible for r in self.routes)

    @property
    def num_requests(self):
        return sum(len(r.requests) for r in self.routes)

    @property
    def requests(self):
        return itertools.chain(*(r.requests for r in self.routes))

    def remove_request(self, request: Request):
        route = self.find_route(request)
        route.remove(request)

    def insert_request(self, request: Request, route: Route, index_of_pickup: int, index_of_dropoff: int):
        route.insert(request, index_of_pickup, index_of_dropoff)

    def get_objective(self, factors: PenaltyFactors) -> float:
        return sum(x.get_objective(factors) for x in self.routes)
