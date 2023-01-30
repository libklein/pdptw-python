from __future__ import annotations
from models import Instance, Vertex, Request, Vehicle
from typing import Iterable
from dataclasses import dataclass
from itertools import pairwise
from copy import copy, deepcopy

@dataclass
class PenaltyFactors:
    delay_factor: float
    overtime_factor: float
    overload_factor: float
    fairness_factor: float

UNWEIGHTED_FACTORS = PenaltyFactors(1.,1.,1.,0.)
@dataclass(frozen=True)
class Cost:
    travel_time: float
    delay: float
    overtime: float
    overload: float
    fairness_violation: float

    def __post_init__(self):
        assert self.travel_time >= 0 and self.delay >= 0 and self.overtime >= 0

    def __str__(self):
        return f"{self.travel_time=} {self.delay=} {self.overtime=} {self.overload=} {self.fairness_violation=} ({self.feasible=})"

    @property
    def feasible(self):
        return self.overtime == 0

    def __mul__(self, other):
        assert isinstance(other, PenaltyFactors)
        return self.travel_time + self.delay*other.delay_factor + self.overtime*other.overtime_factor \
            + self.overload*other.overload_factor + self.fairness_violation*other.fairness_factor

@dataclass
class Label:
    cum_time: float
    cum_travel_time: float
    cum_load: float

    earliest_arrival_time: float
    latest_arrival_time: float

    cum_delay: float
    max_load: float

    def get_cost(self, capacity: float, shift_duration: float):
        return Cost(travel_time=self.cum_travel_time, delay=self.cum_delay, overtime=max(0., self.cum_time - shift_duration), overload=max(0., self.max_load - capacity), fairness_violation=0.)

    @staticmethod
    def FromVertex(vertex: Vertex):
        return Label(cum_time=0., cum_travel_time=0., cum_load=vertex.items, earliest_arrival_time=vertex.tw_start,
                     latest_arrival_time=vertex.tw_end, cum_delay=0., max_load=0.)

def concatenate(prefix: Label, postfix: Label, travel_time: float) -> Label:
    load = prefix.cum_load + postfix.cum_load
    delta = prefix.cum_time - prefix.cum_delay + travel_time
    waiting_time = max(postfix.earliest_arrival_time - delta - prefix.latest_arrival_time, 0.)
    delay = max(prefix.earliest_arrival_time + delta - postfix.latest_arrival_time, 0.)
    return Label(
        cum_time=prefix.cum_time + postfix.cum_time + travel_time + waiting_time,
        cum_travel_time=prefix.cum_travel_time+postfix.cum_travel_time+travel_time,
        cum_load=load,
        earliest_arrival_time=max(postfix.earliest_arrival_time - delta, prefix.earliest_arrival_time) - waiting_time,
        latest_arrival_time=min(postfix.latest_arrival_time - delta, prefix.latest_arrival_time) + delay,
        cum_delay=prefix.cum_delay+postfix.cum_delay+delay,
        max_load=max(prefix.max_load, prefix.cum_load + postfix.max_load)
    )

@dataclass
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

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        route = copy(self)
        route._nodes = deepcopy(self._nodes, memodict)
        self._requests = copy(self._requests)
        return route

    def update(self):
        for node_id, label in zip(range(1, len(self._nodes)), self.calculate_forward_sequence(self._nodes[0], self._nodes[-1])):
            self._nodes[node_id].forward_label = label
        self._nodes[-1].backward_label = Label.FromVertex(self._nodes[-1].vertex)
        for node_id, label in zip(range(len(self._nodes)-2, -1, -1), self.calculate_backward_sequence(self._nodes[-1], self._nodes[0])):
            self._nodes[node_id].backward_label = label

    def insert(self, request: Request, pickup_at: int, dropoff_at: int):
        assert pickup_at <= dropoff_at
        assert pickup_at > 0
        self._nodes.insert(pickup_at, Node.FromVertex(request.pickup))
        self._nodes.insert(dropoff_at+1, Node.FromVertex(request.dropoff))
        self._requests.append(request)

    def remove(self, request: Request):
        pickup_pos, dropoff_pos = self.get_idx_of_request(request)
        del self._nodes[dropoff_pos]
        del self._nodes[pickup_pos]
        self._requests.remove(request)

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
        for i in range(pickup_idx+1, len(self._nodes)):
            if self._nodes[i].vertex.vertex_id == request.dropoff.vertex_id:
                return pickup_idx, i
        raise ValueError

    def get_node_of(self, vertex: Vertex):
        return next(x for x in self._nodes if x.vertex.vertex_id == vertex.vertex_id)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, vertex: Vertex):
        return vertex in self._nodes

    @property
    def feasible(self):
        return self.cost.feasible

    @property
    def cost(self):
        return self.label.get_cost(self._vehicle.capacity, self._vehicle.end_time)

    @property
    def label(self):
        return self._nodes[-1].forward_label

    def calculate_forward_sequence(self, from_node: Node, to_node: Node) -> Iterable[Label]:
        from_pos = self._nodes.index(from_node)
        label = from_node.forward_label
        prev_node = from_node
        for cur_node_pos in range(from_pos+1, len(self._nodes)):
            cur_node = self._nodes[cur_node_pos]
            if prev_node.vertex.vertex_id == to_node.vertex.vertex_id:
                break

            label = concatenate(label,
                                Label.FromVertex(cur_node.vertex),
                                self._instance.get_travel_time(prev_node.vertex, cur_node.vertex))
            yield label
            prev_node = cur_node
        return label

    def calculate_backward_sequence(self, from_node: Node, to_node: Node) -> Iterable[Label]:
        from_pos = self._nodes.index(from_node)
        label = from_node.backward_label
        next_node = from_node
        for cur_node_pos in range(from_pos-1, -1, -1):
            cur_node = self._nodes[cur_node_pos]
            if next_node.vertex.vertex_id == to_node.vertex.vertex_id:
                break

            label = concatenate(Label.FromVertex(cur_node.vertex),
                                label,
                                self._instance.get_travel_time(cur_node.vertex, next_node.vertex))
            yield label
            next_node = cur_node

    def get_objective(self, factors: PenaltyFactors):
        return self.cost*factors


@dataclass
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

    def update(self):
        self.route.update()

@dataclass
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

    def update(self):
        self.route.update()

class Evaluation:

    def __init__(self, instance: Instance, penalty_factors: PenaltyFactors):
        self._instance = instance
        self._penalty_factors = penalty_factors
        assert all(x.capacity == y.capacity for x,y in pairwise(self._instance.vehicles))

    def compute_cost(self, cost: Cost) -> float:
        return cost * self._penalty_factors

    def calculate_removal(self, of: Request, from_route: Route) -> RemovalMove:
        prev_cost = self.compute_cost(from_route.label.get_cost(from_route._vehicle.capacity, from_route._vehicle.end_time))
        pickup_idx, dropoff_idx = from_route.get_idx_of_request(of)
        assert pickup_idx > 0
        prev_node = from_route.nodes[pickup_idx-1]
        label = prev_node.forward_label
        for i in range(pickup_idx+1, dropoff_idx):
            cur_node = from_route.nodes[i]
            label = concatenate(label, Label.FromVertex(cur_node.vertex),
                                self._instance.get_travel_time(prev_node.vertex, cur_node.vertex))
            prev_node = cur_node

        if len(from_route.nodes) > dropoff_idx+1:
            target_node = from_route.nodes[dropoff_idx+1]
            label = concatenate(label, target_node.backward_label,
                                self._instance.get_travel_time(prev_node.vertex, target_node.vertex))

        cost = label.get_cost(from_route._vehicle.capacity, from_route._vehicle.end_time)
        return RemovalMove(delta_cost=self.compute_cost(cost) - prev_cost,
                           feasible=cost.feasible,
                           request=of, route=from_route)

    def calculate_insertion(self, of: Request, into_route: Route, at: int) -> Iterable[InsertionMove]:
        prev_cost = self.compute_cost(into_route.label.get_cost(into_route._vehicle.capacity, into_route._vehicle.end_time))
        # Find best insertion spot
        assert at > 0
        capacity = into_route._vehicle.capacity
        pred_vertex = into_route.nodes[at-1].vertex
        label = concatenate(into_route.nodes[at-1].forward_label, Label.FromVertex(of.pickup),
                            self._instance.get_travel_time(pred_vertex, of.pickup))
        pred_vertex = of.pickup
        for succ_idx in range(at, len(into_route)):
            succ_node = into_route.nodes[succ_idx]
            # Compute cost of inserting here
            inserted_label = concatenate(label, Label.FromVertex(of.dropoff),
                                         self._instance.get_travel_time(pred_vertex, of.dropoff))
            inserted_label = concatenate(inserted_label, succ_node.backward_label,
                                         self._instance.get_travel_time(of.dropoff, succ_node.vertex))
            insertion_cost = inserted_label.get_cost(capacity, into_route._vehicle.end_time)
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

        label = concatenate(label, Label.FromVertex(of.dropoff), self._instance.get_travel_time(pred_vertex, of.dropoff))
        cost = label.get_cost(capacity, into_route._vehicle.end_time)
        yield InsertionMove(delta_cost=self.compute_cost(cost) - prev_cost,
                            feasible=cost.feasible,
                            pickup_insertion_point=at, dropoff_insertion_point=len(into_route),
                            request=of, route=into_route)


class ExactEvaluation:

    def __init__(self, instance: Instance, penalty_factors: PenaltyFactors):
        self._instance = instance
        self._penalty_factors = penalty_factors
        assert all(x.capacity == y.capacity for x,y in pairwise(self._instance.vehicles))

    def compute_cost(self, cost: Cost) -> float:
        return cost * self._penalty_factors

    def calculate_removal(self, of: Request, from_route: Route) -> RemovalMove:
        prev_cost = self.compute_cost(from_route.cost)

        tmp_route = deepcopy(from_route)
        tmp_route.remove(of)
        tmp_route.update()

        return RemovalMove(
            delta_cost=self.compute_cost(tmp_route.cost) - prev_cost,
            feasible=tmp_route.feasible,
            request=of, route=from_route)

    def calculate_insertion(self, of: Request, into_route: Route, at: int) -> Iterable[InsertionMove]:
        assert of not in into_route.requests
        prev_cost = self.compute_cost(into_route.cost)
        # Find best insertion spot
        assert at > 0
        for succ_idx in range(at, len(into_route)+1):
            tmp_route = deepcopy(into_route)
            tmp_route.insert(of, at, succ_idx)
            tmp_route.update()

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
        pass

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        sol = copy(self)
        sol.routes = deepcopy(self.routes, memodict)
        return sol

    def find_route(self, of: Vertex) -> Route:
        return next(r for r in self.routes if of in r)

    def __str__(self):
        return f'Solution with cost {self.get_objective(UNWEIGHTED_FACTORS)} ({self.feasible}):' + '\n\t'.join(map(str, self.routes))

    @property
    def feasible(self):
        return all(r.feasible for r in self.routes)

    def get_objective(self, factors: PenaltyFactors) -> float:
        return sum(x.get_objective(factors) for x in self.routes)
