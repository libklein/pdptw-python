from __future__ import annotations
from models import Instance, Vertex, Request, Vehicle
from typing import Iterable
from dataclasses import dataclass
from itertools import pairwise, islice
from copy import copy, deepcopy


@dataclass
class Label:
    distance: float
    time: float
    cum_load: int

    # Penalties
    delay: float
    max_overload: int

    def propagate_forward(self, to: Vertex, travel_time: float, capacity: int) -> 'Label':
        new_time = max(self.time + travel_time, to.tw_start)
        load = self.cum_load + to.items

        new_label = Label(distance=self.distance + travel_time,
                          time=min(new_time, to.tw_end),
                          cum_load=load,
                          delay=self.delay+max(0, new_time - to.tw_end),
                          max_overload=max(self.max_overload, load - capacity))
        return new_label

    def propagate_backward(self, to: Vertex, travel_time: float, capacity: int) -> 'Label':
        new_time = min(self.time - travel_time, to.tw_end)
        load = self.cum_load - to.items

        new_label = Label(distance=self.distance + travel_time,
                          time=max(new_time, to.tw_start),
                          cum_load=load,
                          delay=self.delay+max(0, to.tw_start - new_time),
                          max_overload=max(self.max_overload, load - capacity))
        return new_label

    def __str__(self):
        return f'[{self.distance=} {self.time=} {self.cum_load=} {self.delay=} {self.max_overload=} | {self.feasible}]'

    @property
    def feasible(self) -> bool:
        return self.delay == 0 and self.max_overload == 0

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
    def FromVertex(vertex: Vertex, time=0.):
        return Node(vertex, Label(
                        distance=0., 
                        time=time,
                        cum_load=vertex.items,
                        delay=0.,
                        max_overload=0), Label(distance=0., time=time,
                                               cum_load=-vertex.items,
                                               delay=0., max_overload=0))

    @staticmethod
    def concatenate(vertex: Vertex, forward_label: Label, backward_label: Label) -> Label:
        load = forward_label.cum_load + backward_label.cum_load - vertex.items
        return Label(distance=forward_label.distance + backward_label.distance,
                     time=forward_label.time,
                     cum_load=load,
                     delay=max(0., forward_label.time - backward_label.time),
                     max_overload=max(forward_label.max_overload, backward_label.max_overload))

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
        return route

    def update(self):
        for node_id, label in zip(range(1, len(self._nodes)), self.calculate_forward_sequence(self._nodes[0], self._nodes[-1])):
            self._nodes[node_id].forward_label = label
        for node_id, label in zip(range(len(self._nodes)-2, -1, -1), self.calculate_backward_sequence(self._nodes[-1], self._nodes[0])):
            self._nodes[node_id].backward_label = label

    def insert(self, request: Request, pickup_at: int, dropoff_at: int):
        assert pickup_at <= dropoff_at
        assert pickup_at > 0
        self._nodes.insert(pickup_at, Node.FromVertex(request.pickup))
        self._nodes.insert(dropoff_at+1, Node.FromVertex(request.dropoff, time=self._vehicle.end_time))
        self._requests.append(request)

    def remove(self, request: Request):
        pickup_pos, dropoff_pos = self.get_idx_of_request(request)
        del self._nodes[dropoff_pos]
        del self._nodes[pickup_pos]
        self._requests.remove(request)

    def append(self, request: Request):
        self.insert(request, len(self._nodes), len(self._nodes))

    def __str__(self):
        return "-".join(map(str, self._nodes))

    @property
    def distance(self) -> float:
        return self.label.distance

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
        return self.label.feasible

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

            label = label.propagate_forward(to=cur_node.vertex,
                                            travel_time=self._instance.get_travel_time(prev_node.vertex, cur_node.vertex), 
                                            capacity=self._vehicle.capacity)
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

            label = label.propagate_backward(to=cur_node.vertex,
                                             travel_time=self._instance.get_travel_time(cur_node.vertex, next_node.vertex), 
                                             capacity=self._vehicle.capacity)
            yield label
            next_node = cur_node

@dataclass
class InsertionMove:
    delta_cost: float
    feasible: bool
    pickup_insertion_point: int
    dropoff_insertion_point: int
    request: Request
    route: Route

    def __post_init__(self):
        assert self.request not in self.route.requests
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

    def __init__(self, instance: Instance):
        self._instance = instance
        assert all(x.capacity == y.capacity for x,y in pairwise(self._instance.vehicles))

    def compute_cost(self, label: Label) -> float:
        # TODO
        return label.distance + 1000. * label.max_overload + 100. * label.delay

    def calculate_removal(self, of: Request, from_route: Route) -> RemovalMove:
        prev_cost = self.compute_cost(from_route.label)
        pickup_idx, dropoff_idx = from_route.get_idx_of_request(of)
        assert pickup_idx > 0
        prev_node = from_route.nodes[pickup_idx-1]
        label = prev_node.forward_label
        for i in range(pickup_idx+1, dropoff_idx):
            cur_node = from_route.nodes[i]
            label = label.propagate_forward(cur_node.vertex, self._instance.get_travel_time(prev_node.vertex, cur_node.vertex), from_route._vehicle.capacity)
            prev_node = cur_node

        if len(from_route.nodes) > dropoff_idx+1:
            target_node = from_route.nodes[dropoff_idx+1]
            label = label.propagate_forward(target_node.vertex,
                                            self._instance.get_travel_time(prev_node.vertex, target_node.vertex),
                                            capacity=from_route._vehicle.capacity)
        else:
            assert dropoff_idx == len(from_route.nodes)-1
            target_node = from_route.nodes[-1]

        label = Node.concatenate(target_node.vertex, label, target_node.backward_label)
        return RemovalMove(delta_cost=self.compute_cost(label) - prev_cost, feasible=label.feasible, request=of, route=from_route)

    def calculate_insertion(self, of: Request, into_route: Route, at: int) -> Iterable[InsertionMove]:
        prev_cost = self.compute_cost(into_route.label)
        # Find best insertion spot
        assert at > 0
        capacity = into_route._vehicle.capacity
        pred_vertex = into_route.nodes[at-1].vertex
        label = into_route.nodes[at-1].forward_label.propagate_forward(of.pickup,
                                                                       self._instance.get_travel_time(
                                                                           pred_vertex, of.pickup),
                                                          capacity=capacity)
        pred_vertex = of.pickup
        for succ_idx in range(at, len(into_route)):
            succ_node = into_route.nodes[succ_idx]
            # Compute cost of inserting here
            inserted_label_fwd = label.propagate_forward(of.dropoff, self._instance.get_travel_time(pred_vertex, of.dropoff), capacity)
            inserted_label_bwd = succ_node.backward_label.propagate_backward(of.dropoff, self._instance.get_travel_time(of.dropoff, succ_node.vertex), capacity)
            inserted_label = Node.concatenate(of.dropoff, inserted_label_fwd, inserted_label_bwd)
            yield InsertionMove(delta_cost=self.compute_cost(inserted_label) - prev_cost,
                                feasible=inserted_label.feasible,
                                pickup_insertion_point=at,
                                dropoff_insertion_point=succ_idx,
                                request=of,
                                route=into_route)

            # Compute next
            label = label.propagate_forward(succ_node.vertex, self._instance.get_travel_time(pred_vertex, succ_node.vertex), capacity)
            pred_vertex = succ_node.vertex

        assert pred_vertex == into_route.nodes[-1].vertex or pred_vertex == of.pickup

        label = label.propagate_forward(of.dropoff, self._instance.get_travel_time(pred_vertex, of.dropoff), capacity)

        yield InsertionMove(delta_cost=self.compute_cost(label) - prev_cost, feasible=label.feasible,
                            pickup_insertion_point=at, dropoff_insertion_point=len(into_route),
                            request=of, route=into_route)

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
        return f'Solution with cost {self.cost} ({self.feasible}):' + '\n\t'.join(map(str, self.routes))

    @property
    def feasible(self):
        return all(r.feasible for r in self.routes)

    @property
    def cost(self) -> float:
        return sum(x.distance for x in self.routes)
