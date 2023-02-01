# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterable

from order_dispatcher.models import Request, Route, Solution
from order_dispatcher.models.solution import Cost


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

class Evaluation(Protocol):
    def compute_cost(self, cost: Cost) -> float:
        ...

    def calculate_removal(self, of: Request, from_route: Route) -> RemovalMove:
        ...
    def calculate_insertion(self, of: Request, into_route: Route, at: int) -> Iterable[InsertionMove]:
        ...


