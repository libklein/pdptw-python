# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from order_dispatcher.models import Request, Route


class Move(Protocol):
    """
    A move, that is, something that modifies a solution. Always modifies the solution in-place.
    A reference to the solution is saved in the move itself.

    Note: A better implementation would take the solution/route as an argument (command pattern).
    But this would require route_ids rather than references.
    """

    @property
    def delta_cost(self) -> float:
        ...

    @property
    def feasible(self):
        """
        :return: True if the solution is feasible after a move.
        """
        ...

    @property
    def worthwhile(self) -> bool:
        """
        :return: True if the move is expected to improve the objective value.
        """
        ...

    def apply(self):
        ...


@dataclass(slots=True, frozen=True)
class InsertionMove:
    """
    Encodes inserting a request into a route at a specified position.
    """
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

    def apply(self):
        self.route.insert(self.request, self.pickup_insertion_point, self.dropoff_insertion_point)


@dataclass(slots=True, frozen=True)
class RemovalMove:
    """
    Encodes removing a request from a route.
    """
    delta_cost: float
    feasible: bool
    request: Request
    route: Route

    def __post_init__(self):
        assert self.request in self.route.requests

    def apply(self):
        self.route.remove(self.request)
