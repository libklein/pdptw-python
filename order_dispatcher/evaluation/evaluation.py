# coding=utf-8
from __future__ import annotations

from typing import Protocol, Iterable

from .move import RemovalMove, InsertionMove
from order_dispatcher.models import Request, Route
from order_dispatcher.models.solution import Cost


class Evaluation(Protocol):
    def compute_cost(self, cost: Cost) -> float:
        ...

    def calculate_removal(self, of: Request, from_route: Route) -> RemovalMove:
        ...

    def calculate_insertion(self, of: Request, into_route: Route, at: int) -> Iterable[InsertionMove]:
        ...
