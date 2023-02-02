# coding=utf-8
from __future__ import annotations

from typing import Protocol, Iterable

from order_dispatcher.models import Request, Route
from order_dispatcher.models.solution import Cost
from .move import RemovalMove, InsertionMove


class Evaluation(Protocol):
    """
    The evaluation interface. Allows to evaluate the impact of an insertion or removal.
    """

    def compute_cost(self, cost: Cost) -> float:
        ...

    def calculate_removal(self, of: Request, from_route: Route) -> RemovalMove:
        """
        :param of: The request
        :param from_route: The route to remove the request from
        :return: A removal move that encodes the removal operator and the impact it's impact (w.r.t. the objective) on the solution.
        """
        ...

    def calculate_insertion(self, of: Request, into_route: Route, at: int) -> Iterable[InsertionMove]:
        """
        Calculates all possible moves that insert request `request` into route `at` assuming that the pickup happens at
        position at. To give an example:
        Insering pickup a and dropoff b in route:
        [..., x, y, z]
            ^ at
        Would evaluate the following moves:
        [..., a, b ,x, y, z]
        [..., a ,x, b, y, z]
        [..., a, x, y, b, z]
        [..., a, x, y, z, b]
        :param of: The request to be inserted
        :param into_route: The route to insert into
        :param at: The position of the pickup in the route
        :return: A sequence of moves, i.e., one for each possible insertion
        """
        ...
