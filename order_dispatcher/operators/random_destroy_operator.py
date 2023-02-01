# coding=utf-8
import random

from order_dispatcher.models import Request
from order_dispatcher.models.solution import Solution


class RandomDestroyOperator:
    def __init__(self, fraction_to_remove: float):
        """
        :param fraction_to_remove: The fraction of requests currently served by any passed solution to remove.
        """
        self._fraction_to_remove = fraction_to_remove

    def destroy(self, solution: Solution) -> set[Request]:
        num_requests = int(self._fraction_to_remove * solution.num_requests)
        requests_to_remove = random.sample(list(solution.requests), k=num_requests)
        for next_request in requests_to_remove:
            solution.remove_request(next_request)
        return set(requests_to_remove)
