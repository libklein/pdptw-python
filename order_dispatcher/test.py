import math
import random
from copy import deepcopy
from pathlib import Path
from sys import argv
from typing import Iterable

from models import create_instance, Instance, Request, requests_per_driver
from solution import Solution, Route, Evaluation, ExactEvaluation, PenaltyFactors

driver_file = Path(argv[1])
order_file = Path(argv[2])

inst = create_instance(order_file, driver_file)


def plot(inst: Instance):
    from matplotlib import pyplot as plt

    c_map = {
        'driver': 'green',
        'pickup': 'red',
        'dropoff': 'blue'
    }

    plt.scatter(x=[x.location[0] for x in inst.vertices], y=[x.location[1] for x in inst.vertices],
                c=[c_map[x.vertex_type] for x in inst.vertices])

    for r in inst.requests:
        plt.annotate('', xy=(r.dropoff.location[0], r.dropoff.location[1]),
                     xytext=(r.pickup.location[0], r.pickup.location[1]), xycoords='data',
                     arrowprops=dict(arrowstyle='->'))

    plt.show()


# Build a solution
sol = Solution(inst)
# Add a request
for r in sol.routes:
    print(f'Route of vehicle {r._vehicle.vehicle_id}: ', r)

for req in [inst.requests[0], inst.requests[1], inst.requests[2]]:
    sol.routes[0].append(req)
    sol.routes[0].update()

    print(sol.routes[0])
    print(sol.routes[0].label)

req = inst.requests[3]

sol.routes[0].insert(req, 1, len(sol.routes[0]))
sol.routes[0].update()

print(sol.routes[0])
print(sol.routes[0].label)


def insert_requests_randomly(route: Route, requests: Iterable[Request]) -> Route:
    new_route = deepcopy(route)
    for r in requests:
        insert_pickup_pos = random.randint(1, len(new_route.nodes))
        insert_dropoff_pos = random.randint(insert_pickup_pos, len(new_route.nodes))
        new_route.insert(r, pickup_at=insert_pickup_pos, dropoff_at=insert_dropoff_pos)
    new_route.update()
    return new_route


def test_removal_evaluation(instance: Instance, penalty_factors: PenaltyFactors, n_tests=1000):
    evaluation = Evaluation(instance=inst, penalty_factors=penalty_factors,
                            target_fairness=requests_per_driver(instance))
    exact_evaluation = ExactEvaluation(instance, penalty_factors=penalty_factors,
                                       target_fairness=requests_per_driver(instance))
    sol = Solution(instance=instance)
    avg_req_per_route = sum(x.num_items for x in instance.requests) / sum(x.vehicle_capacity for x in instance.vehicles)
    # Test removal - create random route
    for test_num in range(n_tests):
        print(f"\rRemoval test {test_num}/{n_tests}", end='', flush=True)
        route = random.choice(sol.routes)
        request_set = random.sample(instance.requests, k=random.randint(0, math.ceil(2 * avg_req_per_route)))
        rand_route = insert_requests_randomly(route=route, requests=request_set)
        # Remove all requests in random order
        while len(rand_route.requests) > 0:
            next_removed_request = random.choice(rand_route.requests)

            simulated = evaluation.calculate_removal(next_removed_request, rand_route)
            exact = exact_evaluation.calculate_removal(next_removed_request, rand_route)

            assert exact.feasible == simulated.feasible
            assert abs(
                exact.delta_cost - simulated.delta_cost) <= 0.01, f'Expected {exact.delta_cost=} got {simulated.delta_cost=}'

            # Perform removal
            rand_route.remove(next_removed_request)
            rand_route.update()
    print()


def test_insertion_evaluation(instance: Instance, penalty_factors: PenaltyFactors, n_tests=1000):
    evaluation = Evaluation(instance=inst, penalty_factors=penalty_factors,
                            target_fairness=requests_per_driver(instance))
    exact_evaluation = ExactEvaluation(instance, penalty_factors=penalty_factors,
                                       target_fairness=requests_per_driver(instance))
    sol = Solution(instance=instance)
    avg_req_per_route = sum(x.num_items for x in instance.requests) / sum(x.vehicle_capacity for x in instance.vehicles)
    # Test removal - create random route
    for test_num in range(n_tests):
        print(f"\rInsertion test {test_num}/{n_tests}", end='', flush=True)
        route = deepcopy(random.choice(sol.routes))
        request_set = random.sample(instance.requests, k=random.randint(0, math.ceil(2 * avg_req_per_route)))
        for request_to_insert in request_set:
            insert_pickup_pos = random.randint(1, len(route.nodes))

            for exact_move, simulated_move in zip(
                    exact_evaluation.calculate_insertion(request_to_insert, route, at=insert_pickup_pos),
                    evaluation.calculate_insertion(request_to_insert, route, at=insert_pickup_pos)):
                assert (
                                   exact_move.delta_cost - simulated_move.delta_cost) < 0.01, f'{exact_move.delta_cost=} is not {simulated_move.delta_cost=}'

            insert_dropoff_pos = random.randint(insert_pickup_pos, len(route.nodes))
            route.insert(request_to_insert, pickup_at=insert_pickup_pos, dropoff_at=insert_dropoff_pos)
            route.update()
    print()


pen = PenaltyFactors(1.0, 1.0, 1., 100.)
test_insertion_evaluation(instance=inst, penalty_factors=pen)
print("------------------------------------")
test_removal_evaluation(instance=inst, penalty_factors=pen)
