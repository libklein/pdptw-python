import math
from copy import deepcopy
from pathlib import Path
from sys import argv
from typing import Iterable

from models import create_instance, Instance, Request
from solver import Solver
from solution import Solution, Route, Node, Label, Evaluation
import random

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

    plt.scatter(x=[x.lat_long[0] for x in inst.vertices],y=[x.lat_long[1] for x in inst.vertices], c=[c_map[x.vertex_type] for x in inst.vertices])

    for r in inst.requests:
        plt.annotate('', xy=(r.dropoff.lat_long[0], r.dropoff.lat_long[1]), xytext=(r.pickup.lat_long[0],r.pickup.lat_long[1]), xycoords='data', arrowprops=dict(arrowstyle='->'))

    plt.show()

# Build a solution
sol = Solution(inst)
# Add a request
for r in sol.routes:
    print(f'Route of vehicle {r._vehicle.vehicle_id}: ',r)

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

def test_removal_evaluation(instance: Instance, n_tests=1000):
    evaluation = Evaluation(instance=inst)
    sol = Solution(instance=instance)
    avg_req_per_route = sum(x.num_items for x in instance.requests)/sum(x.capacity for x in instance.vehicles)
    # Test removal - create random route
    for _ in range(n_tests):
        route = random.choice(sol.routes)
        request_set = random.sample(instance.requests, k=random.randint(0, math.ceil(2*avg_req_per_route)))
        rand_route = insert_requests_randomly(route=route, requests=request_set)
        # Remove all requests in random order
        while len(route.requests) > 0:
            next_removed_request = random.choice(route.requests)
            simulated = evaluation.calculate_removal(next_removed_request, rand_route)

            prev_cost = evaluation.compute_cost(rand_route.label)

            rand_route.remove(next_removed_request)
            rand_route.update()
            assert rand_route.feasible == simulated.feasible
            assert (evaluation.compute_cost(rand_route.label) - prev_cost) == simulated.delta_cost

def test_insertion_evaluation(instance: Instance, n_tests=1000):
    evaluation = Evaluation(instance=inst)
    sol = Solution(instance=instance)
    avg_req_per_route = sum(x.num_items for x in instance.requests)/sum(x.capacity for x in instance.vehicles)
    # Test removal - create random route
    for _ in range(n_tests):
        route = deepcopy(random.choice(sol.routes))
        request_set = random.sample(instance.requests, k=random.randint(0, math.ceil(2*avg_req_per_route)))
        for request_to_insert in request_set:
            insert_pickup_pos = random.randint(1, len(route.nodes))

            for simulated_move in evaluation.calculate_insertion(request_to_insert, route, at=insert_pickup_pos):
                tmp_route = deepcopy(route)
                prev_cost = evaluation.compute_cost(tmp_route.label)

                tmp_route.insert(request_to_insert, pickup_at=simulated_move.pickup_insertion_point,
                                 dropoff_at=simulated_move.dropoff_insertion_point)
                tmp_route.update()
                assert tmp_route.feasible == simulated_move.feasible

                computed_cost = evaluation.compute_cost(tmp_route.label)
                assert ((computed_cost - prev_cost) - simulated_move.delta_cost) < 0.01, f'{computed_cost-prev_cost=} is not {simulated_move.delta_cost=}'

            insert_dropoff_pos = random.randint(insert_pickup_pos, len(route.nodes))
            route.insert(request_to_insert, pickup_at=insert_pickup_pos, dropoff_at=insert_dropoff_pos)
            route.update()

test_insertion_evaluation(instance=inst)
test_removal_evaluation(instance=inst)