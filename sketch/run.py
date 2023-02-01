import math
from copy import deepcopy
from pathlib import Path
from sys import argv
from typing import Iterable

from models import create_instance, Instance, Request, Driver
from solver import Solver
from solution import Solution, Route, Node, Label, Evaluation
import random

# Note, i use the global random here. I'd inject instances of Random() in production code
random.seed(0)

driver_file = Path(argv[1])
order_file = Path(argv[2])

inst = create_instance(order_file, driver_file)

solver = Solver(inst)
*_, best_sol = filter(lambda sol: sol.feasible, solver.solve())

print(best_sol)

def find_request_data(solution: Solution, of: Request) -> tuple[Driver, float, float]:
    route = solution.find_route(of.pickup)
    pickup_node, dropoff_node = route.get_node_of(of.pickup), route.get_node_of(of.dropoff)
    return route._vehicle.driver, pickup_node.forward_label.activity_start_time, dropoff_node.forward_label.activity_start_time

if not best_sol.feasible:
    raise RuntimeError("Failed to find a feasible solution")

# Export the solution
"""
for request in inst.requests:
    driver, pickup_time_sec, dropoff_time_sec = find_request_data(best_sol, request)
    # Since time windows are integer, ceil is always feasible for dropoff, floor for pickup
    print(request.order.order_id, driver.driver_id, math.floor(pickup_time_sec), math.ceil(dropoff_time_sec))
"""