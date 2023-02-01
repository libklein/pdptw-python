import math
import time
from pathlib import Path
import random
from pathlib import Path
import argparse
from typing import Optional

from .models import create_instance, Request, Driver
from .solution import Solution, PenaltyFactors
from .solver import Solver


def cli():
    parser = argparse.ArgumentParser("Order dispatcher")
    parser.add_argument('driver_file', help='Path to the file containing driver data in csv format.')
    parser.add_argument('order_file', help='Path to the file containing order data in csv format.')
    parser.add_argument('--seed', action='store', dest='seed', default=None, help='Seed for the PRNG.')
    parser.add_argument('--time-limit', action='store', dest='time_limit_sec', default=60., help='Time limit in seconds.')

    parser.add_argument('--delay-factor', action='store', dest='delay_factor', type=float, default=1.,
                        help='Weight of the delay.')
    parser.add_argument('--fairness-factor', action='store', dest='fairness_factor', type=float, default=1000.,
                        help='Weight of the fairness.')

    cli_args = parser.parse_args().__dict__

    driver_file = Path(cli_args['driver_file'])
    order_file = Path(cli_args['order_file'])

    # I am not testing if the files are actually readable.
    if not driver_file.exists() or not driver_file.is_file():
        raise ValueError("Driver file {} does not exist or is not a file!")
    if not order_file.exists() or not order_file.is_file():
        raise ValueError("Order file {} does not exist or is not a file!")

    cli_args['driver_file'] = driver_file
    cli_args['order_file'] = order_file

    if cli_args['seed'] is None:
        cli_args['seed'] = int(time.time_ns())

    # I do no type-checking here. I would use a library for this in production.
    solve(**cli_args)


def solve(driver_file: Path, order_file: Path, time_limit_sec: float, delay_factor: float, fairness_factor: float, seed: int):
    random.seed(seed)

    inst = create_instance(order_file, driver_file)

    objective_coefficients = PenaltyFactors(delay_factor=delay_factor, overload_factor=0.,
                                            overtime_factor=0., fairness_factor=fairness_factor)

    solver = Solver(instance=inst, objective_function_factors=objective_coefficients)
    best_sol = None
    for sol in filter(lambda sol: sol.feasible, solver.solve()):
        best_sol = sol
        print(f"Found feasible solution with objective value {best_sol.get_objective(objective_coefficients)}")

    if not best_sol:
        raise RuntimeError("Failed to find a feasible solution")

    print(best_sol)

    def find_request_data(solution: Solution, of: Request) -> tuple[Driver, float, float]:
        route = solution.find_route(of.pickup)
        pickup_node, dropoff_node = route.get_node_of(of.pickup), route.get_node_of(of.dropoff)
        return route._vehicle.driver, pickup_node.forward_label.activity_start_time, dropoff_node.forward_label.activity_start_time

    # Export the solution
    for request in inst.requests:
        driver, pickup_time_sec, dropoff_time_sec = find_request_data(best_sol, request)
        # Since time windows are integer, ceil is always feasible for dropoff, floor for pickup
        print(request.order.order_id, driver.driver_id, math.floor(pickup_time_sec), math.ceil(dropoff_time_sec))

    # Note, i use the global random here. I'd inject instances of Random() in production code


if __name__ == '__main__':
    cli()
