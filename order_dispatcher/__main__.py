import csv
import math
import time
import random
from pathlib import Path
import argparse

from order_dispatcher.models import create_instance, Request, Driver, Timestamp
from order_dispatcher.models.solution import Solution, PenaltyFactors
from order_dispatcher.solver import Solver


def cli():
    parser = argparse.ArgumentParser("Order dispatcher")
    parser.add_argument('driver_file', help='Path to the file containing driver data in csv format.')
    parser.add_argument('order_file', help='Path to the file containing order data in csv format.')
    parser.add_argument('--seed', action='store', dest='seed', default=None, help='Seed for the PRNG.')
    parser.add_argument('--time-limit', action='store', dest='time_limit_sec', default=60., help='Time limit in seconds, wall clock time.')

    parser.add_argument('--delay-factor', action='store', dest='delay_factor', type=float, default=1.,
                        help='Weight of the delay.')
    parser.add_argument('--fairness-factor', action='store', dest='fairness_factor', type=float, default=1000.,
                        help='Weight of the fairness.')
    parser.add_argument('--output-file', dest='output_file', default='solution.csv', help='Path to store the solution at.')

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
    cli_args['output_file'] = Path(cli_args['output_file'])

    if cli_args['seed'] is None:
        cli_args['seed'] = int(time.time_ns())

    # I do no type-checking here. I would use a library for this in production.
    solve(**cli_args)


def solve(driver_file: Path, order_file: Path, time_limit_sec: float, delay_factor: float, fairness_factor: float, seed: int, output_file: Path):
    # Note, i use the global random here. I'd inject instances of Random() in production code
    random.seed(seed)

    objective_coefficients = PenaltyFactors(delay_factor=delay_factor, overload_factor=0.,
                                            overtime_factor=0., fairness_factor=fairness_factor)

    inst = create_instance(order_file, driver_file)

    solver = Solver(instance=inst, objective_function_factors=objective_coefficients, time_limit_sec=time_limit_sec)
    best_sol = None
    for sol in filter(lambda sol: sol.feasible, solver.solve()):
        best_sol = sol
        print(f"Found feasible solution with objective value {best_sol.get_objective(objective_coefficients)}")

    if not best_sol:
        raise RuntimeError("Failed to find a feasible solution")

    print(best_sol)

    def find_request_data(solution: Solution, of: Request) -> tuple[Driver, Timestamp, Timestamp]:
        route = solution.find_route(of.pickup)
        pickup_node, dropoff_node = route.get_node_of(of.pickup), route.get_node_of(of.dropoff)
        return route._vehicle.driver, pickup_node.forward_label.activity_start_time, dropoff_node.forward_label.activity_start_time

    # Export the solution
    with output_file.open('w') as output_stream:
        csv_writer = csv.DictWriter(output_stream, fieldnames=['order_id', 'driver_id', 'estimated_pickuptime_sec', 'estimated_deliverytime_sec'])
        csv_writer.writeheader()
        for request in inst.requests:
            driver, pickup_time_sec, dropoff_time_sec = find_request_data(best_sol, request)
            # Since time windows are integer, ceil is always feasible for dropoff, floor for pickup
            csv_writer.writerow(dict(order_id=request.order.order_id, driver_id=driver.driver_id,
                                     estimated_pickuptime_sec=int(math.floor(pickup_time_sec)),
                                     estimated_deliverytime_sec=int(math.ceil(dropoff_time_sec))))

if __name__ == '__main__':
    cli()
