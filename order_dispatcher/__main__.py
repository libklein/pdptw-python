import argparse
import csv
import math
import random
import time
from pathlib import Path

from order_dispatcher.models import create_instance, Request, Instance
from order_dispatcher.models.solution import Solution, PenaltyFactors
from order_dispatcher.solver import Solver


def cli():
    parser = argparse.ArgumentParser("Order dispatcher")
    parser.add_argument('driver_file', help='Path to the file containing driver data in csv format.')
    parser.add_argument('order_file', help='Path to the file containing order data in csv format.')
    parser.add_argument('--seed', action='store', dest='seed', default=None, help='Seed for the PRNG.')
    parser.add_argument('--time-limit', action='store', dest='time_limit_sec', default=60.,
                        help='Time limit in seconds, wall clock time.')

    parser.add_argument('--delay-factor', action='store', dest='delay_factor', type=float, default=1.,
                        help='Weight of the delay.')
    parser.add_argument('--fairness-factor', action='store', dest='fairness_factor', type=float, default=1000.,
                        help='Weight of the fairness.')
    parser.add_argument('--output-file', dest='output_file', default='solution.csv',
                        help='Path to store the solution at.')

    cli_args = parser.parse_args().__dict__

    driver_file = Path(cli_args['driver_file'])
    order_file = Path(cli_args['order_file'])

    # I am not testing if the files are actually readable.
    if not driver_file.exists() or not driver_file.is_file():
        raise ValueError("Driver file {} does not exist or is not a file!")
    if not order_file.exists() or not order_file.is_file():
        raise ValueError("Order file {} does not exist or is not a file!")

    # Do necessary conversions. We could also do this via reflection on the signature of solve or using the arguments
    # list of parser, but I wanted to keep the code simple.
    cli_args['driver_file'] = driver_file
    cli_args['order_file'] = order_file
    cli_args['output_file'] = Path(cli_args['output_file'])
    cli_args['time_limit_sec'] = int(cli_args['time_limit_sec'])
    cli_args['delay_factor'] = float(cli_args['delay_factor'])
    cli_args['fairness_factor'] = float(cli_args['fairness_factor'])

    if cli_args['seed'] is None:
        cli_args['seed'] = int(time.time_ns())
    else:
        cli_args['seed'] = int(cli_args['seed'])

    # I do no type-checking here. I would use a library for this in production.
    solve(**cli_args)


def solve(driver_file: Path, order_file: Path, time_limit_sec: float, delay_factor: float, fairness_factor: float,
          seed: int, output_file: Path):
    # Note, I use the global random here. I'd inject instances of Random() in production code
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

    print(f"Terminating search with objective value {best_sol.get_objective(objective_coefficients)}")

    _export_solution(best_sol, inst, output_file)


def _export_solution(best_sol: Solution, instance: Instance, output_file: Path):
    def find_request_data(request: Request):
        request_route = best_sol.find_route(request)
        pickup_node, dropoff_node = request_route.get_node_of(request.pickup), request_route.get_node_of(
            request.dropoff)
        return request_route.assigned_driver.driver_id, pickup_node.activity_start_time, dropoff_node.activity_start_time

    # Export the solution
    with output_file.open('w') as output_stream:
        csv_writer = csv.DictWriter(output_stream, fieldnames=['order_id', 'driver_id', 'estimated_pickuptime_sec',
                                                               'estimated_deliverytime_sec'], delimiter=' ')
        csv_writer.writeheader()
        for request in instance.requests:
            driver_id, pickup_time_sec, dropoff_time_sec = find_request_data(request)
            # Since time windows are integer, ceil is always feasible for dropoff, floor for pickup
            csv_writer.writerow(dict(order_id=request.order.order_id, driver_id=driver_id,
                                     estimated_pickuptime_sec=int(math.floor(pickup_time_sec)),
                                     estimated_deliverytime_sec=int(math.ceil(dropoff_time_sec))))


if __name__ == '__main__':
    cli()
