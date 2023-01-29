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

solver = Solver(inst)
best_sol = solver.solve()