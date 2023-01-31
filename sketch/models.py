from dataclasses import dataclass, fields
from typing import TypeVar, Type, Callable
from csv import DictReader
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
from functools import partial

Coordinate = tuple[float, float]

@dataclass(frozen=True)
class Order:
    order_id: int
    restaurant_lat: float
    restaurant_long: float
    customer_lat: float
    customer_long: float
    no_of_items: int
    prep_duration_sec: float
    preferred_otd_sec: float

@dataclass(frozen=True)
class Driver:
    driver_id: int
    shift_start_sec: float
    shift_end_sec: float
    start_location_lat: float
    start_location_long: float
    capacity: int

@dataclass(frozen=True)
class Vertex:
    vertex_id: int
    vertex_type: str
    tw_start: float
    tw_end: float
    lat_long: Coordinate
    items: int

    def __str__(self):
        prefix = 'S'
        if self.is_pickup:
            prefix = 'P'
        elif self.is_dropoff:
            prefix = 'D'

        return f'{prefix}{self.vertex_id}'
    
    def __post_init__(self):
        assert sum(map(int, (self.is_start,self.is_pickup,self.is_dropoff))) == 1
        assert self.tw_start <= self.tw_end

    @property
    def is_start(self) -> bool:
        return self.vertex_type == 'driver'

    @property
    def is_pickup(self) -> bool:
        return self.vertex_type == 'pickup'

    @property
    def is_dropoff(self) -> bool:
        return self.vertex_type == 'dropoff'

def compute_distance(from_lat_lon: Coordinate, to_lat_lon: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = radians(from_lat_lon[0]), radians(from_lat_lon[1]), radians(to_lat_lon[0]), radians(to_lat_lon[1])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    earth_radius = 5300
    return c * earth_radius

def compute_travel_time(*args, speed_kmh: float, **kwargs) -> float:
    return (compute_distance(*args, **kwargs) / speed_kmh) * 3600

def compute_distance_matrix(distance: Callable[[Coordinate, tuple[float, float]],float], coordinates: list[tuple[float, float]]) -> list[list[float]]:
    return [[distance(i, j) for j in coordinates] for i in coordinates]

@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    start_time: float
    end_time: float
    start: Vertex
    capacity: int
    driver: Driver

@dataclass(frozen=True)
class Request:
    pickup: Vertex
    dropoff: Vertex
    num_items: int
    order: Order

    def __post_init__(self):
        assert self.num_items > 0
        assert self.pickup.is_pickup
        assert self.dropoff.is_dropoff

        assert self.pickup.items == self.num_items
        assert self.pickup.items == -self.dropoff.items

class Instance:
    def __init__(self, vertices: list[Vertex], travel_times: list[list[float]], vehicles: list[Vehicle], requests: list[Request]):
        self.vertices = vertices
        self.travel_times = travel_times
        self.vehicles = vehicles
        self.requests = requests

        self._adjacency = [sorted(filter(lambda x: x.vertex_id == i.vertex_id, self.vertices), key=lambda j: self.get_travel_time(i, j)) for i in self.vertices]
        self._non_start_vertices = list(filter(lambda x: not x.is_start, self.vertices))

    @property
    def non_start_vertices(self) -> list[Vertex]:
        return self._non_start_vertices

    def get_travel_time(self, i: Vertex, j: Vertex) -> float:
        return self.travel_times[i.vertex_id][j.vertex_id]

    def get_adjacent_vertices(self, of: Vertex, include_start=False):
        if include_start:
            return self._adjacency[of.vertex_id]
        else:
            return filter(lambda x: not x.is_start, self._adjacency[of.vertex_id])

T = TypeVar('T', Type[Order], Type[Driver])

def construct_typed(cls: T, **data):
    _fields = {x.name: x for x in fields(cls)}
    return cls(**{key: _fields[key].type(val) for key, val in data.items()})

def parse(cls: T, file: Path):
    with file.open() as fh:
        reader: DictReader = DictReader(fh)
        return [construct_typed(cls, **x) for x in reader]

def parse_orders(file: Path) -> list[Order]:
    return parse(Order, file)


def parse_drivers(file: Path) -> list[Driver]:
    return parse(Driver, file)

def create_instance(order_file: Path, driver_file: Path) -> Instance:
    orders: list[Order] = parse_orders(order_file)
    drivers: list[Driver] = parse_drivers(driver_file)

    # Create vertices
    vertices = [
        Vertex(vertex_id=v_id, vertex_type='driver', tw_start=driver.shift_start_sec, tw_end=driver.shift_end_sec, lat_long=(driver.start_location_lat, driver.start_location_long), items=0) for v_id, driver in enumerate(drivers)
    ]

    vehicles = [Vehicle(vehicle_id=i, start_time=driver_start.tw_start, end_time=driver_start.tw_end, start=driver_start, capacity=driver.capacity, driver=driver) for i, (driver, driver_start) in enumerate(zip(drivers, vertices))]

    requests: list[Request] = []

    for order in orders:
        vertices.append(Vertex(vertex_id=len(vertices), vertex_type='pickup',
                               tw_start=order.prep_duration_sec, tw_end=order.preferred_otd_sec,
                               lat_long=(order.restaurant_lat, order.restaurant_long), items=order.no_of_items))
        vertices.append(Vertex(vertex_id=len(vertices), vertex_type='dropoff',
                               tw_start=order.prep_duration_sec, tw_end=order.preferred_otd_sec,
                               lat_long=(order.customer_lat, order.customer_long), items=-order.no_of_items))

        pickup, dropoff = vertices[-2], vertices[-1]
        requests.append(Request(pickup=pickup, dropoff=dropoff, num_items=order.no_of_items, order=order))

    # Create distance matrix
    travel_times = compute_distance_matrix(partial(compute_travel_time, speed_kmh=15), [x.lat_long for x in vertices])

    return Instance(vertices, travel_times, vehicles, requests)

def requests_per_driver(instance: Instance) -> float:
    return len(instance.requests)/len(instance.vehicles)

