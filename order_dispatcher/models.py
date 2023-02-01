from csv import DictReader
from dataclasses import dataclass, fields
from enum import IntEnum
from functools import partial
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import TypeVar, Type, Callable

# Pair of (lat, long)
Degrees = float
Location = tuple[Degrees, Degrees]
Timestamp = float
Duration = float


class VertexType(IntEnum):
    START = 0
    PICKUP = 1
    DROPOFF = 2


@dataclass(frozen=True)
class Order:
    order_id: int
    restaurant_location: Location
    customer_location: Location
    no_of_items: int
    prep_duration_sec: Duration
    preferred_otd_sec: Timestamp


@dataclass(frozen=True)
class Driver:
    driver_id: int
    shift_start_sec: Timestamp
    shift_end_sec: Timestamp
    start_location: Location
    vehicle_capacity: int


@dataclass(frozen=True)
class Vertex:
    vertex_id: int
    vertex_type: VertexType
    tw_start: Timestamp
    tw_end: Timestamp
    location: Location
    # The number of items available at this vertex. Negative for dropoff vertices.
    no_of_items: int

    def __str__(self):
        prefix = 'S'
        if self.is_pickup:
            prefix = 'P'
        elif self.is_dropoff:
            prefix = 'D'

        return f'{prefix}{self.vertex_id}'

    def __post_init__(self):
        assert sum(map(int, (self.is_start, self.is_pickup, self.is_dropoff))) == 1
        assert self.tw_start <= self.tw_end
        if self.is_start:
            assert self.no_of_items == 0
        elif self.is_dropoff:
            assert self.no_of_items < 0
        else:
            assert self.no_of_items > 0

    @property
    def is_start(self) -> bool:
        return self.vertex_type == VertexType.START

    @property
    def is_pickup(self) -> bool:
        return self.vertex_type == VertexType.PICKUP

    @property
    def is_dropoff(self) -> bool:
        return self.vertex_type == VertexType.DROPOFF


def compute_distance(from_lat_lon: Location, to_lat_lon: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = radians(from_lat_lon[0]), radians(from_lat_lon[1]), radians(to_lat_lon[0]), radians(
        to_lat_lon[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    earth_radius = 5300
    return c * earth_radius


def compute_travel_time(*args, speed_kmh: float, **kwargs) -> float:
    return (compute_distance(*args, **kwargs) / speed_kmh) * 3600


def compute_distance_matrix(distance: Callable[[Location, tuple[float, float]], float],
                            coordinates: list[tuple[float, float]]) -> list[list[float]]:
    return [[distance(i, j) for j in coordinates] for i in coordinates]


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    start: Vertex
    driver: Driver

    @property
    def capacity(self):
        return self.driver.vehicle_capacity

    @property
    def end_time(self):
        return self.driver.shift_end_sec

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

        assert self.pickup.no_of_items == self.num_items
        assert self.pickup.no_of_items == -self.dropoff.no_of_items


class Instance:
    def __init__(self, vertices: list[Vertex], travel_times: list[list[float]], vehicles: list[Vehicle],
                 requests: list[Request]):
        self._vertices = vertices
        self._travel_times = travel_times
        self._vehicles = vehicles
        self._requests = requests

    def get_travel_time(self, i: Vertex, j: Vertex) -> float:
        return self._travel_times[i.vertex_id][j.vertex_id]

    @property
    def vehicles(self):
        return self._vehicles

    @property
    def vertices(self) -> list[Vertex]:
        pass

    @property
    def requests(self) -> list[Request]:
        return self._requests

    @property
    def avg_requests_per_driver(self):
        return len(self._requests) / len(self._vehicles)


T = TypeVar('T')


def parse(factory: Callable[[dict[str, str]], T], file: Path) -> list[T]:
    with file.open() as fh:
        reader: DictReader = DictReader(fh)
        return [factory(x) for x in reader]


def parse_order(data: dict[str, str]) -> Order:
    return Order(order_id=int(data['order_id']), restaurant_location=(Degrees(data['restaurant_lat']),
                                                                      Degrees(data['restaurant_long'])),
                 customer_location=(Degrees(data['restaurant_lat']), Degrees(data['restaurant_long'])),
                 no_of_items=int(data['no_of_items']), prep_duration_sec=Timestamp(data['prep_duration_sec']),
                 preferred_otd_sec=Timestamp(data['preferred_otd_sec']))


def parse_orders(file: Path) -> list[Order]:
    return parse(parse_order, file)


def parse_driver(data: dict[str, str]) -> Driver:
    return Driver(int(data['driver_id']), shift_start_sec=Timestamp(data['shift_start_sec']),
                  shift_end_sec=Timestamp(data['shift_end_sec']), start_location=(Degrees(data['start_location_lat']),
                                                                                  Degrees(data['start_location_long'])),
                  vehicle_capacity=int(data['capacity']))


def parse_drivers(file: Path) -> list[Driver]:
    return parse(parse_driver, file)


def create_instance(order_file: Path, driver_file: Path) -> Instance:
    orders: list[Order] = parse_orders(order_file)
    drivers: list[Driver] = parse_drivers(driver_file)

    # Create vertices
    vertices = [
        Vertex(vertex_id=v_id, vertex_type=VertexType.START, tw_start=driver.shift_start_sec, tw_end=driver.shift_end_sec,
               location=driver.start_location, no_of_items=0) for v_id, driver in
        enumerate(drivers)
    ]

    vehicles = [
        Vehicle(vehicle_id=driver.driver_id, start=driver_start, driver=driver)
        for driver, driver_start in zip(drivers, vertices)]

    requests: list[Request] = []

    for order in orders:
        vertices.append(Vertex(vertex_id=len(vertices), vertex_type=VertexType.PICKUP,
                               tw_start=order.prep_duration_sec, tw_end=order.preferred_otd_sec,
                               location=order.restaurant_location, no_of_items=order.no_of_items))
        vertices.append(Vertex(vertex_id=len(vertices), vertex_type=VertexType.DROPOFF,
                               tw_start=order.prep_duration_sec, tw_end=order.preferred_otd_sec,
                               location=order.customer_location, no_of_items=-order.no_of_items))

        pickup, dropoff = vertices[-2], vertices[-1]
        requests.append(Request(pickup=pickup, dropoff=dropoff, num_items=order.no_of_items, order=order))

    # Create distance matrix
    travel_times = compute_distance_matrix(partial(compute_travel_time, speed_kmh=15), [x.location for x in vertices])

    return Instance(vertices, travel_times, vehicles, requests)
