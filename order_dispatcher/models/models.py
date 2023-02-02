import itertools
from csv import DictReader
from dataclasses import dataclass, fields
from enum import IntEnum
from functools import partial
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import TypeVar, Type, Callable

# Degrees of Lat/Long
Degrees = float
# Pair of (lat, long)
Location = tuple[Degrees, Degrees]
# Absolute time in seconds
Timestamp = float
# Duration between two timestamps, in seconds
Duration = float
# Distance in Kilometers
Distance = float
# Matrix mapping a pair of locations to the duration required to travel between them. Should be moved into a class
# and could potentially be implemented as a map. I chose not to do this in this draft for performance reasons.
TravelTimeMatrix = list[list[Duration]]
# Generic type
T = TypeVar('T')

class VertexType(IntEnum):
    """Enum representing the type of vertex, either START, PICKUP, or DROPOFF"""
    START = 0
    PICKUP = 1
    DROPOFF = 2


@dataclass(frozen=True)
class Order:
    """Dataclass representing an order.

    Attributes:
        order_id (int): The order's unique identifier.
        restaurant_location (Location): The location of the restaurant.
        customer_location (Location): The location of the customer.
        no_of_items (int): The number of items in the order.
        prep_duration_sec (Duration): The preparation time of the order in seconds.
        preferred_otd_sec (Timestamp): The preferred time of delivery in seconds.
    """
    order_id: int
    restaurant_location: Location
    customer_location: Location
    no_of_items: int
    prep_duration_sec: Duration
    preferred_otd_sec: Timestamp


@dataclass(frozen=True)
class Driver:
    """Dataclass representing a driver.

    Attributes:
        driver_id (int): The driver's unique identifier.
        shift_start_sec (Timestamp): The start time of the driver's shift in seconds.
        shift_end_sec (Timestamp): The end time of the driver's shift in seconds.
        start_location (Location): The starting location of the driver.
        vehicle_capacity (int): The capacity of the driver's vehicle.
    """
    driver_id: int
    shift_start_sec: Timestamp
    shift_end_sec: Timestamp
    start_location: Location
    vehicle_capacity: int


@dataclass(frozen=True)
class Vertex:
    """Dataclass representing a vertex. Vertices form the base of the routing problem:
    Each vertex represents a (not nessesarily unique) location, such that we have a vertex
    for each pickup, dropoff, and driver starting location.

    Attributes:
        vertex_id (int): The unique identifier of the vertex.
        vertex_type (VertexType): The type of vertex, either START, PICKUP, or DROPOFF.
        tw_start (Timestamp): The start time window of the vertex in seconds.
        tw_end (Timestamp): The end time window of the vertex in seconds.
        location (Location): The location of the vertex.
        no_of_items (int): The number of items available at this vertex. Negative for dropoff vertices.
    """
    vertex_id: int
    vertex_type: VertexType
    tw_start: Timestamp
    tw_end: Timestamp
    location: Location
    no_of_items: int

    def __str__(self):
        """Convert the vertex to a human-readable string representation."""
        prefix = 'S'
        if self.is_pickup:
            prefix = 'P'
        elif self.is_dropoff:
            prefix = 'D'

        return f'{prefix}{self.vertex_id}'

    def __post_init__(self):
        """Check the validity of the vertex after initialization."""
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


def compute_distance(from_lat_lon: Location, to_lat_lon: Location) -> Distance:
    """
    Computes the haversine function between two Locations. Assumes a earth radius of 6364,757km. This corresponds to
    the radius observed at the average over all locations in the sample CSV files.

    :param from_lat_lon: origin location
    :param to_lat_lon: target location
    :return: The function in kilometers.
    """
    lat1, lon1, lat2, lon2 = radians(from_lat_lon[0]), radians(from_lat_lon[1]), radians(to_lat_lon[0]), radians(
        to_lat_lon[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    earth_radius = 6364.757
    return c * earth_radius


def compute_travel_time(from_lat_lon: Location, to_lat_lon: Location, speed_kmh: float) -> Duration:
    """
    Computes the travel time between two locations.
    :param from_lat_lon: origin location
    :param to_lat_lon: target location
    :param speed_kmh: The average vehicle speed
    :return: The travel time in seconds.
    """
    return (compute_distance(from_lat_lon, to_lat_lon) / speed_kmh) * 3600


def compute_distance_matrix(function: Callable[[Location, Location], T],
                            locations: list[Location]) -> list[list[T]]:
    """
    Computes a matrix from a list of locations by applying a function to every pairwise combination of locations, i.e.,
    matrix[i][j] = function(locations[i], locations[j]).
    :param function: A function mapping a pair of locations to some value.
    :param locations: A list of locations
    :return: The calculated matrix.
    """
    return [[function(i, j) for j in locations] for i in locations]


@dataclass(frozen=True)
class Vehicle:
    """
    Class to represent a Vehicle instance.

    Attributes:
    vehicle_id (int): ID of the vehicle.
    start (Vertex): Starting vertex of the vehicle.
    driver (Driver): Driver of the vehicle.

    Properties:
    capacity (int): Capacity of the vehicle.
    shift_begin (Timestamp): Start time of the driver's shift.
    shift_end (Timestamp): End time of the driver's shift.
    """
    vehicle_id: int
    start: Vertex
    driver: Driver

    @property
    def capacity(self):
        return self.driver.vehicle_capacity

    @property
    def shift_start(self)-> Timestamp:
        return self.driver.shift_start_sec

    @property
    def shift_end(self)-> Timestamp:
        return self.driver.shift_end_sec

@dataclass(frozen=True)
class Request:
    """
    Class to represent a Request instance.

    Attributes:
    pickup (Vertex): Vertex where pickup occurs.
    dropoff (Vertex): Vertex where dropoff occurs.
    num_items (int): Number of items being requested.
    order (Order): Order instance associated with the request.
    """
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
    """
    A instance of the routing problem.
    The class assumes that the ID of a vertex corresponds to it's position in the list of vertices.
    """
    def __init__(self, vertices: list[Vertex], travel_times: TravelTimeMatrix, vehicles: list[Vehicle],
                 requests: list[Request]):
        assert len(vertices) > 0
        assert vertices[0].vertex_id == 0
        assert all((x.vertex_id == y.vertex_id - 1) for x, y in itertools.pairwise(vertices))
        assert len(travel_times) == len(vertices) and all(len(x) == len(vertices) for x in travel_times)
        assert all(x.dropoff in vertices and x.pickup in vertices for x in requests)
        assert all(x.start in vertices for x in vehicles)

        self._vertices = vertices
        self._travel_times = travel_times
        self._vehicles = vehicles
        self._requests = requests

    def get_travel_time(self, i: Vertex, j: Vertex) -> Duration:
        """
        Get the travel time required to travel from vertex i to vertex j.
        :param i: Origin vertex
        :param j: Target vertex
        :return: The travel time between i and j.
        """
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

def parse(factory: Callable[[dict[str, str]], T], file: Path) -> list[T]:
    """
    Parses a csv file using a used-defined factory function.
    :param factory: Constructs and object from a row read from the csv file
    :param file: Path to a csv file
    :return: A list of objects obtained by applying factory to each row.
    """
    with file.open() as fh:
        reader: DictReader = DictReader(fh)
        return [factory(x) for x in reader]


def parse_order(data: dict[str, str]) -> Order:
    """
    Deserializes data into an Order.
    """
    return Order(order_id=int(data['order_id']), restaurant_location=(Degrees(data['restaurant_lat']),
                                                                      Degrees(data['restaurant_long'])),
                 customer_location=(Degrees(data['customer_lat']), Degrees(data['customer_long'])),
                 no_of_items=int(data['no_of_items']), prep_duration_sec=Timestamp(data['prep_duration_sec']),
                 preferred_otd_sec=Timestamp(data['preferred_otd_sec']))


def parse_orders(file: Path) -> list[Order]:
    """
    Parses a list of orders from a csv formatted file.
    :param file: Path to the file
    :return: A list of orders
    """
    return parse(parse_order, file)


def parse_driver(data: dict[str, str]) -> Driver:
    """
    Deserializes data into a Driver.
    """
    return Driver(int(data['driver_id']), shift_start_sec=Timestamp(data['shift_start_sec']),
                  shift_end_sec=Timestamp(data['shift_end_sec']), start_location=(Degrees(data['start_location_lat']),
                                                                                  Degrees(data['start_location_long'])),
                  vehicle_capacity=int(data['capacity']))


def parse_drivers(file: Path) -> list[Driver]:
    """
    Parses a list of drivers from a csv formatted file.
    :param file: Path to the file
    :return: A list of drivers
    """
    return parse(parse_driver, file)


def create_instance(order_file: Path, driver_file: Path) -> Instance:
    """
    Creates an instance from order and driver csv files.
    :param order_file: Path to the order csv file.
    :param driver_file: Path to the driver csv file.
    :return:
    :rtype:
    """
    orders: list[Order] = parse_orders(order_file)
    drivers: list[Driver] = parse_drivers(driver_file)

    # Create a vertex for each driver starting location
    vertices = [
        Vertex(vertex_id=v_id, vertex_type=VertexType.START, tw_start=driver.shift_start_sec, tw_end=driver.shift_end_sec,
               location=driver.start_location, no_of_items=0) for v_id, driver in
        enumerate(drivers)
    ]

    # Create vehicles from drivers and their starting locations
    vehicles = [
        Vehicle(vehicle_id=driver.driver_id, start=driver_start, driver=driver)
        for driver, driver_start in zip(drivers, vertices)]

    requests: list[Request] = []
    for order in orders:
        # Create a vertex for each pickup and each dropoff location.
        vertices.append(Vertex(vertex_id=len(vertices), vertex_type=VertexType.PICKUP,
                               tw_start=order.prep_duration_sec, tw_end=order.preferred_otd_sec,
                               location=order.restaurant_location, no_of_items=order.no_of_items))
        vertices.append(Vertex(vertex_id=len(vertices), vertex_type=VertexType.DROPOFF,
                               tw_start=order.prep_duration_sec, tw_end=order.preferred_otd_sec,
                               location=order.customer_location, no_of_items=-order.no_of_items))

        pickup, dropoff = vertices[-2], vertices[-1]
        requests.append(Request(pickup=pickup, dropoff=dropoff, num_items=order.no_of_items, order=order))

    # Create travel time matrix
    travel_times = compute_distance_matrix(partial(compute_travel_time, speed_kmh=15), [x.location for x in vertices])

    return Instance(vertices, travel_times, vehicles, requests)
