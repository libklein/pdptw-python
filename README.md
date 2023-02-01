# Assumptions

* Haversine radius 6364.757km (computed from avg lat and avg long (height above sea level), https://rechneronline.de/earth-radius/)
* Restaurants prepare order in parallel. (In alignment with Pol).
* Fairness relates to the number of orders assigned to each courier, not the courier's travel time.

# Approach

## Model

We model this problem as a `Heterogeneous Pickup and Delivery Problem with Soft Time Windows and Maximum Tour Duration`.
Here, a fleet of vehicles (i.e. Drivers), each with a certain starting location and capacity, needs to serve a set of requests.
A request comprises a pickup and a dropoff location. Picking up a request allocates a certain capacity released when dropping off. 
Each request has a time window that specifies the earliest pickup and latest dropoff time. 

Each order then represents a Pickup- and Delivery request where pickup occurs at the restaurant and dropoff at the customer's location.

We model the objective with a weighted objective function:
$$ c(S) = distance_cost + \alpha * OtDViolation + \beta * \sum_{c \in R} |c.numRequests - avgNumRequests| $$
and assume that weights are picked by domain experts.

## Methodology

We solve this problem using a LNS. 
The implementation comprises only a very limited number of operators, but provides easy-to-use interfaces to implement additional operators.

We allow solutions infeasible with respect to driver shift duration and freight capacity during the search procedure, penalized according to a generalized cost function:
$$ c(S) = distance_cost + \alpha * OtDViolation + \beta * \sum_{c \in R} |c.numRequests - avgNumRequests| + \gamma * \text{max capacity violation} + \mu * \text{overtime}$$
This allows to travel through infeasible solution spaces. We adapt these penalties according to the feasibility violations in the current solution. 
We note that allowing only feasible moves instead is straightforward in our implementation.
We define neighborhood moves on the request level, that is, move pickup and delivery locations together, such that Pickup & Delivery vertices cannot be mismatched.

We evaluate moves in constant time by tracking the state of partial routes.
This is straightforward for the OtdViolation, the capacity violation, distance, and maximum shift duration. 
Evaluating a move's impact on **courier fairness** is more challenging, as fairness is defined on the solution and not the route level. 
We attempt to satisfy it on a route-level by computing the optimal average utilization (orders per driver) and penalizing moves accordingly.
We chose to not normalize the deviation from the optimal average utilization as this skews the objective, i.e., the number of orders assigned determines fairness, not the average time spent en-route.
Finally, we always apply the first improving move found during the local search. This allows a efficient caching mechanism: 
we track the modification time of each route and the last time an operator evaluated moves for that route. This allows to, 
skip evaluating moves where all involved routes have not changed since the last evaluation.



## Implementation

In what follows, we briefly detail the fundamental classes of our local search procedure.

`[Solution]`

An aggregate of routes, one for each driver.

Invariant:
* Always one route for each driver

`[Route]`

An aggregate of `nodes` and `requests`. Represents the delivery route of a driver.

Invariant:
* Each request assigned to this route
* Node labels remain in a valid state (represent the )

* A sequence of nodes, i.e., vertices with forward and backward labels, 
* Time stamp of last modification

`[Nodes]`

An aggregate of vertices and labels. Associates each vertex with labels.

* We do not allow solutions with missmatched Pickup & Delivery.
* We allow infeasible solutions with respect to demand and shift length

* Efficient concatenation of demand is not optimal because it takes the maximum into account

* Guiding heuristic adapts penalty terms dynamically (Also cost terms?)