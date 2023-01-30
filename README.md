# The Problem

* We may have several orders per restaurant
* Drivers start at random points

### Objective

* Minimize travel time

### Constraints

#### Hard

* Number of drivers
* Driver capacity
* Driver shift

* Order release time (0 + order_ready_time), Pol has mentioned that a restaurant could prepare multiple orders in parallel
* All orders receive service

#### Soft

* OtD (order to delivery duration), latest time an order should appear. This is a route-based constraint.
* Courier fairness. This is a solution-based constraint.



# Assumptions

* Haversine radius 6364.757km (computed from avg lat and avg long (height above sea level), https://rechneronline.de/earth-radius/)

# Approach

## Model

We model this problem as a Pickup and Delivery Problem with Time Windows.
* Time Window of order equals [ready_time, driver_shift_end]. We do not use OtD here as it is not a hard constraint
	* Alternative, set to min(driver_shift_end, $OtD * \alpha$)
* Pick up location is the restaurant, drop off is the customer
* Heterogeneous Vehicles

I think it makes most sense to use a weighted cost function for this problem:
$$ c(S) = distance_cost + \alpha * OtDViolation + \beta * \sum_{c \in R} |c.numRequests - avgNumRequests| $$

One of the "problems" is that we have this **courier fairness** constraint. This constraint is, in contrast to all others,
related to the cost of a solution, not an individual route. We attempt to satisfy it on a route-level by computing the 
optimal average utilization (orders per driver) and penalizing moves accordingly. TODO: Normalization? Use avg. shift time instead?

Using the avg. shift time:
* May lead to bad solutions w.r.t. distance travelled (assign worse order to driver)

Using the avg. num of orders:
* Thats actually what we want to capture
* Not normalized, i.e., hard to choose $\beta$

## Solution Representation

* We do not allow solutions with missmatched Pickup & Delivery.
* We allow infeasible solutions with respect to demand and shift length

* Efficient concatenation of demand is not optimal because it takes the maximum into account

* Local Search procedure adapts penalty terms dynamically (Also cost terms?)