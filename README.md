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

* OtD (order to delivery duration), latest time an order should appear
* Courier fairness



# Assumptions

* Haversine radius 6364.757km (computed from avg lat and avg long (height above sea level), https://rechneronline.de/earth-radius/)

# Approach

## Model

We model this problem as a Pickup and Delivery Problem with Time Windows.
* Time Window of order equals [ready_time, driver_shift_end]. We do not use OtD here as it is not a hard constraint
	* Alternative, set to min(driver_shift_end, $OtD * \alpha$)
* Pick up location is the restaurant, drop off is the customer
