The solver is implemented in Python version 3.10. I have chosen to not rely on external dependencies I'd be using in production to remain within the scope of this challenge. 

Assumptions:
* Restaurants prepare order in parallel. (In alignment with Pol).
* Fairness relates to the number of orders assigned to each courier, not the courier's travel time.

# Approach
## Model

We model this problem as a `Pickup and Delivery Problem (PDP)` with heterogeneous vehicles, soft time windows, maximum tour duration, and fairness constraints.
Here, a set of couriers $c \in C$, possibly heterogeneous with respect to starting location $s_c$ and capacity $Q_c$, 
needs to serve a set of delivery requests $R$ where a quantity $q_r$ of items needs to be picked up at location $p_r$ 
and dropped of at another location $d_r$. For the sake of conciseness, we refer to the set of pickup, dropoff, and starting locations by $P$, $D$, and $S$ respectively.

Picking up a request allocates a certain amount of the vehicle's storage 
capacity released when dropping off. Each request $r$ has a time window $[e_r, l_r]$ that specifies the earliest 
pickup and latest dropoff time. We allow but penalize late service. Early service on the other hand is not allowed 
such that a vehicle arriving early at a pickup/dropoff point needs to wait. Beyond this, the maximum tour duration, 
i.e., total driving and waiting time, is limited by a driver's shift length $b_c$. Drivers start their shifts at time
$a_c$.

We define this `PDP` on a directed graph of delivery locations $G = (V, A)$, where $V = P \cup D \cup S$.
Each vertex $v \in V$ has a demand $q_v$ and a time window $[e_v, l_v]$. We set $q_v = 0, e_v = b_c, l_v = b_c$ for vertices 
representing courier starting locations and $q_v = q_r, e_v = e_r, l_v = l_r$ and $q_v = -q_r, e_v = e_r, l_v = l_r$ 
for pickup and dropoff vertices, respectively. We associate with each arc $(i, j) \in A$ the travel time from vertex 
$i$ to vertex $j$. 

The problem asks for a set of routes that minimize the following weighted objective function:
$$ obj(S) = \text{total travel time} + \alpha * \text{total OtD violation} + \beta * \text{deviation from fairest solution} $$
Here, we calculate the deviation from the fairest solution according to $\sum_{c \in C} | k_c - \frac{|R|}{|C|} |$, where $k_c$ corresponds to the number of requests assigned to courier $c$ in the current solution $S$. We chose to not normalize the deviation from the optimal average utilization as this skews the objective, i.e., the number of orders assigned determines fairness, not the average time spent en-route. We assume that weights $\alpha$ and $\beta$ are picked by domain experts. 

### Mapping Input Data to the Problem

Each order then represents a request where pickup occurs at the restaurant and dropoff at the customer's location.
We set the time window of the request according to the preparation time and OtD. The requested quantity of a request 
corresponds to number of items of the order.
We create a courier for each driver in the input data, setting starting location, capacity, and shift time/length accordingly.
Note that this entails a 1:N relationship between locations in the input data and vertices in the graph underlying the `PDP`.

## Methodology

We solve this problem using a LNS. We assume that the reader is familiar with the basic LNS concept and operators and refer to the [Handbook of Metaheuristics](https://link.springer.com/chapter/10.1007/978-1-4419-1665-5_13) for details.

## Aims and Scope

The implementation focuses on providing boilerplate code to easily implement a sophisticated LNS and hence
comprises only a very limited number of operators. However, implementing additional local search, destroy, or 
repair operators is trivial with the provided boilerplate code.

### Destroy/Repair

We use canonical random destroy and sequential best insertion operators.

### Local Search

We allow solutions infeasible with respect to driver shift duration and freight capacity during the search procedure, penalized according to a generalized cost function:
$$  obj(S) = \text{total travel time} + \alpha * \text{total OtD violation} + \beta * \text{deviation from fairest solution}  + \gamma * \text{max capacity violation} + \mu * \text{overtime}$$
where the maximum capacity violation captures the capacity that would be required to serve all requests in the route minus the actual vehicle capacity, and overtime refers to the time the courier spends en-route beyond her shift duration. 

This generalized cost function allows to travel through infeasible solution spaces. We adapt these penalties according to the feasibility violations in the current solution. 
We note that allowing only feasible moves instead is straightforward in our implementation.
We define neighborhood moves on the request level, that is, move pickup and delivery locations together, such that Pickup & Delivery vertices cannot be mismatched.

We evaluate moves in constant time by tracking the state of partial routes (See [Vidal 2013](https://doi.org/10.1016/j.cor.2012.07.018)).
This is straightforward for the OtD violation, the capacity violation, distance, and maximum shift duration. 
Evaluating a move's impact on **courier fairness** is more challenging, as fairness is defined on the solution and not the route level. 
We attempt to satisfy it on a route-level by computing the optimal (most fair) average utilization (orders per driver) and penalizing moves accordingly.
Finally, we always apply the first improving move found during the local search. This allows an efficient caching mechanism: 
we track the modification time of each route and the last time an operator evaluated moves for that route, 
such that the evaluation of moves where all involved routes did not change since the last evaluation can be skipped. 
We shuffle the route evaluation order at the start of each local search to avoid an overly greedy local search.

The local search procedure uses a single operator: relocate.

### Starting solution

We generate a starting solution by inserting requests into an initially empty solution in random order. Each request is
inserted at it's best possible position.

### Running the code

I have provided a shell script to run the code in default configuration. Use
```shell
python -m order_dispatcher -h
```
to see a list of supported command line arguments.

# Further improvements

* Benchmark against exact solver to validate the heuristic's performance. This was out-of-scope for this challenge.
* Improve performance. I did not focus on performance in this challenge, such that the heuristic is quite slow. For instance, right now, each update creates new labels rather than reusing already created ones.
* More operators. The heuristic would clearly benefit from better neighborhoods. There exist several well-known PDP/VRP operators for the large neighborhood and local search.
* Logging. I would use logging in production code.
* Tests. I did not implement any tests beyond a basic test of the constant time evaluation.
* Assertions/Validation. The models do not validate.