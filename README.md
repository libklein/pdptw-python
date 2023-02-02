The solver is implemented in Python version 3.10. I decided against external dependencies to not cause additional overhead on your side.

### Running the code

I have provided a shell script to run the code in default configuration. Use
```shell
python -m order_dispatcher -h
```
to see a list of supported command line arguments.

# Approach

### Assumptions:
* Restaurants prepare orders in parallel. Specifically, two orders with preparation times a and b assigned to the same restaurant become available at a and b, respectively (In alignment with Pol).
* *Fairness* relates to the number of orders assigned to each courier, not the courier's travel time.
* Couriers can, if their capacity allows, deliver multiple orders simultaneously.

## Model

I model this problem as a `Pickup and Delivery Problem (PDP)` with heterogeneous vehicles, soft time windows, 
maximum tour duration, and fairness constraints. Here, a set of couriers $c \in C$, possibly heterogeneous concerning
starting location $s_c$ and capacity $Q_c$, must serve a set of delivery requests $R$. Fulfilling a 
request entails picking up a quantity $q_r$ of items at pick up location $p_r$, and dropping those of at another 
location $d_r$ at a later time. The maximum number of orders served simultaneously is thus limited by the 
vehicle's capacity. For conciseness, I refer to the set of pickup, dropoff, and starting locations by $P$,
$D$, and $S$, respectively.

Each request $r$ has a time window $[e_r, l_r]$ that specifies the earliest 
pickup and latest dropoff time. I allow but penalize late service. Early service, however, is not allowed 
such that a vehicle arriving early at a pickup/dropoff point must wait. Beyond this, the maximum tour duration, 
i.e., total driving and waiting time, is limited by a courier's shift length $b_c$. Drivers start their shifts at time
$a_c$.

I define this `PDP` on a directed graph of delivery locations $G = (V, A)$, where $V = P \cup D \cup S$.
Each vertex $v \in V$ has a demand $q_v$ and a time window $[e_v, l_v]$. I set $q_v = 0, e_v = b_c, l_v = b_c$ for vertices 
representing courier starting locations and $q_v = q_r, e_v = e_r, l_v = l_r$ and $q_v = -q_r, e_v = e_r, l_v = l_r$ 
for pickup and dropoff vertices, respectively. I associate with each arc $(i, j) \in A$ the travel time from vertex 
$i$ to vertex $j$. 

The problem asks for a set of routes that minimize the following weighted objective function:
$$ obj(S) = \text{total travel time} + \alpha * \text{total OtD violation} + \beta * \text{deviation from fairest solution} $$
Here, I calculate the deviation from the fairest solution according to $\sum_{c \in C} | k_c - \frac{|R|}{|C|} |$, 
where $k_c$ corresponds to the number of requests assigned to courier $c$ in the current solution $S$. 
I chose not to normalize the deviation from the optimal average utilization as this skews the objective, i.e., 
the number of orders assigned determines fairness, not the average time spent en-route. 
I assume that domain experts pick weights $\alpha$ and $\beta$.

### Mapping Input Data to the Problem

Each order then represents a request where pickup occurs at the restaurant and dropoff at the customer's location.
I set the time window of the request according to the preparation time and OtD. The requested quantity of a request 
corresponds to the number of items in the order.
I create a courier for each driver in the input data, setting starting location, capacity, and shift time/length accordingly.
Note that this entails a 1:N relationship between locations in the input data and vertices in the graph underlying the `PDP`.

## Methodology

I solve this problem using a LNS. I assume that the reader is familiar with the basic LNS concept and operators and 
refer to the [Handbook of Metaheuristics](https://link.springer.com/chapter/10.1007/978-1-4419-1665-5_13) for details.

## Aims and Scope

The implementation focuses on providing boilerplate code to easily implement a sophisticated LNS and hence
comprises only a limited number of operators. However, implementing additional local search, destroy, and 
repair operators remains trivial with the provided boilerplate code.

### Destroy/Repair

I use canonical random destroy and sequential best insertion operators.

### Local Search

I allow solutions infeasible concerning courier shift duration and freight capacity during the search procedure, 
penalized according to a generalized cost function:
$$ obj(S) = \text{total travel time} + \alpha * \text{total OtD violation} + \beta * \text{deviation from fairest solution} + \gamma * \text{max capacity violation} + \mu * \text{overtime} $$
where the maximum capacity violation captures the capacity that would be required to serve all requests in the route 
minus the actual vehicle capacity, and overtime refers to the time the courier spends en-route beyond her shift duration. 

This generalized cost function allows traveling through infeasible solution spaces. I adapt these penalties according 
to the feasibility violations in the current solution. I note that allowing only feasible moves instead is 
straightforward in our implementation. I define neighborhood moves on the request level, that is, 
move pickup and delivery locations together, such that Pickup & Delivery vertices cannot be mismatched.

I evaluate moves in constant time by tracking the state of partial routes (See [Vidal 2013](https://doi.org/10.1016/j.cor.2012.07.018)).
This is straightforward for the OtD violation, the capacity violation, distance, and maximum shift duration. 
Evaluating a move's impact on **courier fairness** is more challenging, as fairness is defined on the solution and 
not the route level. I attempt to satisfy it on a route level by computing the optimal (most fair) average utilization 
(orders per courier) and penalizing moves accordingly. Finally, I apply the first improving move found during the 
local search. This allows an efficient caching mechanism: I track the modification time of each route and the last time 
an operator evaluated moves for that route, such that the evaluation of moves where all involved routes did not change 
since the last evaluation can be skipped. I shuffle the route evaluation order at the start of each local search to 
avoid an overly greedy local search.

The local search procedure uses a single operator: relocate.

### Starting solution

I generate a starting solution by inserting requests into an initially empty solution in random order. Each request is
inserted at the position causing the least increase of the objective function.

# Further improvements

* Benchmark against exact solver to validate the heuristic's performance. This was out-of-scope for this challenge.
* Improve performance. I did not focus on performance in this challenge, such that the heuristic is quite slow. For instance, right now, each update creates new labels rather than reusing already created ones.
* More operators. The heuristic would benefit from more neighborhoods. There exist several well-known PDP/VRP operators for the large neighborhood and local search.
* Logging. I would use proper logging in production code.
* Tests. I did not implement any tests beyond a basic test of the constant time evaluation.
* Assertions/Validation. The models do not validate their invariant in the constructor.