# CVRPTW
## Implement algorithms listed below for Capacitated Vehicle Routeing Problem with Time Windows:
 - Iterated Local Search
 - Guided Local Search

Datasets: Solomon instances + Homberger instances
C108
C203
R202
RC105
RC207
R146
R168
C249
C266
RC148
BONUS (1000 customers):
R1103
R1104
R1105
R1107
R11010
R - Random
C - Clustered
RC - Random clustered


Input file: instancename.txt (e.g. R101.txt, C108.txt)
Input format:
R101
VEHICLE
NUMBER CAPACITY
25 200
CUSTOMER
CUST NO. XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME
0 35 35 0 0 230 0
1 41 49 10 161 171 10
2 35 17 7 50 60 10
< TO BE CONTINUED ...>
Customer with ID = 0 is a DEPOT.
See the attachment for some additional examples.
IMPORTANT: We assume that distance between customers/depot is Euclidian.
IMPORTANT: We assume that travel time/cost between customers/depot = distance.


Feasible solutions.
Each solution is a set of routes.
- Each customer is served
- Each route begins and ends at the DEPOT
- Each vehicle is loaded <= CAPACITY
- Number of vehicles used (total number of routes) is <= VEHICLE NUMBER
- Each vehicle starts its unloading at any customer in time T where
READY TIME (TIME WINDOW BEGINS) <= T < DUE DATE(TIME WINDOW ENDS)
- Each vehicle leaves depot and arrives depot in time K where
READY TIME DEPOT<= K < DUE DATE DEPOT
- Total cost/distance of all routes is minimized


Output file: instancename.sol (e.g. C108.sol, C203.sol)
Output file contains list of routes.
Each row in a file is a single route (With Depot customers!).
Each row consists of customerId and startTime where start time is the time we start serving this
customer (this time may be different from arrive time!).
Output file format:
depotCustomer leaveDepotTime1 route1customer1 startTimeRoute1customer1 route1customer2
startTimeroute1customer2 ... route1customerK1 startTimeroute1customerK1 depotCustomer
arriveDepotTime1
depotCustomer leaveDepotTime2 route2customer1 startTimeRoute2customer1 route2customer2
startTimeroute2customer2 ... route1customerK2 startTimeroute2customerK2 depotCustomer
arriveDepotTime2
depotCustomer leaveDepotTime3 route3customer1 startTimeRoute3customer1 route3customer2
startTimeroute3customer2 ... route3customerK3 startTimeroute3customerK3 depotCustomer
arriveDepotTime3
...
depotCustomer leaveDepotTimeS routeScustomer1 startTimeRouteScustomer1 routeScustomer2
startTimerouteScustomer2 ... routeScustomerKS startTimeRouteScustomerKS depotCustomer
arriveDepotTimeS
EXAMPLE
0 0 13 30.80584 15 384 16 479 14 571 12 664 0 792.0789
0 0 32 31.62278 33 123.6228 31 219.0079 35 314.0079 37 409.8389 38 501.8389 39 596.8389 36
691.8389 34 784.8389 0 907.2272
Here we have 2 routes. First route starts at the depot and vehicle leaves depot at time 0.
Then we arrive to customer 13 and start serving customer 13 at time 30.80584.
Then we arrive to customer 15 and start serving customer 13 at time 384.
Then we arrive to customer 16 and start serving customer 13 at time 479.
...
