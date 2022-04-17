import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import re
import itertools
import networkx as nx
from matplotlib import colors
from itertools import cycle


class Customer:
    def __init__(self, number, x, y, demand, ready_time, due_date, service_time):
        self.number = number
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time
        self.isVisited = False
        self.clusterLabel = 0

    def distance(self, point):
        diff_1 = self.x - point.x
        diff_2 = self.y - point.y
        return math.sqrt(diff_1*diff_1 + diff_2*diff_2)


class Problem:
    def __init__(self, name, customers: list, vehicle_number, vehicle_capacity):
        self.name = name
        self.customers = customers
        self.vehicle_number = vehicle_number
        self.vehicle_capacity = vehicle_capacity
        self.depot: Customer = [x for x in customers if (x.number == 0)][0]
        self.depot.isVisited = True

    def getAvailableCustomers(self):
        return sorted(filter(lambda x: not x.isVisited, self.customers[1:]), key=lambda x: x.due_date)

    def getCluster(self, clusterIndex):
        return [x for x in self.customers[1:] if x.clusterLabel == clusterIndex and x.isVisited == False]

    def dist_Routes(self, routes):
        return sum(map(lambda x: x.total_distance, routes))


class Route:
    def __init__(self, problem: Problem, customers: list):
        self.problem: Problem = problem
        self._customers: list = [self.problem.depot, *customers, self.problem.depot]

    @property
    def getRouteCustomers(self):
        return " ".join(str(customer.number) for customer in self._customers)

    def for_print(self):
        time = 0
        result = [0, 0]
        for source, goal in zip(self._customers, self._customers[1:]):
            start_time = max([goal.ready_time, time + source.distance(goal)])
            time = start_time + goal.service_time
            result.append(goal.number)
            result.append(round(start_time, 5))
        return " ".join(str(x) for x in result)

    @property
    def customers(self):
        return self._customers[1:-1]

    def all_customers_with_depot(self):
        return self._customers

    @property
    def edges(self):
        #need for GLS
        return list(zip(self._customers, self._customers[1:]))

    @property
    def total_distance(self):
        return sum(a.distance(b) for (a, b) in zip(self._customers, self._customers[1:]))

    @property
    def is_possible(self):
        time = 0
        capacity = self.problem.vehicle_capacity
        possible = True
        for source, goal in zip(self._customers, self._customers[1:]):
            start_service_time = max([goal.ready_time, time + source.distance(goal)])
            if start_service_time >= goal.due_date:
                possible = False
            time = start_service_time + goal.service_time
            capacity -= goal.demand
        if time >= self.problem.depot.due_date or capacity < 0:
            possible = False
        return possible

    @property
    def numeric_edges(self):
        # need for GLS
        return [edge for edge in list(map(lambda x: (x[0].number, x[1].number), self.edges))]

    def augmented_dist_Route(self, lambda_, penalties):
        # need for GLS
        g = self.total_distance
        penalty_sum = 0
        for edge in self.numeric_edges:
            penalty_sum = penalties[edge[0]][edge[1]]
        return g + lambda_ * penalty_sum


def data_processing(fileName) -> Problem:
    with open(fileName, 'r') as f:
        lines = list(map(lambda l: l.replace('\n', '').split(), f.readlines()))
    name = fileName[0:-4]
    vehicle_number, vehicle_capacity = list(map(int, lines[4]))
    customers = []
    for line in lines[9:]:
        customers.append(Customer(*list(map(int, line))))
    return Problem(name, customers, vehicle_number, vehicle_capacity)


def plotRoutes(routes):
    for route in routes:
        X = []
        Y = []
        for customer in route.all_customers_with_depot():
            X.append(customer.x)
            Y.append(customer.y)
            #plt.text(customer.x, customer.y, customer.number)
        plt.plot(X, Y, marker='.')
    plt.show()


def plotClusters(points, labels, num_colors):
    plt.scatter(points[:, 0], points[:, 1], marker='.', c=labels, cmap=plt.cm.get_cmap("jet", num_colors))
    plt.colorbar(ticks=range(num_colors))
    #iter = 1
    #for point in x:
        #plt.text(point[0], point[1], iter)
        #iter +=1
    plt.grid()
    plt.show()


def plotPoints(problem):
    coords = np.column_stack([[c.x for c in problem.customers[:]], [c.y for c in problem.customers[:]]])
    plt.plot(coords[0,0], coords[0,1], marker='o', color='red', ls='')
    plt.plot(coords[1:,0], coords[1:,1], marker='.', color='black', ls='')
    plt.grid()
    plt.show()


def getInitialSolution(problem):
    averageVehicle = math.ceil(np.sum([x.demand for x in problem.customers]) / int(problem.vehicle_capacity))
    coords = np.column_stack([[cust.x for cust in problem.customers[1:]], [cust.y for cust in problem.customers[1:]]])

    kmeans = KMeans(n_clusters=averageVehicle, random_state=0).fit(coords)
    for i in range(len(kmeans.labels_)):
        problem.customers[i + 1].clusterLabel = kmeans.labels_[i]

    #plotClusters(coords, kmeans.labels_, averageVehicle) ############

    routes = []

    while len(problem.getAvailableCustomers()) > 0:
        route = []
        current = problem.getAvailableCustomers()[0]
        nearest = problem.getCluster(current.clusterLabel)
        if (len([x for x in nearest if x.isVisited == False]) == 1):
            route.append(current)
            problem.customers[problem.customers.index(current)].isVisited = True
        else:
            for customer in nearest:
                if Route(problem, route + [customer]).is_possible:
                    route.append(customer)
                    customer.isVisited = True
                    nearest.remove(customer)
        routes.append(Route(problem, route))
    #plotRoutes(routes)#########################
    return routes

def getInitialSolution2(problem):
    routes = []
    while len(problem.getAvailableCustomers()) > 0:
        customers = problem.getAvailableCustomers()
        route = []
        for customer in customers:
            if Route(problem, route + [customer]).is_possible:
                customer.isVisited = True
                route.append(customer)
        routes.append(Route(problem, route))
    return routes

def two_opt(c, i, j):
    if i == 0:
        return c[j:i:-1] + [c[i]] + c[j + 1:]
    return c[:i] + c[j:i - 1:-1] + c[j + 1:]


def insert(customers_1, customers_2, i, j):
    if len(customers_1) == 0:
        return customers_1, customers_2
    while i >= len(customers_1):
        i -= len(customers_1)
    return customers_1[:i] + customers_1[i + 1:], customers_2[:j] + [customers_1[i]] + customers_2[j:]


def cross(customers_1, customers_2, i, j):
    return customers_1[:i] + customers_2[j:], customers_2[:j] + customers_1[i:]


def swap(customers_1, customers_2, i, j):
    if i >= len(customers_1) or j >= len(customers_2):
        return customers_1, customers_2
    customers_1, customers_2 = customers_1.copy(), customers_2.copy()
    customers_1[i], customers_2[j] = customers_2[j], customers_1[i]
    return customers_1, customers_2


class LocalSearch:
    def __init__(self, problem: Problem):
        self.problem: Problem = problem

    def local_search(self, routes: list) -> list:
        new_routes = list(routes)
        for i in range(len(new_routes)):
            flag = False
            while not flag:
                route = new_routes[i]
                flag = True
                for k, l in itertools.combinations(range(len(route.customers)), 2):
                    new_route = Route(self.problem, two_opt(route.customers, k, l))
                    if new_route.is_possible:
                        if new_route.total_distance < route.total_distance:
                            new_routes[i] = new_route
                            flag = False
        return new_routes

    def local_search_gls(self, routes: list, lambda_, penalty) -> list:
        #need fo GLS
        new_routes = list(routes)
        for i in range(len(new_routes)):
            flag = False
            while not flag:
                route = new_routes[i]
                ag_cost = route.augmented_dist_Route(lambda_, penalty)
                flag = True
                for k, l in itertools.combinations(range(len(route.customers)), 2):
                    new_route = Route(self.problem, two_opt(route.customers, k, l))
                    new_route_augmented = new_route.augmented_dist_Route(lambda_, penalty)
                    if new_route.is_possible:
                        if new_route_augmented < ag_cost:
                            new_routes[i] = new_route
                            ag_cost = new_route.augmented_dist_Route(lambda_, penalty)
                            flag = False
        return new_routes


class IteratedLocalSearch(LocalSearch):
    def __init__(self, problem: Problem, dist_Routes=None):
        super().__init__(problem)
        if not dist_Routes:
            dist_Routes = self.problem.dist_Routes
        self.dist_Routes = dist_Routes
        self.initial_solution = getInitialSolution(problem)

    def perturbation(self, routes: list) -> list:
        best = [Route(self.problem, route.customers) for route in routes]
        flag = False
        operations = [cross, insert, swap]
        while not flag:
            flag = True
            for i, j in itertools.combinations(range(len(best)), 2):
                for k, l in itertools.product(range(len(best[i].customers) + 2), range(len(best[j].customers) + 2)):
                    for f in operations:
                        cust_1, cust_2 = f(best[i].customers, best[j].customers, k, l)
                        route_1 = Route(self.problem, cust_1)
                        route_2 = Route(self.problem, cust_2)
                        if route_1.is_possible and route_2.is_possible:
                            if route_1.total_distance + route_2.total_distance < best[i].total_distance + best[j].total_distance:
                                best[i] = route_1
                                best[j] = route_2
                                flag = False
            best = list(filter(lambda x: len(x.customers) != 0, best))
        return best

    def execute(self):
        best = self.local_search(self.initial_solution)
        flag = False
        while not flag:
            flag = True
            new_solution = self.perturbation(best)
            new_solution = self.local_search(new_solution)
            if self.dist_Routes(new_solution) < self.dist_Routes(best):
                flag = False
                best = list(filter(lambda x: len(x.customers) != 0, new_solution))
        #print("Dist: {}".format(self.dist_Routes(best)))###########
        return best


class GuidedLocalSearch(LocalSearch):
    def __init__(self, problem: Problem, lambda_=0.5):
        super().__init__(problem)
        self.lambda_ = lambda_
        self.initial_solution = getInitialSolution2(problem)
        self.penalties = [[0 for _ in self.problem.customers] for _ in self.problem.customers]


    def update_penalties(self, routes):
        utility = [[0 for _ in self.problem.customers] for _ in self.problem.customers]
        for e in [e for route in map(lambda x: x.edges, routes) for e in route]:
            utility[e[0].number][e[1].number] = (e[0].distance(e[1]) / (1 + self.penalties[e[0].number][e[1].number]))
        max_utility_value = max(max(x) for x in utility)
        for i, _ in enumerate(utility):
            for j, _ in enumerate(utility[i]):
                if utility[i][j] == max_utility_value:
                    self.penalties[i][j] = self.penalties[i][j] + 1


    def execute_(self):
        local_min = self.initial_solution
        best = None
        M = 0
        while M < 20:
            local_min = self.local_search_gls(local_min, self.lambda_, self.penalties)
            self.update_penalties(local_min)
            if best == None or self.problem.dist_Routes(local_min) < self.problem.dist_Routes(best):
                best = local_min
            M += 1
        #print("Dist: {}".format(self.problem.dist_Routes(best)))###########
        return best


def write_results(problem, routes):
    f = open(problem.name + ".sol", "w")
    for route in routes:
        f.write(route.for_print()+"\n")


def check(problem, routes):

    def ispossible(route):
        time = 0
        capacity = problem.vehicle_capacity
        possible = True
        for source, goal in zip(route._customers, route._customers[1:]):
            start_service_time = max([goal.ready_time, time + source.distance(goal)])
            if start_service_time >= goal.due_date:
                possible = False
            time = start_service_time + goal.service_time
            capacity -= goal.demand
        if time >= problem.depot.due_date or capacity < 0:
            possible = False
        return possible

    check_ = []
    for route in routes:
        check_.append(ispossible(route))
    print(check_)


def printSolution(solution):
    i=1
    for route in solution:
        print('Route', i, ':', [x.number for x in route._customers])
        i+=1


problem = data_processing("C108.txt")
#plotPoints(problem)
'''
    If you want to know the solution distance uncomment the line in ILS or GLS execute()
    IteratedLocalSearch and GuidedLocalSearch must be run separately 
    (you need to comment out one function and uncomment another)
'''
ils = IteratedLocalSearch(problem)
solution = ils.execute()
#plotRoutes(solution)
#printSolution(solution)

#gls = GuidedLocalSearch(problem)
#solution = gls.execute_()
#plotRoutes(solution)
#printSolution(solution)

#check(problem, solution)

write_results(problem, solution)

def save_pic_solution(name):
    solution = """0 0 2 73 82 115.38827 83 141.00877 95 170.93363 31 212.99607 28 229.39919 63 268.19155 19 303.27142 18 338 21 352.0 23 364.0 59 405.76476 86 427.46946 99 449.55251 84 491.12781 50 517.28331 27 550.28331 32 569.71729 89 611.73291 24 652.40863 56 685.73244 54 721.22754 0 749.25529
0 0 72 27.45906 29 131 52 206.19202 98 238.21474 14 270.35068 47 283.35068 78 305.88065 73 323.09175 79 342.03602 5 374.05874 69 418.84379 90 436.65404 10 474.54669 11 486.54669 9 501.93186 66 542.99631 60 613 1 649.92582 70 678.79379 39 716.10679 0 762.1623
0 0 12 166 75 216.11234 71 304.21484 61 336.57552 81 369 67 400.93171 85 429.95801 51 448.50201 65 486.78628 22 522.28138 77 553.09003 58 573.72018 97 597.32165 55 656.71801 68 676.71801 35 723.51475 0 777.71882
0 0 7 281 16 329.07887 87 365 94 426.0 44 470.05877 88 519.26336 8 553.03309 42 603.03309 37 623.23113 41 643.13063 96 674.34383 91 697.76024 48 745.41658 25 762.41658 13 814.84299 17 836.02333 0 886.33461
0 0 40 233 38 248.38516 30 314.74118 76 358.02781 49 423 64 451.24829 57 477.05968 15 518.82444 53 549.4157 6 584.49557 46 599.49557 4 611.49557 45 623.49557 43 668.62391 36 688.62391 93 731.86545 100 775.40647 74 833.67654 0 879.73205
0 0 3 473 92 535.20153 62 559.62374 34 589.62374 26 610.80408 20 675.88583 80 718.2741 0 735.88988
0 0 33 51.47815 0 112.9563""".split('\n')

    def to_point(line):
        names = ['number', 'x', 'y', 'demand', 'ready_time', 'due_date', 'service_time']
        line = line.replace('\n', '').split()
        return {k: int(v) for k, v in zip(names, line)}

    with open(name + '.txt', 'r') as f:
        data = f.readlines()
    points = list(map(to_point, data[9:]))

    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    for route in solution:
        G.add_path(list(map(int, route.split()[::2])))

    paths = []
    for route in solution:
        path = []
        route = list(map(int, route.split()[::2]))
        for i in range(len(route) - 1):
            path.append((route[i], route[i + 1]))
        paths.append(path)

    col = cycle(colors.TABLEAU_COLORS)
    pos = {point['number']: (point['x'], point['y']) for point in points}
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)

    for path in paths:
        path_color = next(col)
        nodes_in_path = {x for p in path for x in p}
        node_sizes = [point['demand'] for point in points if point['number'] in nodes_in_path]

        nx.draw_networkx_edges(G, pos=pos, edgelist=path, ax=ax, edge_color=path_color)
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes_in_path, node_size=node_sizes, node_color=path_color, ax=ax)
    plt.title(name)
    plt.savefig(name+".png", dpi=300)


#save_pic_solution('RC207') #need to insert the solution inside function(to copy from a file .sol)



