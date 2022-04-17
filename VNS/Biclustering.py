import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from copy import deepcopy


class CellFormationProblem:
    def __init__(self, name, machines, parts, matrix: np.ndarray):
        self.name = name
        self.machines = machines
        self.parts = parts
        self.matrix: np.ndarray = matrix
        self.all_ones = np.sum(self.matrix)

    def print_matrix(self):
        print(self.matrix)

    def plot(self):
        #'''
        '1st realization'
        d = {'20x20': [9, 6], '24x40': [10, 7], '30x50': [11, 12], '30x90': [24, 10], '37x53': [13, 15]}
        i1 = d.get(self.name)[0]
        i2 = d.get(self.name)[1]
        f, ax = plt.subplots(figsize=(i1, i2))
        sns.heatmap(self.matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
        plt.title('Problem ' + self.name)
        plt.show()
        #'''
        '''
        '2nd realisation'
        plt.imshow(self.matrix)
        plt.xticks(range(len(self.parts)))
        plt.yticks(range(len(self.machines)))
        plt.colorbar()
        plt.show()
        '''


def data_processing(fileName) -> CellFormationProblem:
    with open(fileName, 'r') as f:
        lines = list(map(lambda l: l.replace('\n', '').split(), f.readlines()))
    name = fileName[0:-4]
    machines_number, parts_number = list(map(int, lines[0]))
    matrix = np.zeros((machines_number, parts_number), dtype=int)
    for line in lines[1:]:
        machine = int(line[0]) - 1
        parts = np.array(list(map(int, line[1:]))) - 1
        for i in parts:
            matrix[machine, i] = 1
    machines = [i for i in range(machines_number)]
    parts = [i for i in range(parts_number)]
    return CellFormationProblem(name, machines, parts, matrix)


class Cluster:
    def __init__(self, machines: set, parts: set, matrix):
        self.machines = machines
        self.parts = parts
        self.matrix = matrix
        self.ones = np.sum(self.matrix[list(self.machines), :][:, list(self.parts)])
        self.zeros = len(list(self.machines))*len(list(self.parts)) - self.ones


class Solution:
    def __init__(self, problem: CellFormationProblem, clusters):
        self.problem: CellFormationProblem = problem
        self.clusters = clusters
        self.feasible = self.is_feasible
        assert self.feasible
        self.ones_in = sum([c.ones for c in self.clusters])
        self.zeros_in = sum([c.zeros for c in self.clusters])
        self.obj_func = self.objective_function

    @property
    def objective_function(self):
        return self.ones_in/(self.problem.all_ones + self.zeros_in)

    @property
    def is_feasible(self):
        feasible = True
        # Each cluster contains at least one machine and one part
        feasible &= all([len(cluster.machines) > 0 and len(cluster.parts) > 0 for cluster in self.clusters])
        # Each machine belongs to only one cluster
        feasible &= all([len(self.clusters[i].machines & self.clusters[j].machines) == 0
                      for i in range(len(self.clusters))
                      for j in range(i, len(self.clusters)) if i != j])
        # Each part belongs to only one cluster
        feasible &= all([len(self.clusters[i].parts & self.clusters[j].parts) == 0
                          for i in range(len(self.clusters))
                          for j in range(i, len(self.clusters)) if i != j])
        return feasible

    def draw(self):
        #'''
        '1st realization'
        if not os.path.exists('answers/'):
            os.mkdir('answers')
        d = {'20x20': [9, 6], '24x40': [10, 7], '30x50': [11, 12], '30x90': [24, 10], '37x53': [13, 15]}
        i1 = d.get(self.problem.name)[0]
        i2 = d.get(self.problem.name)[1]
        plt.subplots(figsize=(i1, i2))
        labels = np.zeros((self.problem.matrix.shape[0], self.problem.matrix.shape[1]), dtype=int)
        for ind, c in enumerate(self.clusters):
            for i in list(c.machines):
                for j in list(c.parts):
                    labels[i, j] = ind + 1
        colors_ = ['whitesmoke', 'red', 'green', 'yellow', 'blue', 'orange', 'salmon', 'pink', 'plum', 'crimson',
                   'lime', 'olive', 'aquamarine', 'gold', 'skyblue', 'darkolivegreen', 'lawngreen', 'navy',
                   'darkslateblue', 'lightcoral']
        cmap = colors.ListedColormap(colors_[0:len(self.clusters)+1])
        #bounds = range(0, len(self.clusters)+1)
        #norm = colors.BoundaryNorm(bounds, cmap.N)

        #plt.pcolor(labels, cmap=cmap, edgecolors='k', linewidths=0.5) #ha="left", va="bottom"
        plt.imshow(labels, interpolation='nearest', cmap=cmap)
        for i in range(len(self.problem.machines)):
            for j in range(len(self.problem.parts)):
                text = plt.text(j, i, self.problem.matrix[i, j],
                                ha="center", va="center", color="black")
        plt.title(self.problem.name)
        plt.xticks(range(len(self.problem.parts)))
        plt.yticks(range(len(self.problem.machines)))
        plt.colorbar()
        plt.tight_layout()
        #plt.show()

        plt.savefig('answers/' + self.problem.name + ".png")
        #'''
        '''
        '2nd realization'
        fig, ax = plt.subplots()
        im = ax.imshow(self.problem.matrix)
        # I want to show all ticks...
        ax.set_xticks(np.arange(len(self.problem.machines)))
        ax.set_yticks(np.arange(len(self.problem.parts)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(self.problem.machines)
        ax.set_yticklabels(self.problem.parts)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        labels = np.zeros((self.problem.matrix.shape[0], self.problem.matrix.shape[1]), dtype=int)
        for i in range(len(self.problem.machines)):
            for j in range(len(self.problem.parts)):
                labels[i, j] = -1
        for ind, c in enumerate(self.clusters):
            for i in list(c.machines):
                for j in list(c.parts):
                    labels[i, j] = ind+1

        for i in range(len(self.problem.machines)):
            for j in range(len(self.problem.parts)):
                text = ax.text(j, i, labels[i, j],
                                ha="center", va="center", color="black")
        plt.show()
        '''

    def save_solution(self):
        if not os.path.exists('answers/'):
            os.mkdir('answers')
        machines = dict()
        parts = dict()
        for i, cluster in enumerate(self.clusters):
            for part in cluster.parts:
                parts[part] = i + 1
            for machine in cluster.machines:
                machines[machine] = i + 1
        file = 'answers/' + self.problem.name + '.sol'
        with open(file, "w") as f:
            f.write(" ".join(sorted(["m{0}_{1}".format(i + 1, machines[i]) for i in machines],
                                  key=lambda x: int(x.split("_")[0].replace("m", "")))) + "\n")
            f.write(" ".join(sorted(["p{0}_{1}".format(i + 1, parts[i]) for i in parts],
                                       key=lambda x: int(x.split("_")[0].replace("p", "")))) + "\n")

    def __repr__(self):
        machines = dict()
        parts = dict()
        for i, cluster in enumerate(self.clusters):
            for part in cluster.parts:
                parts[part] = i + 1
            for machine in cluster.machines:
                machines[machine] = i + 1
        return "{}\n{}".format(" ".join(sorted(["m{0}_{1}".format(i + 1, machines[i]) for i in machines],
                          key=lambda x: int(x.split("_")[0].replace("m", "")))),
                               " ".join(sorted(["p{0}_{1}".format(i + 1, parts[i]) for i in parts],
                          key=lambda x: int(x.split("_")[0].replace("p", "")))))


class GeneralVariableNeighborhoodSearch:
    def __init__(self, problem: CellFormationProblem):
        self.problem = problem

    def initial_solution(self):
        ''' 2 random clusters '''
        clusters = []
        rand_machines = set(random.sample(range(self.problem.matrix.shape[0]), math.trunc(self.problem.matrix.shape[0]/2)))
        rand_parts = set(random.sample(range(self.problem.matrix.shape[1]), math.trunc(self.problem.matrix.shape[1]/2)))
        clusters.append(Cluster(rand_machines, rand_parts, self.problem.matrix))
        remaining_machines = set(self.problem.machines) - set(rand_machines)
        remaining_parts = set(self.problem.parts) - set(rand_parts)
        clusters.append(Cluster(remaining_machines, remaining_parts, self.problem.matrix))
        return Solution(self.problem, clusters)

    def initial_solution_(self):
        ''' 1 cluster - whole matrix '''
        clusters = []
        clusters.append(Cluster(set(self.problem.machines), set(self.problem.parts), self.problem.matrix))
        return Solution(self.problem, clusters)

    def merge_clusters(self, solution: Solution):
        if len(solution.clusters) == 1:
            return solution
        num_cluster1, num_cluster2 = random.sample(range(len(solution.clusters)), 2)
        machines_in_new_cluster = list(solution.clusters[num_cluster1].machines) + list(solution.clusters[num_cluster2].machines)
        parts_in_new_cluster = list(solution.clusters[num_cluster1].parts) + list(solution.clusters[num_cluster2].parts)
        new_cluster = Cluster(set(machines_in_new_cluster), set(parts_in_new_cluster), self.problem.matrix)
        new_clusters = deepcopy(solution.clusters)
        if num_cluster1 < num_cluster2:
            new_clusters.pop(num_cluster2)
            new_clusters.pop(num_cluster1)
        else:
            new_clusters.pop(num_cluster1)
            new_clusters.pop(num_cluster2)
        new_clusters.append(new_cluster)
        result = Solution(self.problem, new_clusters)
        return result

    def split_cluster(self, solution: Solution):
        num_cluster_to_split = random.choice(range(len(solution.clusters)))
        cluster_to_split = solution.clusters[num_cluster_to_split]
        if len(cluster_to_split.machines) == 1 or len(cluster_to_split.parts) == 1:
            return solution

        rand_num_machines = random.randint(1, len(cluster_to_split.machines) - 1)
        rand_num_parts = random.randint(1, len(cluster_to_split.parts) - 1)
        machines_in_cluster1 = list(cluster_to_split.machines)[:rand_num_machines]
        parts_in_cluster1 = list(cluster_to_split.parts)[:rand_num_parts]
        cluster1 = Cluster(set(machines_in_cluster1), set(parts_in_cluster1), self.problem.matrix)
        machines_in_cluster2 = cluster_to_split.machines - set(machines_in_cluster1)
        parts_in_cluster2 = cluster_to_split.parts - set(parts_in_cluster1)
        cluster2 = Cluster(set(machines_in_cluster2), set(parts_in_cluster2), self.problem.matrix)

        new_clusters = deepcopy(solution.clusters)
        new_clusters.pop(num_cluster_to_split)
        new_clusters.extend([cluster1, cluster2])
        result = Solution(self.problem, new_clusters)
        return result

    def move_machine(self, solution: Solution):

        def move(clusters, num_cluster_from, num_cluster_to):
            if num_cluster_from < num_cluster_to:
                cluster_to = clusters.pop(num_cluster_to)
                cluster_from = clusters.pop(num_cluster_from)
            else:
                cluster_from = clusters.pop(num_cluster_from)
                cluster_to = clusters.pop(num_cluster_to)
            result = []
            if len(cluster_from.machines) == 1:
                return result
            for machine in cluster_from.machines:
                list_new_clusters = deepcopy(clusters)
                list_new_clusters.append(Cluster(cluster_from.machines - {machine}, cluster_from.parts, self.problem.matrix))
                list_new_clusters.append(Cluster(cluster_to.machines | {machine}, cluster_to.parts, self.problem.matrix))
                result.append(Solution(self.problem, list_new_clusters))
            return result

        clusters_ = deepcopy(solution.clusters)
        solutions = [solution]
        num_pairs_of_clusters = [(i, j) for i in range(len(clusters_)) for j in range(i, len(clusters_)) if i != j]
        for num_cluster1, num_cluster2 in num_pairs_of_clusters:
            solutions.extend(move(deepcopy(solution.clusters), num_cluster1, num_cluster2))
        return max(solutions, key=lambda x: x.obj_func)

    def move_part(self, solution: Solution):

        def move(clusters, num_cluster_from, num_cluster_to):
            if num_cluster_from < num_cluster_to:
                cluster_to = clusters.pop(num_cluster_to)
                cluster_from = clusters.pop(num_cluster_from)
            else:
                cluster_from = clusters.pop(num_cluster_from)
                cluster_to = clusters.pop(num_cluster_to)
            result = []
            if len(cluster_from.parts) == 1:
                return result
            for part in cluster_from.parts:
                list_new_clusters = deepcopy(clusters)
                list_new_clusters.append(Cluster(cluster_from.machines, cluster_from.parts - {part}, self.problem.matrix))
                list_new_clusters.append(Cluster(cluster_to.machines, cluster_to.parts | {part}, self.problem.matrix))
                result.append(Solution(self.problem, list_new_clusters))
            return result

        clusters_ = deepcopy(solution.clusters)
        solutions = [solution]
        num_pairs_of_clusters = [(i, j) for i in range(len(clusters_)) for j in range(i, len(clusters_)) if i != j]
        for num_cluster1, num_cluster2 in num_pairs_of_clusters:
            solutions.extend(move(deepcopy(solution.clusters), num_cluster1, num_cluster2))
        return max(solutions, key=lambda x: x.obj_func)

    def solve(self):

        def local_search_by_vnd(solution):
            vnd_functions = [self.move_part, self.move_machine]
            l = 0
            while l < len(vnd_functions):
                opt_new_solution = vnd_functions[l](deepcopy(solution))
                l += 1
                if opt_new_solution.obj_func > solution.obj_func:
                    solution = opt_new_solution
                    l = 0
            return solution

        init_sol = [self.initial_solution(), self.initial_solution_()]
        dict_sol = {'20x20': 0, '24x40': 0, '30x50': 1, '30x90': 0, '37x53': 0}
        ind = dict_sol.get(self.problem.name)
        best_solution = init_sol[ind]
        shaking_functions = [self.split_cluster, self.merge_clusters]
        iteration = 0
        while iteration < 11:
            iteration += 1
            k = 0
            while k < len(shaking_functions):
                new_solution = shaking_functions[k](deepcopy(best_solution))
                new_solution = local_search_by_vnd(new_solution)
                k += 1
                if new_solution.obj_func > best_solution.obj_func:
                    best_solution = new_solution
                    k = 0
                    iteration = 0
                    #print(best_solution, '\n', best_solution.obj_func)####
        return best_solution


problem = data_processing("20x20.txt")
#problem.print_matrix()
#problem.plot()
sol = GeneralVariableNeighborhoodSearch(problem).solve()
#print(sol, '\n', sol.obj_func, '\n', sol.feasible)
sol.draw()
sol.save_solution()

