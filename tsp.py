import heapq
from collections import defaultdict, Counter

# For simulated annealing
import math
import random

class MetricUndirectedGraph:
    def __init__(self):
        # Initialize adjacency list
        self.graph = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def get_weight(self, node1, node2):
        # Return the weight of edge (node1, node2) if it exists
        for neighbor, weight in self.graph.get(node1, []):
            if neighbor == node2:
                return weight
        return None  # No direct edge exists

    def add_edge(self, node1, node2, weight):
        # Ensure positive weight
        if weight < 0:
            raise ValueError("Edge weight must be non-negative.")

        self.add_node(node1)
        self.add_node(node2)

        # Check triangle inequality for existing paths
        for neighbor, w1 in self.graph[node1]:
            w2 = self.get_weight(neighbor, node2)
            if w2 is not None:
                if weight > w1 + w2:
                    raise ValueError(f"Triangle inequality violated: {node1}-{neighbor}-{node2}")

        for neighbor, w2 in self.graph[node2]:
            w1 = self.get_weight(neighbor, node1)
            if w1 is not None:
                if weight > w1 + w2:
                    raise ValueError(f"Triangle inequality violated: {node2}-{neighbor}-{node1}")

        # Add the edge (symmetric)
        self.graph[node1].append((node2, weight))
        self.graph[node2].append((node1, weight))

    def display(self):
        for node in self.graph:
            print(f"{node}: {self.graph[node]}")


# Function for the nearst-neighbor heuristic
def nearest_neighbor_tsp(graph, start):
    visited = set()
    path = [start]
    total_cost = 0
    current = start
    visited.add(current)

    while len(visited) < len(graph.graph):
        neighbors = [(neighbor, weight) for neighbor, weight in graph.graph[current] if neighbor not in visited]
        if not neighbors:
            break  # Dead end (shouldn't happen in a complete TSP instance)

        # Select nearest unvisited neighbor
        next_city, min_weight = min(neighbors, key=lambda x: x[1])
        path.append(next_city)
        total_cost += min_weight
        visited.add(next_city)
        current = next_city

    # Return to start city to complete the tour
    return_to_start_weight = graph.get_weight(current, start)
    if return_to_start_weight is not None:
        total_cost += return_to_start_weight
        path.append(start)
    else:
        raise ValueError(f"No path from {current} to {start} to complete the tour.")

    return path, total_cost

# To compute MST using prim's algorithm
def prim_mst(graph):
    start = next(iter(graph.graph))
    visited = set([start])
    edges = [(weight, start, neighbor) for neighbor, weight in graph.graph[start]]
    heapq.heapify(edges)
    mst = defaultdict(list)

    while edges:
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst[u].append((v, weight))
            mst[v].append((u, weight))
            for neighbor, w in graph.graph[v]:
                if neighbor not in visited:
                    heapq.heappush(edges, (w, v, neighbor))
    return mst

def find_odd_degree_vertices(mst):
    degree = Counter()
    for u in mst:
        for v, _ in mst[u]:
            degree[u] += 1
    return [node for node, deg in degree.items() if deg % 2 == 1]

def minimum_weight_matching(graph, odd_vertices):
    matched = set()
    matching = []
    while odd_vertices:
        u = odd_vertices.pop()
        min_edge = None
        min_weight = float('inf')
        for v in odd_vertices:
            w = graph.get_weight(u, v)
            if w is not None and w < min_weight:
                min_edge = v
                min_weight = w
        if min_edge:
            odd_vertices.remove(min_edge)
            matching.append((u, min_edge, min_weight))
            matched.add(u)
            matched.add(min_edge)
    return matching

def multigraph_union(mst, matching):
    multigraph = defaultdict(list)
    for u in mst:
        for v, w in mst[u]:
            multigraph[u].append((v, w))
    for u, v, w in matching:
        multigraph[u].append((v, w))
        multigraph[v].append((u, w))
    return multigraph

def eulerian_tour(multigraph, start):
    tour = []
    stack = [start]
    local_graph = {u: list(neighbors) for u, neighbors in multigraph.items()}
    while stack:
        u = stack[-1]
        if local_graph[u]:
            v, _ = local_graph[u].pop()
            for i, (n, _) in enumerate(local_graph[v]):
                if n == u:
                    del local_graph[v][i]
                    break
            stack.append(v)
        else:
            tour.append(stack.pop())
    return tour

def shortcut_tour(tour):
    seen = set()
    final_path = []
    for city in tour:
        if city not in seen:
            final_path.append(city)
            seen.add(city)
    final_path.append(final_path[0])  # close the tour
    return final_path

def christofides_tsp(graph, start=None):
    mst = prim_mst(graph)
    odd_vertices = find_odd_degree_vertices(mst)
    matching = minimum_weight_matching(graph, odd_vertices)
    multigraph = multigraph_union(mst, matching)
    start = next(iter(graph.graph))
    tour = eulerian_tour(multigraph, start)
    final_tour = shortcut_tour(tour)

    # Compute cost
    total_cost = 0
    for i in range(len(final_tour) - 1):
        total_cost += graph.get_weight(final_tour[i], final_tour[i + 1])

    return final_tour, total_cost

# local search via 2-opt move
def simulated_annealing_tsp(graph, initial_solution_fn, start, initial_temp=1000, cooling_rate=0.995, num_iters=10000):
    # Generate initial tour and cost
    current_tour, current_cost = initial_solution_fn(graph, start)
    best_tour, best_cost = current_tour[:], current_cost

    temperature = initial_temp

    def total_tour_cost(tour):
        cost = 0
        for i in range(len(tour) - 1):
            cost += graph.get_weight(tour[i], tour[i+1])
        return cost

    def swap_2opt(tour):
        # Randomly pick two edges and swap the path between them (2-opt move)
        a, b = sorted(random.sample(range(1, len(tour) - 1), 2))
        new_tour = tour[:a] + tour[a:b+1][::-1] + tour[b+1:]
        return new_tour

    for iteration in range(num_iters):
        # Create new tour via 2-opt swap
        new_tour = swap_2opt(current_tour)
        new_cost = total_tour_cost(new_tour)

        # Decide whether to accept new solution
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_tour, current_cost = new_tour, new_cost
            # Update best solution if improved
            if current_cost < best_cost:
                best_tour, best_cost = current_tour, current_cost

        # Cool down
        temperature *= cooling_rate

        # Optional: stop if temperature is very low
        if temperature < 1e-8:
            break

    return best_tour, best_cost




g = MetricUndirectedGraph()

# Example:
g.add_edge("A", "B", 4)
g.add_edge("A", "C", 2)
g.add_edge("A", "D", 7)
g.add_edge("B", "C", 5)
g.add_edge("B", "D", 3)
g.add_edge("C", "D", 6)
# g.display()

# pure nearest neighbor ->
path1, cost1 = nearest_neighbor_tsp(g, "A")
print("\nPure Nearest-neighbor TSP Path:", path1)
print("Total Cost:", cost1)

#pure christofides ->
path2, cost2 = christofides_tsp(g)
print("\nPure Christofides TSP Path:", path2)
print("Total Cost:", cost2)

# using SA
path, cost = simulated_annealing_tsp(g, nearest_neighbor_tsp, "A")
print("\nSA (Nearest Neighbor) Path:", path)
print("Total Cost:", cost)

path, cost = simulated_annealing_tsp(g, christofides_tsp, "A")
print("\nSA (Christofides) Path:", path)
print("Total Cost:", cost)