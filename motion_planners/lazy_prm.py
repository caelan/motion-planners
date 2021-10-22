from heapq import heappush, heappop
from collections import namedtuple, defaultdict

from .nearest import NearestNeighbors
from .utils import INF, elapsed_time, get_pairs, default_selector, refine_waypoints, irange, \
    merge_dicts, compute_path_cost, get_length, is_path

import time
import numpy as np

Metric = namedtuple('Metric', ['p_norm', 'weights'])
Node = namedtuple('Node', ['g', 'parent'])
unit_cost_fn = lambda v1, v2: 1.
zero_heuristic_fn = lambda v: 0

ORDINAL = 1e3

def retrace_path(visited, vertex):
    if vertex is None:
        return []
    return retrace_path(visited, visited[vertex].parent) + [vertex]

def dijkstra(start_v, neighbors_fn, cost_fn=unit_cost_fn):
    # Update the heuristic over time
    # TODO: overlap with discrete
    # TODO: all pairs shortest paths
    start_g = 0.
    visited = {start_v: Node(start_g, None)}
    queue = [(start_g, start_v)]
    while queue:
        current_g, current_v = heappop(queue)
        if visited[current_v].g < current_g:
            continue
        for next_v in neighbors_fn(current_v):
            next_g = current_g + cost_fn(current_v, next_v)
            if (next_v not in visited) or (next_g < visited[next_v].g):
                visited[next_v] = Node(next_g, current_v)
                heappush(queue, (next_g, next_v))
    return visited

def get_priority_fn(w=1.):
    assert 0. <= w <= INF
    def priority_fn(g, h):
        if w == 0.:
            return g
        if w == INF:
            return (h, g)
        return g + w*h
    return priority_fn

def wastar_search(start_v, end_v, neighbors_fn, cost_fn=unit_cost_fn,
                  heuristic_fn=zero_heuristic_fn, max_cost=INF, max_time=INF, **kwargs):
    # TODO: lazy wastar to get different paths
    # TODO: multi-start / multi-goal
    #heuristic_fn = lambda v: cost_fn(v, end_v)
    priority_fn = get_priority_fn(**kwargs)
    goal_test = lambda v: v == end_v

    start_time = time.time()
    start_g = 0.
    start_h = heuristic_fn(start_v)
    visited = {start_v: Node(start_g, None)}
    queue = [(priority_fn(start_g, start_h), start_g, start_v)]
    while queue and (elapsed_time(start_time) < max_time):
        _, current_g, current_v = heappop(queue)
        if visited[current_v].g < current_g:
            continue
        if goal_test(current_v):
            return retrace_path(visited, current_v)
        for next_v in neighbors_fn(current_v):
            next_g = current_g + cost_fn(current_v, next_v)
            if (next_v not in visited) or (next_g < visited[next_v].g):
                visited[next_v] = Node(next_g, current_v)
                next_h = heuristic_fn(next_v)
                if (next_g + next_h) < max_cost: # Assumes admissible
                    next_p = priority_fn(next_g, next_h)
                    heappush(queue, (next_p, next_g, next_v))
    return None

##################################################

def get_embed_fn(weights):
    return lambda q: weights * q

def get_distance_fn(weights, p_norm=2):
    embed_fn = get_embed_fn(weights)
    return lambda q1, q2: np.linalg.norm(embed_fn(q2) - embed_fn(q1), ord=p_norm)

##################################################

class Roadmap(object):
    def __init__(self, weights, samples=[], p_norm=2, max_degree=6, max_distance=INF, approximate_eps=0., **kwargs):
        self.weights = tuple(weights)
        self.p_norm = p_norm
        self.max_degree = max_degree
        self.max_distance = max_distance
        self.approximate_eps = approximate_eps
        self.nearest = NearestNeighbors(embed_fn=get_embed_fn(self.weights), **kwargs)
        self.edges = set()
        self.outgoing_from_edges = defaultdict(set)
        self.colliding_vertices = {}
        self.colliding_edges = {}
        self.add_samples(samples)
    @property
    def samples(self):
        return self.nearest.data
    @property
    def vertices(self):
        return list(range(len(self.samples))) # TODO: inefficient
    def __iter__(self):
        return iter([self.samples, self.vertices, self.edges])
    def add_samples(self, samples):
        edges = set()
        for v1, sample in self.nearest.add_data(samples):
            # TODO: could dynamically compute distances
            max_degree = min(self.max_degree, len(self.samples))
            for d, v2, _ in self.nearest.query_neighbors(sample, k=max_degree, eps=self.approximate_eps,
                                                         p=self.p_norm, distance_upper_bound=self.max_distance):
                if (v1 != v2): # and (d <= self.max_distance):
                    self.outgoing_from_edges[v1].add(v2)
                    self.outgoing_from_edges[v2].add(v1)
                    edges.update([(v1, v2), (v2, v1)])
        self.edges.update(edges)
        return edges
    def neighbors_fn(self, v1):
        for v2 in self.outgoing_from_edges[v1]:
            if not self.colliding_vertices.get(v2, False) and not self.colliding_edges.get((v1, v2), False):
                yield v2
    def check_vertex(self, v, collision_fn):
        x = self.samples[v]
        colliding_vertices = self.colliding_vertices
        if v not in colliding_vertices:
            # TODO: could update the colliding adjacent edges as well
            colliding_vertices[v] = collision_fn(x)
        return not colliding_vertices[v]
    def check_edge(self, v1, v2, collision_fn, extend_fn):
        colliding_edges = self.colliding_edges
        if (v1, v2) not in colliding_edges:
            x1 = self.samples[v1]
            x2 = self.samples[v2]
            segment = default_selector(extend_fn(x1, x2))
            colliding_edges[v1, v2] = any(map(collision_fn, segment))
            colliding_edges[v2, v1] = colliding_edges[v1, v2]
        return not colliding_edges[v1, v2]
    def check_path(self, path, extend_fn, collision_fn):
        for v in default_selector(path):
            if not self.check_vertex(v, collision_fn):
                return False
        for v1, v2 in default_selector(get_pairs(path)):
            if not self.check_edge(v1, v2, collision_fn, extend_fn):
                return False
        return True

##################################################

def sample_roadmap(start, goal, sample_fn, distance_fn=None, num_samples=100,
                   weights=None, max_cost=INF, max_time=INF, **kwargs):
    start_time = time.time()
    samples = [start, goal]
    # TODO: compute number of rejected samples
    while (len(samples) < num_samples) and (elapsed_time(start_time) < max_time):
        conf = sample_fn() # TODO: include
        # TODO: bound function based on distance_fn(start, conf) and individual distances
        #if (max_cost == INF) or (distance_fn(start, conf) + distance_fn(conf, goal)) < max_cost:
        # TODO: only keep edges that move toward the goal
        samples.append(conf)
    if weights is None:
        weights = np.ones(len(samples[0]))
    return Roadmap(weights, samples=samples, leafsize=10, compact_nodes=True,
                   copy_data=False, balanced_tree=True, boxsize=None, **kwargs)

def calculate_radius(d=2):
    # TODO: unify with get_threshold_fn
    # Sampling-based Algorithms for Optimal Motion Planning
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.419.5503&rep=rep1&type=pdf
    # https://en.wikipedia.org/wiki/Volume_of_an_n-ball
    interval = (1 - 0)
    vol_free = interval ** d
    radius = 1./2
    vol_ball = np.pi * (radius ** d)
    gamma = 2 * ((1 + 1. / d) * (vol_free / vol_ball)) ** (1. / d)
    # threshold = gamma * (math.log(n) / n) ** (1. / d)
    return gamma

##################################################

def lazy_prm(start, goal, sample_fn, extend_fn, collision_fn, cost_fn=None, num_samples=100,
             weights=None, p_norm=2, lazy=True, max_cost=INF, max_time=INF, w=1., verbose=True, **kwargs): #, max_paths=INF):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_time: Maximum runtime - float
    :param kwargs: Keyword arguments
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: compute hyperparameters using start, goal, and sample_fn statistics
    # TODO: scale default parameters based on
    # TODO: precompute and store roadmap offline
    # TODO: multiple collision functions to allow partial reuse
    # TODO: multi-query motion planning
    start_time = time.time()
    # TODO: can embed pose and/or points on the robot for other distances
    d = len(start)
    if weights is None:
        weights = np.ones(d)
    distance_fn = get_distance_fn(weights, p_norm=p_norm)
    # TODO: can compute cost between waypoints from extend_fn

    roadmap = sample_roadmap(start, goal, sample_fn, distance_fn, num_samples=num_samples,
                             p_norm=p_norm, max_cost=INF, **kwargs) # max_cost=max_cost,
    samples, vertices, edges = roadmap
    neighbors_from_index = roadmap.outgoing_from_edges
    start_index, end_index = 0, 1
    degree = np.average(list(map(len, neighbors_from_index.values())))

    # TODO: update collision occupancy based on proximity to existing colliding (for diversity as well)
    # TODO: minimize the maximum distance to colliding

    if not lazy:
        for vertex in vertices:
            roadmap.check_vertex(vertex, collision_fn)
        for vertex1, vertex2 in edges:
            roadmap.check_edge(vertex1, vertex2, collision_fn, extend_fn)

    if cost_fn is None:
        cost_fn = distance_fn # TODO: additive cost, acceleration cost

    weight_cache = {}
    def weight_fn(v1, v2):
        if (v1, v2) not in weight_cache:
            weight_cache[v1, v2] = weight_cache[v2, v1] = cost_fn(samples[v1], samples[v2])
        return weight_cache[v1, v2]

    #weight_fn = lambda v1, v2: cost_fn(samples[v1], samples[v2])
    #lazy_fn = lambda v1, v2: (v2 not in colliding_vertices)
    #lazy_fn = lambda v1, v2: ((v1, v2) not in colliding_edges) # TODO: score by length
    #weight_fn = lazy_fn
    #weight_fn = lambda v1, v2: (lazy_fn(v1, v2), cost_fn(samples[v1], samples[v2])) # TODO:
    #weight_fn = lambda v1, v2: ORDINAL*lazy_fn(v1, v2) + cost_fn(samples[v1], samples[v2])
    #w = 0

    visited = dijkstra(end_index, roadmap.neighbors_fn, weight_fn)
    heuristic_fn = lambda v: visited[v].g if (v in visited) else INF
    #heuristic_fn = zero_heuristic_fn
    #heuristic_fn = lambda v: weight_fn(v, end_index)
    path = None
    while (elapsed_time(start_time) < max_time) and (path is None): # TODO: max_attempts
        # TODO: extra cost to prioritize reusing checked edges
        lazy_path = wastar_search(start_index, end_index, neighbors_fn=roadmap.neighbors_fn,
                                  cost_fn=weight_fn, heuristic_fn=heuristic_fn,
                                  max_cost=max_cost, max_time=max_time-elapsed_time(start_time), w=w)
        if lazy_path is None:
            break
        cost = sum(weight_fn(v1, v2) for v1, v2 in get_pairs(lazy_path))
        if verbose:
            print('Length: {} | Cost: {:.3f} | Vertices: {} | Edges: {} | Degree: {:.3f} | Time: {:.3f}'.format(
                len(lazy_path), cost, len(roadmap.colliding_vertices), len(roadmap.colliding_edges),
                degree, elapsed_time(start_time)))
        if roadmap.check_path(lazy_path, extend_fn, collision_fn):
            path = lazy_path

    if path is None:
        return path, samples, edges, roadmap.colliding_vertices, roadmap.colliding_edges
    waypoints = [samples[v] for v in path]
    solution = [start] + refine_waypoints(waypoints, extend_fn)
    return solution, samples, edges, roadmap.colliding_vertices, roadmap.colliding_edges

##################################################

def create_param_sequence(initial_samples=100, step_samples=100, **kwargs):
    # TODO: iteratively increase the parameters
    # TODO: generalize to degree, distance, cost
    return (merge_dicts(kwargs, {'num_samples': num_samples})
            for num_samples in irange(start=initial_samples, stop=INF, step=step_samples))

def lazy_prm_star(start, conf, sample_fn, extend_fn, collision_fn, cost_fn=None, max_cost=INF, success_cost=0,
                  param_sequence=None, weights=None, p_norm=2, max_time=INF, verbose=True, **kwargs):
    # TODO: bias to stay near the (past/hypothetical) path
    # TODO: proximity pessimistic collision checking
    # TODO: roadmap reuse in general
    start_time = time.time()
    if cost_fn is None:
        if weights is None:
            d = len(start)
            weights = np.ones(d)
        distance_fn = get_distance_fn(weights, p_norm=p_norm)
        cost_fn = distance_fn # TODO: additive cost, acceleration cost
    if param_sequence is None:
        param_sequence = create_param_sequence()
    best_path = None
    best_cost = max_cost
    for i, params in enumerate(param_sequence):
        remaining_time = max_time - elapsed_time(start_time)
        if remaining_time <= 0.:
            break
        if verbose:
            print('\nIteration: {} | Cost: {:.3f} | Elapsed: {:.3f} | Remaining: {:.3f} | Params: {}'.format(
                i, best_cost, elapsed_time(start_time), remaining_time, params))
        new_path = lazy_prm(start, conf, sample_fn, extend_fn, collision_fn,
                                cost_fn=cost_fn, weights=weights, p_norm=p_norm,
                                max_time=remaining_time, max_cost=best_cost,
                                verbose=verbose, **params, **kwargs)[0]
        new_cost = compute_path_cost(new_path, cost_fn=cost_fn)
        if verbose:
            print(is_path(new_path), new_cost, get_length(new_path))
        if new_cost < best_cost:
            best_path = new_path
            best_cost = new_cost
        if best_cost < success_cost:
            break
    return best_path
