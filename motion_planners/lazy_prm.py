from scipy.spatial.kdtree import KDTree
from heapq import heappush, heappop
from collections import namedtuple

from .utils import INF, elapsed_time, get_pairs, random_selector, default_selector
from .smoothing import smooth_path

import time
import numpy as np

Metric = namedtuple('Metric', ['p_norm', 'weights'])
Node = namedtuple('Node', ['g', 'parent'])
unit_cost_fn = lambda v1, v2: 1.
zero_heuristic_fn = lambda v: 0

def retrace_path(visited, vertex):
    if vertex is None:
        return []
    return retrace_path(visited, visited[vertex].parent) + [vertex]

def dijkstra(start_v, neighbors_fn, cost_fn=unit_cost_fn):
    # Update the heuristic over time
    # TODO: overlap with discrete
    start_g = 0
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

def wastar_search(start_v, end_v, neighbors_fn, cost_fn=unit_cost_fn,
                  heuristic_fn=zero_heuristic_fn, w=1., max_cost=INF, max_time=INF):
    # TODO: lazy wastar to get different paths
    #heuristic_fn = lambda v: cost_fn(v, end_v)
    priority_fn = lambda g, h: g + w*h
    goal_test = lambda v: v == end_v

    start_time = time.time()
    start_g = 0
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
                if priority_fn(next_g, next_h) < max_cost:
                    heappush(queue, (priority_fn(next_g, next_h), next_g, next_v))
    return None

##################################################

def get_embed_fn(weights):
    return lambda q: weights * q

def get_distance_fn(weights, p_norm=2):
    embed_fn = get_embed_fn(weights)
    return lambda q1, q2: np.linalg.norm(embed_fn(q2) - embed_fn(q1), ord=p_norm)

##################################################

def check_vertex(v, samples, colliding_vertices, collision_fn):
    if v not in colliding_vertices:
        colliding_vertices[v] = collision_fn(samples[v])
    return not colliding_vertices[v]

def check_edge(v1, v2, samples, colliding_edges, collision_fn, extend_fn):
    if (v1, v2) not in colliding_edges:
        segment = default_selector(extend_fn(samples[v1], samples[v2]))
        colliding_edges[v1, v2] = any(map(collision_fn, segment))
        colliding_edges[v2, v1] = colliding_edges[v1, v2]
    return not colliding_edges[v1, v2]

def check_path(path, colliding_vertices, colliding_edges, samples, extend_fn, collision_fn):
    for v in random_selector(path):
        if not check_vertex(v, samples, colliding_vertices, collision_fn):
            return False
    for v1, v2 in default_selector(get_pairs(path)):
        if not check_edge(v1, v2, samples, colliding_edges, collision_fn, extend_fn):
            return False
    return True

##################################################

def compute_graph(samples, weights=None, p_norm=2, max_degree=10, max_distance=INF, approximate_eps=0.):
    vertices = list(range(len(samples)))
    edges = set()
    if not vertices:
        return vertices, edges
    if weights is None:
        weights = np.ones(len(samples[0]))
    embed_fn = get_embed_fn(weights)
    embedded = list(map(embed_fn, samples))
    kd_tree = KDTree(embedded)
    for v1 in vertices:
        # TODO: could dynamically compute distances
        distances, neighbors = kd_tree.query(embedded[v1], k=max_degree + 1, eps=approximate_eps,
                                             p=p_norm, distance_upper_bound=max_distance)
        for d, v2 in zip(distances, neighbors):
            if (d < max_distance) and (v1 != v2):
                edges.update([(v1, v2), (v2, v1)])
    # print(time.time() - start_time, len(edges), float(len(edges))/len(samples))
    return vertices, edges

##################################################

def lazy_prm(start, goal, sample_fn, extend_fn, collision_fn, num_samples=100,
             weights=None, p_norm=2, lazy=False, max_cost=INF, max_time=INF, **kwargs): #, max_paths=INF):
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
    # TODO: compute parameters using start, goal, and sample_fn statistics
    # TODO: multi-query motion planning
    start_time = time.time()
    # TODO: can embed pose and/or points on the robot for other distances
    if weights is None:
        weights = np.ones(len(start))
    distance_fn = get_distance_fn(weights, p_norm=p_norm)
    # TODO: can compute cost between waypoints from extend_fn

    samples = []
    while len(samples) < num_samples:
        conf = sample_fn()
        if (distance_fn(start, conf) + distance_fn(conf, goal)) < max_cost:
            samples.append(conf)
    start_index, end_index = 0, 1
    samples[start_index] = start
    samples[end_index] = goal
    cost_fn = lambda v1, v2: distance_fn(samples[v1], samples[v2])

    vertices, edges = compute_graph(samples, p_norm=p_norm, **kwargs)
    neighbors_from_index = {v: set() for v in vertices}
    for v1, v2 in edges:
        neighbors_from_index[v1].add(v2)

    colliding_vertices, colliding_edges = {}, {}
    def neighbors_fn(v1):
        for v2 in neighbors_from_index[v1]:
            if not colliding_vertices.get(v2, False) and not colliding_edges.get((v1, v2), False):
                yield v2

    if not lazy:
        for vertex in vertices:
            check_vertex(vertex, samples, colliding_vertices, collision_fn)
        for vertex1, vertex2 in edges:
            check_edge(vertex1, vertex2, samples, colliding_edges, collision_fn, extend_fn)

    visited = dijkstra(end_index, neighbors_fn, cost_fn)
    heuristic_fn = lambda v: visited[v].g if v in visited else INF
    path = None
    while (elapsed_time(start_time) < max_time) and (path is None): # TODO: max_attempts
        # TODO: extra cost to prioritize reusing checked edges
        lazy_path = wastar_search(start_index, end_index, neighbors_fn=neighbors_fn,
                                  cost_fn=cost_fn, heuristic_fn=heuristic_fn,
                                  max_cost=max_cost, max_time=max_time-elapsed_time(start_time))
        if lazy_path is None:
            break
        cost = sum(cost_fn(v1, v2) for v1, v2 in get_pairs(lazy_path))
        print('Length: {} | Cost: {:.3f} | Vertices: {} | Edges: {} | Time: {:.3f}'.format(
            len(lazy_path), cost, len(colliding_vertices), len(colliding_edges), elapsed_time(start_time)))
        if check_path(lazy_path, colliding_vertices, colliding_edges, samples, extend_fn, collision_fn):
            path = lazy_path

    if path is None:
        return path, edges, colliding_vertices, colliding_edges
    solution = [start]
    for q1, q2 in get_pairs(path):
        solution.extend(extend_fn(samples[q1], samples[q2]))
    return solution, samples, edges, colliding_vertices, colliding_edges

##################################################

def replan_loop(start_conf, end_conf, sample_fn, extend_fn, collision_fn, params_list, smooth=0, **kwargs):
    if collision_fn(start_conf) or collision_fn(end_conf):
        return None
    from .meta import direct_path
    path = direct_path(start_conf, end_conf, extend_fn, collision_fn)
    if path is not None:
        return path
    for num_samples in params_list:
        path = lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn,
                        num_samples=num_samples, **kwargs)
        if path is not None:
            return smooth_path(path, extend_fn, collision_fn, max_iterations=smooth)
    return None
