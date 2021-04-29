from collections import deque
from heapq import heappop, heappush

from recordclass import recordclass
import numpy as np
import time

from .utils import INF, elapsed_time

# https://github.mit.edu/caelan/lis-openrave/tree/master/manipulation/motion
# https://github.mit.edu/caelan/lis-openrave/commit/4d8683407ec79a7c39dab62d6779804730ff598d

Node = recordclass('Node', ['g', 'parent'])


def retrace(visited, q):
    if q is None:
        return []
    return retrace(visited, visited[tuple(q)].parent) + [q]


def bfs(start, goal, neighbors, collision, max_iterations=INF, max_time=INF):
    start_time = time.time()
    if collision(start) or collision(goal):
        return None
    iterations = 0
    visited = {tuple(start): Node(0, None)}
    queue = deque([start])
    while queue and (iterations < max_iterations) and (elapsed_time(start_time) < max_time):
        iterations += 1
        current = queue.popleft()
        if goal is not None and tuple(current) == tuple(goal):
            return retrace(visited, current)
        for next in neighbors(current):
            # TODO - make edges for real (and store bad edges)
            if (tuple(next) not in visited) and not collision(next):
                visited[tuple(next)] = Node(
                    next, visited[tuple(current)].g + 1, current)
                queue.append(next)
    return None

##################################################

def weighted(weight=1.):
    if weight == INF:
        return lambda g, h: h
    return lambda g, h: g + weight*h

uniform = weighted(0)
astar = weighted(1)
wastar2 = weighted(2)
wastar3 = weighted(2)
greedy = weighted(INF)
lexicographic = lambda g, h: (h, g)

def best_first(start, goal, distance, neighbors, collision,
               max_iterations=INF, max_time=INF, priority=lexicographic):  # TODO - put start and goal in neighbors
    start_time = time.time()
    if collision(start) or collision(goal):
        return None
    queue = [(priority(0, distance(start, goal)), 0, start)]
    visited = {tuple(start): Node(0, None)}
    iterations = 0
    while queue and (iterations < max_iterations) and (elapsed_time(start_time) < max_time):
        _, current_g, current = heappop(queue)
        current = np.array(current)
        if visited[tuple(current)].g != current_g:
            continue
        # TODO: lazy collision checking
        iterations += 1
        if tuple(current) == tuple(goal):
            return retrace(visited, current)
        for next in neighbors(current):
            next_g = current_g + distance(current, next)
            if (tuple(next) not in visited or next_g < visited[tuple(next)].g) and not collision(next):
                visited[tuple(next)] = Node(next_g, current)
                # ValueError: The truth value of an array with more than one
                # element is ambiguous.
                heappush(queue, (priority(next_g, distance(next, goal)), next_g, next))
    return None
