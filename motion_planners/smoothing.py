from random import randint, random, choice

from .utils import INF, elapsed_time, irange, waypoints_from_path, get_pairs, get_distance, \
    convex_combination, compute_path_cost, default_selector, refine_waypoints, flatten
from .primitives import distance_fn_from_extend_fn

import time
import numpy as np

def unit_cost_fn(*args, **kwargs):
    # TODO: fixed cost function
    return 1

##################################################

def smooth_path_old(path, extend_fn, collision_fn, cost_fn=None,
                    max_iterations=50, max_time=INF, verbose=False, **kwargs):
    """
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_iterations: Maximum number of iterations - int
    :param max_time: Maximum runtime - float
    :param kwargs: Keyword arguments
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    if (path is None) or (max_iterations is None):
        return path
    assert (max_iterations < INF) or (max_time < INF)
    start_time = time.time()
    smoothed_path = path
    for iteration in irange(max_iterations):
        if (elapsed_time(start_time) > max_time) or (len(smoothed_path) <= 2):
            break
        if verbose:
            cost = compute_path_cost(smoothed_path, cost_fn=cost_fn) # TODO: incorporate costs
            print('Iteration: {} | Waypoints: {} | Cost: {:.3f} | Time: {:.3f}'.format(
                iteration, len(smoothed_path), cost, elapsed_time(start_time)))
        i = randint(0, len(smoothed_path) - 1)
        j = randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend_fn(smoothed_path[i], smoothed_path[j]))
        if (len(shortcut) < (j - i)) and all(not collision_fn(q) for q in default_selector(shortcut)):
            smoothed_path = smoothed_path[:i + 1] + shortcut + smoothed_path[j + 1:]
    return smoothed_path

##################################################

def smooth_path(path, extend_fn, collision_fn, distance_fn=None, cost_fn=None, sample_fn=None, # TODO: deprecate sample_fn
                max_iterations=100, max_time=INF, converge_fraction=0.0, converge_time=INF, print_frequency=1., verbose=False):
    """
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_iterations: Maximum number of iterations - int
    :param max_time: Maximum runtime - float
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: makes an assumption on extend_fn (to avoid sampling the same segment)
    # TODO: smooth until convergence
    # TODO: dynamic expansion of the nearby graph
    start_time = last_time = time.time()
    print_time = -INF
    if not path or not max_iterations:
        return path
    assert (max_iterations < INF) or (max_time < INF) or (converge_time < INF)
    if distance_fn is None:
        distance_fn = distance_fn_from_extend_fn(extend_fn)
    if cost_fn is None:
        cost_fn = distance_fn
    # TODO: use extend_fn to extract waypoints
    waypoints = waypoints_from_path(path, difference_fn=None) # TODO: difference_fn
    cost = compute_path_cost(waypoints, cost_fn=cost_fn)
    initial_cost = cost
    #paths = [extend_fn(*pair) for pair in get_pairs(waypoints)] # TODO: update incrementally
    #costs = [cost_fn(*pair) for pair in get_pairs(waypoints)]
    for iteration in irange(max_iterations):
        current_duration = elapsed_time(start_time)
        remaining_duration = max_time - current_duration
        last_duration = elapsed_time(last_time)
        improve_duration = converge_time - last_duration
        if (remaining_duration <= 0.) or (improve_duration <= 0.) or (len(waypoints) <= 2):
            break

        segments = get_pairs(range(len(waypoints)))
        weights = [distance_fn(waypoints[i], waypoints[j]) for i, j in segments] # TODO: avoid recomputing
        #paths = [list(extend_fn(*pair)) for pair in get_pairs(waypoints)]
        #weights = [len(paths[i]) for i, j in segments]
        probabilities = np.array(weights) / sum(weights)
        if verbose and (print_frequency is not None) and (elapsed_time(print_time) >= print_frequency):
            # TODO: only if an improvement
            print('Iteration: {} | Waypoints: {} | Cost: {:.5f} | Improve: {:.3f} | Converge: {:.3f} | Current: {:.3f} | Remaining: {:.3f}'.format(
                iteration, len(waypoints), cost, last_duration, improve_duration, current_duration, remaining_duration))
            print_time = time.time()

        #segment1, segment2 = choices(segments, weights=probabilities, k=2)
        seg_indices = list(range(len(segments)))
        seg_idx1, seg_idx2 = np.random.choice(seg_indices, size=2, replace=True, p=probabilities)
        # TODO: max bridging distance
        # TODO: improve normalized by distance
        # TODO: distribution of distances
        # TODO: shuffle pairs to avoid redoing the same extensions
        if seg_idx1 == seg_idx2: # TODO: ensure not too far away
            continue
        if seg_idx2 < seg_idx1: # choices samples with replacement
            seg_idx1, seg_idx2 = seg_idx2, seg_idx1
        segment1, segment2 = segments[seg_idx1], segments[seg_idx2]
        # TODO: option to sample_fn only adjacent pairs
        #point1, point2 = [convex_combination(waypoints[i], waypoints[j], w=random())
        #                  for i, j in [segment1, segment2]]
        # point1, point2 = [choice(paths[i]) for i, _ in [segment1, segment2]]
        paths = [list(extend_fn(waypoints[i], waypoints[j]))[:-1] for i, j in [segment1, segment2]]
        if not all(paths):
            # TODO: probably should include endpoints
            # TODO: increase extend_fn resolution
            continue
        point1, point2 = [choice(path) for path in paths]

        i, _ = segment1
        _, j = segment2
        shortcut = [point1, point2]
        # if sample_fn is not None:
        #     shortcut = [point1, sample_fn(), point2]
        #shortcut_paths = [extend_fn(*pair) for pair in get_pairs(waypoints)]
        new_waypoints = waypoints[:i+1] + shortcut + waypoints[j:] # TODO: reuse computation
        new_cost = compute_path_cost(new_waypoints, cost_fn=cost_fn)
        improve_fraction = (cost - new_cost) / cost
        if (new_cost >= cost) or (improve_fraction <= converge_fraction) or any(map(collision_fn, shortcut)) or \
                any(collision_fn(q) for q in default_selector(refine_waypoints(shortcut, extend_fn)[:-1])):
            continue
        # print('Iteration: {}) Start index: {} | End index: {} | Old cost: {:.5f} | New cost: {:.5f} | Improve: {:.3%}'.format(
        #     iteration, i, j, cost, new_cost, improve_fraction))
        waypoints = new_waypoints
        cost = new_cost
        last_time = time.time()
        # TODO: time between last improvement
    if verbose:
        print('Iterations: {} | Waypoints: {} | DOFs: {} | Final cost: {:.5f} | Initial cost: {:.5f} | '
              'Final Improve: {:.3f} | Elapsed: {:.3f}'.format(
            iteration, len(waypoints), len(waypoints[0]), cost, initial_cost, last_time - start_time, elapsed_time(start_time)))
    #return waypoints
    return [waypoints[0]] + refine_waypoints(waypoints, extend_fn)

#smooth_path = smooth_path_old
