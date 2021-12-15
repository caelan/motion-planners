from random import randint, random

from .utils import INF, elapsed_time, irange, waypoints_from_path, get_pairs, get_distance, \
    convex_combination, compute_path_cost, default_selector, refine_waypoints, flatten

import time
import numpy as np

##################################################

def smooth_path_old(path, extend_fn, collision_fn, distance_fn=None,
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
            cost = compute_path_cost(smoothed_path, cost_fn=distance_fn) # TODO: incorporate costs
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

def smooth_path(path, extend_fn, collision_fn, distance_fn=None, sample_fn=None,
                max_iterations=50, max_time=INF, converge_time=INF, verbose=False):
    """
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_iterations: Maximum number of iterations - int
    :param max_time: Maximum runtime - float
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: makes an assumption on the distance_fn metric (to avoid sampling the same segment)
    # TODO: rename distance_fn to cost_fn
    # TODO: smooth until convergence
    # TODO: dynamic expansion of the nearby graph
    if (path is None) or (max_iterations is None):
        return path
    assert (max_iterations < INF) or (max_time < INF)
    start_time = last_time = time.time()
    if distance_fn is None:
        distance_fn = get_distance # TODO: don't use distance but instead just use samples
    waypoints = waypoints_from_path(path)
    cost = compute_path_cost(waypoints, cost_fn=distance_fn)
    for iteration in irange(max_iterations):
        #waypoints = waypoints_from_path(waypoints)
        if (elapsed_time(start_time) > max_time) or (elapsed_time(last_time) > converge_time) or (len(waypoints) <= 2):
            break
        # TODO: smoothing in the same linear segment when circular

        indices = list(range(len(waypoints)))
        segments = list(get_pairs(indices))
        distances = [distance_fn(waypoints[i], waypoints[j]) for i, j in segments]
        probabilities = np.array(distances) / sum(distances)
        if verbose:
            print('Iteration: {} | Waypoints: {} | Cost: {:.3f} | Elapsed: {:.3f} | Remaining: {:.3f}'.format(
                iteration, len(waypoints), cost, elapsed_time(start_time), max_time-elapsed_time(start_time)))

        #segment1, segment2 = choices(segments, weights=probabilities, k=2)
        seg_indices = list(range(len(segments)))
        seg_idx1, seg_idx2 = np.random.choice(seg_indices, size=2, replace=True, p=probabilities)
        if seg_idx1 == seg_idx2: # TODO: ensure not too far away
            continue
        if seg_idx2 < seg_idx1: # choices samples with replacement
            seg_idx1, seg_idx2 = seg_idx2, seg_idx1
        segment1, segment2 = segments[seg_idx1], segments[seg_idx2]
        # TODO: option to sample_fn only adjacent pairs
        point1, point2 = [convex_combination(waypoints[i], waypoints[j], w=random())
                          for i, j in [segment1, segment2]]
        i, _ = segment1
        _, j = segment2
        shortcut = [point1, point2]
        #shortcut = [point1, sample_fn(), point2]
        new_waypoints = waypoints[:i+1] + shortcut + waypoints[j:] # TODO: reuse computation
        new_cost = compute_path_cost(new_waypoints, cost_fn=distance_fn)
        if new_cost >= cost: # TODO: cost must have percent improvement above a threshold
            continue
        if not any(collision_fn(q) for q in default_selector(refine_waypoints(shortcut, extend_fn))):
            waypoints = new_waypoints
            cost = new_cost
            last_time = time.time()
    #return waypoints
    return refine_waypoints(waypoints, extend_fn)

#smooth_path = smooth_path_old
