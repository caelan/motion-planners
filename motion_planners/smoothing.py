from random import randint, random
from .utils import INF, elapsed_time, irange, waypoints_from_path, get_pairs, get_distance, \
    convex_combination, flatten, traverse, compute_path_cost

import time
import numpy as np

def smooth_path_old(path, extend, collision, iterations=50, max_time=INF, verbose=False, **kwargs):
    assert (iterations < INF) or (max_time < INF)
    start_time = time.time()
    smoothed_path = path
    for iteration in irange(iterations):
        if (elapsed_time(start_time) > max_time) or (len(smoothed_path) <= 2):
            break
        if verbose:
            print('Iteration: {} | Waypoints: {} | Euclidean distance: {:.3f} | Time: {:.3f}'.format(
                iteration, len(smoothed_path), compute_path_cost(smoothed_path), elapsed_time(start_time)))
        i = randint(0, len(smoothed_path) - 1)
        j = randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))
        if (len(shortcut) < (j - i)) and all(not collision(q) for q in traverse(shortcut)):
            smoothed_path = smoothed_path[:i + 1] + shortcut + smoothed_path[j + 1:]
    return smoothed_path

def refine_waypoints(waypoints, extend_fn):
    #if len(waypoints) <= 1:
    #    return waypoints
    return list(flatten(extend_fn(q1, q2) for q1, q2 in get_pairs(waypoints))) # [waypoints[0]] +

def smooth_path(path, extend, collision, distance_fn=None, iterations=50, max_time=INF, verbose=False):
    # TODO: makes an assumption on the distance metric
    # TODO: smooth until convergence
    assert (iterations < INF) or (max_time < INF)
    start_time = time.time()
    if distance_fn is None:
        distance_fn = get_distance
    waypoints = waypoints_from_path(path)
    for iteration in irange(iterations):
        #waypoints = waypoints_from_path(waypoints)
        if (elapsed_time(start_time) > max_time) or (len(waypoints) <= 2):
            break
        # TODO: smoothing in the same linear segment when circular

        indices = list(range(len(waypoints)))
        segments = list(get_pairs(indices))
        distances = [distance_fn(waypoints[i], waypoints[j]) for i, j in segments]
        total_distance = sum(distances)
        if verbose:
            print('Iteration: {} | Waypoints: {} | Distance: {:.3f} | Time: {:.3f}'.format(
                iteration, len(waypoints), total_distance, elapsed_time(start_time)))
        probabilities = np.array(distances) / total_distance

        #segment1, segment2 = choices(segments, weights=probabilities, k=2)
        seg_indices = list(range(len(segments)))
        seg_idx1, seg_idx2 = np.random.choice(seg_indices, size=2, replace=True, p=probabilities)
        if seg_idx1 == seg_idx2:
            continue
        if seg_idx2 < seg_idx1: # choices samples with replacement
            seg_idx1, seg_idx2 = seg_idx2, seg_idx1
        segment1, segment2 = segments[seg_idx1], segments[seg_idx2]
        # TODO: option to sample only adjacent pairs
        point1, point2 = [convex_combination(waypoints[i], waypoints[j], w=random())
                          for i, j in [segment1, segment2]]
        i, _ = segment1
        _, j = segment2
        new_waypoints = waypoints[:i+1] + [point1, point2] + waypoints[j:] # TODO: reuse computation
        if compute_path_cost(new_waypoints, cost_fn=distance_fn) >= total_distance:
            continue
        if all(not collision(q) for q in traverse(extend(point1, point2))):
            waypoints = new_waypoints
    #return waypoints
    return refine_waypoints(waypoints, extend)

#smooth_path = smooth_path_old
