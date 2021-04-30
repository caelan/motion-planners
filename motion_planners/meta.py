import time

from .smoothing import smooth_path
from .utils import RRT_RESTARTS, RRT_SMOOTHING, INF, irange, elapsed_time, compute_path_cost, default_selector


def direct_path(start, goal, extend_fn, collision_fn):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: version which checks whether the segment is valid
    if collision_fn(start) or collision_fn(goal):
        return None
    path = list(extend_fn(start, goal))
    path = [start] + path
    if any(collision_fn(q) for q in default_selector(path)):
        return None
    return path
    # path = [start]
    # for q in extend_fn(start, goal):
    #     if collision_fn(q):
    #         return None
    #     path.append(q)
    # return path


def random_restarts(solve_fn, start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                    restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING,
                    success_cost=0., max_time=INF, max_solutions=1, **kwargs):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_time: Maximum runtime - float
    :param kwargs: Keyword arguments
    :return: Paths [[q', ..., q"], [[q'', ..., q""]]
    """
    start_time = time.time()
    solutions = []
    if any(collision_fn(q) for q in [start, goal]):
        return solutions
    path = direct_path(start, goal, extend_fn, collision_fn)
    if path is not None:
        solutions.append(path)

    for attempt in irange(restarts + 1):
        if (len(solutions) >= max_solutions) or (elapsed_time(start_time) >= max_time):
            break
        attempt_time = (max_time - elapsed_time(start_time))
        path = solve_fn(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                        max_time=attempt_time, **kwargs)
        if path is None:
            continue
        if smooth is not None:
            path = smooth_path(path, extend_fn, collision_fn, iterations=smooth)
        solutions.append(path)
        if compute_path_cost(path, distance_fn) < success_cost:
            break
    solutions = sorted(solutions, key=lambda path: compute_path_cost(path, distance_fn))
    print('Solutions ({}): {} | Time: {:.3f}'.format(len(solutions), [(len(path), round(compute_path_cost(
        path, distance_fn), 3)) for path in solutions], elapsed_time(start_time)))
    return solutions

def solve_and_smooth(solve_fn, q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs):
    return random_restarts(solve_fn, q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, restarts=0, **kwargs)
