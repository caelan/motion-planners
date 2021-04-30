from itertools import islice

import numpy as np

from motion_planners.discrete import best_first


def get_nth(generator, n=0):
    return next(islice(generator, n, None))


def get_neighbors_fn(extend_fn, targets=[], scale=1e3): # TODO: could also include diagonal
    # https://github.mit.edu/caelan/lis-openrave/blob/master/manipulation/motion/cspace.py#L171
    def neighbors_fn(current):
        d = len(current)
        for target in targets:
            yield get_nth(extend_fn(current, target), n=1)
        for k in range(d):
            direction = np.zeros(d)
            direction[k] = scale
            for sign in [-1, +1]:
                # TODO: hash the confs
                target = tuple(np.array(current) + sign * direction)
                #yield target
                yield get_nth(extend_fn(current, target), n=1)
    return neighbors_fn


def lattice(start, goal, distance_fn, extend_fn, collision_fn, **kwargs):
    #collision_fn = lambda q: False
    neighbors_fn = get_neighbors_fn(extend_fn, targets=[goal])
    return best_first(start, goal, distance_fn, neighbors_fn, collision_fn, **kwargs)