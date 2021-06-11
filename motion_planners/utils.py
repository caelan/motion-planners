from random import shuffle
from itertools import islice
from collections import deque
import time
import contextlib
import pstats
import cProfile
import random

import numpy as np

INF = float('inf')

RRT_ITERATIONS = 20
RRT_RESTARTS = 2
RRT_SMOOTHING = 20


try:
   user_input = raw_input
except NameError:
   user_input = input


RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)


def apply_alpha(color, alpha=1.):
   return tuple(color[:3]) + (alpha,)


def irange(start, stop=None, step=1):  # np.arange
    if stop is None:
        stop = start
        start = 0
    while start < stop:
        yield start
        start += step


def negate(test):
    return lambda *args, **kwargs: not test(*args, **kwargs)


def argmin(function, sequence):
    # TODO: use min
    values = list(sequence)
    scores = [function(x) for x in values]
    return values[scores.index(min(scores))]


def get_pairs(lst):
    return list(zip(lst[:-1], lst[1:]))


def merge_dicts(*args):
    result = {}
    for d in args:
        result.update(d)
    return result
    # return dict(reduce(operator.add, [d.items() for d in args]))


def flatten(iterable_of_iterables):
    return (item for iterables in iterable_of_iterables for item in iterables)


def randomize(sequence):
    sequence = list(sequence)
    shuffle(sequence)
    return sequence


def is_even(num):
    return num % 2 == 0


def is_odd(num):
    return num % 2 == 1


def bisect(sequence):
    sequence = list(sequence)
    indices = set()
    queue = deque([(0, len(sequence)-1)])
    while queue:
        lower, higher = queue.popleft()
        if lower > higher:
            continue
        index = int((lower + higher) / 2.)
        assert index not in indices
        #if is_even(higher - lower):
        yield sequence[index]
        queue.extend([
            (lower, index-1),
            (index+1, higher),
        ])


def take(iterable, n=INF):
    if n == INF:
        n = None  # NOTE - islice takes None instead of INF
    elif n is None:
        n = 0  # NOTE - for some of the uses
    return islice(iterable, n)


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    enums['names'] = sorted(enums.keys(), key=lambda k: enums[k])
    return type('Enum', (), enums)


def elapsed_time(start_time):
    return time.time() - start_time


@contextlib.contextmanager
def profiler(field='tottime', num=10):
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    pstats.Stats(pr).sort_stats(field).print_stats(num) # cumtime | tottime


def inf_sequence():
    return iter(int, 1)


def find(test, sequence):
    for item in sequence:
        if test(item):
            return item
    raise RuntimeError()


def get_sign(x):
    if x > 0:
        return +1
    if x < 0:
        return -1
    return x


def strictly_increasing(sequence):
    return all(x2 > x1 for x1, x2 in get_pairs(sequence))

##################################################

def get_delta(q1, q2):
    return np.array(q2) - np.array(q1)


def get_distance(q1, q2):
    return np.linalg.norm(get_delta(q1, q2))


def get_unit_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm


def compute_path_cost(path, cost_fn=get_distance):
    if path is None:
        return INF
    return sum(cost_fn(*pair) for pair in get_pairs(path))


def get_difference(q2, q1):
    return np.array(q2) - np.array(q1)


def remove_redundant(path, tolerance=1e-3):
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = get_difference(new_path[-1], np.array(conf))
        if not np.allclose(np.zeros(len(difference)), difference, atol=tolerance, rtol=0):
            new_path.append(conf)
    return new_path


def waypoints_from_path(path, tolerance=1e-3):
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path
    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(get_difference(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(get_difference(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(get_difference(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints

##################################################

def convex_combination(x, y, w=0.5):
    return (1-w)*np.array(x) + w*np.array(y)


def uniform_generator(d):
    while True:
        yield np.random.uniform(size=d)


def halton_generator(d, seed=None):
    import ghalton
    if seed is None:
        seed = random.randint(0, 1000)
    #sequencer = ghalton.Halton(d)
    sequencer = ghalton.GeneralizedHalton(d, seed)
    #sequencer.reset()
    while True:
        [weights] = sequencer.get(1)
        yield np.array(weights)

def unit_generator(d, use_halton=False):
    if use_halton:
        try:
            import ghalton
        except ImportError:
            print('ghalton is not installed (https://pypi.org/project/ghalton/)')
            use_halton = False
    return halton_generator(d) if use_halton else uniform_generator(d)


def interval_generator(lower, upper, **kwargs):
    assert len(lower) == len(upper)
    assert np.less_equal(lower, upper).all()
    if np.equal(lower, upper).all():
        return iter([lower])
    return (convex_combination(lower, upper, w=weights) for weights in unit_generator(d=len(lower), **kwargs))

##################################################

def forward_selector(path):
    return path


def backward_selector(path):
    return reversed(list(path))


def random_selector(path):
    return randomize(path)


def bisect_selector(path):
    return bisect(path)


default_selector = bisect_selector # random_selector
