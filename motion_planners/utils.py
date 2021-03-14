from random import shuffle
from itertools import islice
import time
import contextlib
import pstats
import cProfile

INF = float('inf')

RRT_ITERATIONS = 20
RRT_RESTARTS = 2
RRT_SMOOTHING = 20


try:
   user_input = raw_input
except NameError:
   user_input = input


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


def pairs(lst):
    return zip(lst[:-1], lst[1:])


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


def compute_path_cost(path, cost_fn):
    if path is None:
        return INF
    return sum(cost_fn(*pair) for pair in pairs(path))
