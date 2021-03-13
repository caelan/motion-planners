from __future__ import print_function

import numpy as np
import math

from motion_planners.tkinter.viewer import sample_box, get_distance, is_collision_free, \
    create_box, draw_solution, draw_roadmap, draw_environment, point_collides, sample_line, add_points, \
    add_segments, add_roadmap, get_box_center
from motion_planners.utils import user_input, pairs, profiler
from motion_planners.rrt_connect import birrt


##################################################

def get_sample_fn(region, obstacles=[]):
    samples = []

    def region_gen():
        #lower, upper = region
        #area = np.product(upper - lower) # TODO: sample proportional to area
        q = sample_box(region)
        samples.append(q)
        return q # TODO: sampling with state (e.g. deterministic sampling)
    return region_gen, samples

def get_connected_test(obstacles, max_distance=0.25): # 0.25 | 0.2 | 0.25 | 0.5 | 1.0
    roadmap = []

    def connected_test(q1, q2):
        #n = len(samples)
        #threshold = gamma * (math.log(n) / n) ** (1. / d)
        threshold = max_distance
        are_connected = (get_distance(q1, q2) <= threshold) and is_collision_free((q1, q2), obstacles)
        if are_connected:
            roadmap.append((q1, q2))
        return are_connected
    return connected_test, roadmap

def get_threshold_fn():
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.419.5503&rep=rep1&type=pdf
    d = 2
    vol_free = (1 - 0) * (1 - 0)
    vol_ball = math.pi * (1 ** 2)
    gamma = 2 * ((1 + 1. / d) * (vol_free / vol_ball)) ** (1. / d)
    threshold_fn = lambda n: gamma * (math.log(n) / n) ** (1. / d)
    return threshold_fn

def get_collision_fn(obstacles):
    return lambda q: point_collides(q, obstacles)

def get_extend_fn():
    roadmap = []

    def extend_fn(q1, q2):
        path = [q1]
        for q in sample_line(segment=(q1, q2)):
            yield q
            roadmap.append((path[-1], q))
            path.append(q)

    return extend_fn, roadmap

##################################################

# TODO: algorithms that take advantage of metric space (RRT)

def main(max_time=20):
    """
    Creates and solves the 2D motion planning problem.
    """
    # https://github.com/caelan/pddlstream/blob/master/examples/motion/run.py

    np.set_printoptions(precision=3)

    obstacles = [
        create_box(center=(.5, .5), extents=(.25, .25)),
    ]

    # TODO: alternate sampling from a mix of regions
    regions = {
        'env': create_box(center=(.5, .5), extents=(1., 1.)),
        'green': create_box(center=(.8, .8), extents=(.1, .1)),
    }

    q0 = np.array([0., 0.])
    goal = 'green'
    if isinstance(goal, str) and (goal in regions):
        goal = get_box_center(regions[goal])
    else:
        goal = np.array([1., 1.])

    viewer = draw_environment(obstacles, regions)

    #########################

    sample_fn, samples = get_sample_fn(regions['env'])
    #connected_test, roadmap = get_connected_test(obstacles)
    collision_fn = get_collision_fn(obstacles)
    distance_fn = get_distance
    extend_fn, roadmap = get_extend_fn()

    with profiler():
        path = birrt(q0, goal, distance=distance_fn, sample=sample_fn,
                     extend=extend_fn, collision=collision_fn, smooth=1000) #, **kwargs)

    #samples = list(islice(region_gen('env'), 100))
    #prm = DegreePRM(distance=get_distance, extend=None, collision=get_collision_fn(obstacles),
    #                samples=samples, connect_distance=.25)

    #########################

    print(path)
    add_roadmap(viewer, roadmap, color='black')
    add_points(viewer, samples, color='blue')

    if path is None:
        user_input('Finish?')
        return

    segments = list(pairs(path))
    add_segments(viewer, segments, color='green')
    user_input('Finish?')


if __name__ == '__main__':
    main()
