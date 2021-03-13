from __future__ import print_function

import numpy as np
import math

from motion_planners.tkinter.viewer import sample_box, get_distance, is_collision_free, \
    create_box, draw_solution, draw_roadmap, draw_environment
from motion_planners.utils import user_input, INF
from motion_planners.rrt_connect import birrt
from motion_planners.prm import DegreePRM

##################################################

ARRAY = np.array # No hashing
#ARRAY = list # No hashing
#ARRAY = tuple # Hashing

def get_sample_fn(regions):
    samples = []

    def region_gen(region):
        lower, upper = regions[region]
        area = np.product(upper - lower)
        # TODO: sample proportional to area
        while True:
            q = ARRAY(sample_box(regions[region]))
            samples.append(q)
            yield (q,)
    return region_gen, samples

def get_connected_test(obstacles, max_distance=0.5): # max_distance = 0.25 # 0.2 | 0.25 | 0.5 | 1.0
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

def solve_birrt(q1, q2, **kwargs):
    return birrt(start_positions, end_positions, distance=get_distance, sample=sample_fn,
                 extend=extend_fn, collision=collision_fn, **kwargs)

##################################################

# TODO: algorithms that take advantage of metric space (RRT)

def main(max_time=20):
    """
    Creates and solves the 2D motion planning problem.
    """
    # https://github.com/caelan/pddlstream/blob/master/examples/motion/run.py

    np.set_printoptions(precision=3)

    obstacles = [
        create_box((.5, .5), (.25, .25))
    ]
    regions = {
        'env': create_box((.5, .5), (1., 1.)),
        'green': create_box((.8, .8), (.1, .1)),
    }

    q0 = ARRAY([0, 0])
    goal = 'green'
    if goal not in regions:
        goal = ARRAY([1, 1])

    region_gen, samples = get_sample_fn(regions)
    connected_test, roadmap = get_connected_test(obstacles)

    #path = birrt(start_positions, end_positions, distance=get_distance, sample=sample_fn,
    #             extend=extend_fn, collision=collision_fn, **kwargs)
    #prm = DegreePRM(distance=get_distance, )

    #viewer = draw_environment(obstacles, regions)
    #for sample in samples:
    #    viewer.draw_point(sample)
    #user_input('Continue?')

    # TODO: use the same viewer here
    draw_roadmap(roadmap, obstacles, regions) # TODO: do this in realtime
    user_input('Continue?')

    #if plan is None:
    #    return
    #segments = [args for name, args in plan]
    draw_solution(segments, obstacles, regions)
    user_input('Finish?')


if __name__ == '__main__':
    main()
