from __future__ import print_function

import numpy as np
import math
import argparse
import time

from .viewer import sample_box, is_collision_free, \
    create_box, draw_environment, point_collides, sample_line, add_points, \
    add_roadmap, get_box_center, add_path, get_distance_fn
from ..utils import user_input, profiler, INF, compute_path_cost, get_distance, elapsed_time, interval_generator
from ..prm import prm
from ..lazy_prm import lazy_prm
from ..rrt_connect import rrt_connect, birrt
from ..rrt import rrt
from ..rrt_star import rrt_star
from ..smoothing import smooth_path
from motion_planners.lattice import lattice
from ..meta import random_restarts
from ..diverse import score_portfolio, exhaustively_select_portfolio

ALGORITHMS = [
    prm,
    lazy_prm,
    rrt,
    rrt_connect,
    birrt,
    rrt_star,
    lattice,
    # TODO: https://ompl.kavrakilab.org/planners.html
]

##################################################

def get_sample_fn(region, obstacles=[], use_halton=True): #, check_collisions=False):
    # TODO: Gaussian sampling for narrow passages
    samples = []
    collision_fn, _ = get_collision_fn(obstacles)
    lower, upper = region
    generator = interval_generator(lower, upper, use_halton=use_halton)

    def region_gen():
        #area = np.product(upper - lower) # TODO: sample_fn proportional to area
        for q in generator:
            #q = sample_box(region)
            if collision_fn(q):
                continue
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

def get_threshold_fn(d=2):
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.419.5503&rep=rep1&type=pdf
    vol_free = (1 - 0) * (1 - 0)
    vol_ball = math.pi * (1 ** 2)
    gamma = 2 * ((1 + 1. / d) * (vol_free / vol_ball)) ** (1. / d)
    threshold_fn = lambda n: gamma * (math.log(n) / n) ** (1. / d)
    return threshold_fn

def get_collision_fn(obstacles):
    cfree = []

    def collision_fn(q):
        #time.sleep(1e-3)
        if point_collides(q, obstacles):
            return True
        cfree.append(q)
        return False

    return collision_fn, cfree

def get_extend_fn(obstacles=[]):
    collision_fn, _ = get_collision_fn(obstacles)
    roadmap = []

    def extend_fn(q1, q2):
        path = [q1]
        for q in sample_line(segment=(q1, q2)):
            yield q
            if collision_fn(q):
                path = None
            if path is not None:
                roadmap.append((path[-1], q))
                path.append(q)

    return extend_fn, roadmap

##################################################

def main():
    """
    Creates and solves the 2D motion planning problem.
    """
    # https://github.com/caelan/pddlstream/blob/master/examples/motion/run.py
    # TODO: 3D work and CSpace
    # TODO: visualize just the tool frame of an end effector

    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='rrt_connect',
                        help='The algorithm seed to use.')
    parser.add_argument('-d', '--draw', action='store_true',
                        help='When enabled, draws the roadmap')
    parser.add_argument('-r', '--restarts', default=0, type=int,
                        help='The number of restarts.')
    parser.add_argument('-s', '--smooth', action='store_true',
                        help='When enabled, smooths paths.')
    parser.add_argument('-t', '--time', default=1., type=float,
                        help='The maximum runtime.')
    args = parser.parse_args()

    #########################

    obstacles = [
        create_box(center=(.35, .75), extents=(.25, .25)),
        create_box(center=(.75, .35), extents=(.225, .225)),
        create_box(center=(.5, .5), extents=(.225, .225)),
    ]

    # TODO: alternate sampling from a mix of regions
    regions = {
        'env': create_box(center=(.5, .5), extents=(1., 1.)),
        'green': create_box(center=(.8, .8), extents=(.1, .1)),
    }

    start = np.array([0., 0.])
    goal = 'green'
    if isinstance(goal, str) and (goal in regions):
        goal = get_box_center(regions[goal])
    else:
        goal = np.array([1., 1.])

    title = args.algorithm
    if args.smooth:
        title += '+shortcut'
    viewer = draw_environment(obstacles, regions, title=title)

    #########################

    #connected_test, roadmap = get_connected_test(obstacles)
    distance_fn = get_distance_fn(weights=[1, 1]) # distance_fn

    # samples = list(islice(region_gen('env'), 100))
    with profiler(field='cumtime'): # cumtime | tottime
        # TODO: cost bound & best cost
        for _ in range(args.restarts+1):
            start_time = time.time()
            collision_fn, cfree = get_collision_fn(obstacles)
            sample_fn, samples = get_sample_fn(regions['env'], obstacles=[]) # obstacles
            extend_fn, roadmap = get_extend_fn(obstacles=obstacles)  # obstacles | []

            if args.algorithm == 'prm':
                path = prm(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                           num_samples=200)
            elif args.algorithm == 'lazy_prm':
                path = lazy_prm(start, goal, sample_fn, extend_fn, collision_fn,
                                num_samples=200, max_time=args.time)[0]
            elif args.algorithm == 'rrt':
                path = rrt(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                           iterations=INF, max_time=args.time)
            elif args.algorithm == 'rrt_connect':
                path = rrt_connect(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                                   max_time=args.time)
            elif args.algorithm == 'birrt':
                path = birrt(start, goal, distance_fn=distance_fn, sample_fn=sample_fn,
                             extend_fn=extend_fn, collision_fn=collision_fn,
                             max_time=args.time, smooth=100)
            elif args.algorithm == 'rrt_star':
                path = rrt_star(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                                radius=1, max_iterations=INF, max_time=args.time)
            elif args.algorithm == 'lattice':
                path = lattice(start, goal, extend_fn, collision_fn, distance_fn=distance_fn)
            else:
                raise NotImplementedError(args.algorithm)
            paths = [] if path is None else [path]

            #paths = random_restarts(rrt_connect, start, goal, distance_fn=distance_fn, sample_fn=sample_fn,
            #                         extend_fn=extend_fn, collision_fn=collision_fn, restarts=INF,
            #                         max_time=args.time, max_solutions=INF, smooth=100) #, smooth=1000, **kwargs)

            # paths = exhaustively_select_portfolio(paths, k=2)
            # print(score_portfolio(paths))

            #########################

            if args.draw:
                # roadmap = samples = cfree = []
                add_roadmap(viewer, roadmap, color='black')
                add_points(viewer, samples, color='red', radius=2)
                #add_points(viewer, cfree, color='blue', radius=2)

            print('Solutions ({}): {} | Time: {:.3f}'.format(len(paths), [(len(path), round(compute_path_cost(
                path, distance_fn), 3)) for path in paths], elapsed_time(start_time)))
            for path in paths:
                add_path(viewer, path, color='green')

            if args.smooth:
                for path in paths:
                    extend_fn, roadmap = get_extend_fn(obstacles=obstacles)  # obstacles | []
                    smoothed = smooth_path(path, extend_fn, collision_fn, iterations=INF, max_time=args.time)
                    print('Smoothed distance_fn: {:.3f}'.format(compute_path_cost(smoothed, distance_fn)))
                    add_path(viewer, smoothed, color='red')
            user_input('Finish?')

if __name__ == '__main__':
    main()
