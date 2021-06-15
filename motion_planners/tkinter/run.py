from __future__ import print_function

import numpy as np
import argparse
import time
import random

from ..meta import solve
from ..trajectory.linear import solve_multi_linear
from ..trajectory.discretize import time_discretize_curve, V_MAX, A_MAX
from .samplers import get_sample_fn, get_collision_fn, get_extend_fn, get_distance_fn
from ..trajectory.smooth import smooth_curve, get_curve_collision_fn
from ..trajectory.limits import analyze_continuity
from .viewer import create_box, draw_environment, add_points, \
    add_roadmap, get_box_center, add_path, create_cylinder, add_timed_path
from ..utils import user_input, profiler, INF, compute_path_cost, elapsed_time, remove_redundant, waypoints_from_path
from ..prm import prm
from ..lazy_prm import lazy_prm
from ..rrt_connect import rrt_connect, birrt
from ..rrt import rrt
from ..rrt_star import rrt_star
from ..smoothing import smooth_path
from ..lattice import lattice

ALGORITHMS = [
    prm,
    lazy_prm,
    rrt,
    rrt_connect,
    birrt,
    rrt_star,
    lattice,
    # TODO: RRT in position/velocity space using spline interpolation
    # TODO: https://ompl.kavrakilab.org/planners.html
]

##################################################

def buffer_durations(durations, switch_t=0., min_t=0.):
    durations = durations + switch_t*np.ones(len(durations))
    durations = np.maximum(min_t * np.ones(len(durations)), durations)
    return durations

def dump_spline(positions_curve):
    print(positions_curve.c[0, ...]) # Cubic parameters
    print(positions_curve.c.shape)
    for d in range(positions_curve.c.shape[-1]):
        print(d, positions_curve.c[..., d])

def retime_path(path, collision_fn=lambda q: False, smooth=False, **kwargs):
    d = len(path[0])
    # v_max = 5.*np.ones(d)
    # a_max = v_max / 1.
    v_max, a_max = V_MAX, A_MAX
    print('Max vel: {} | Max accel: {}'.format(v_max, a_max))

    waypoints = remove_redundant(path)
    waypoints = waypoints_from_path(waypoints)
    positions_curve = solve_multi_linear(waypoints, v_max, a_max)
    if not smooth:
        return positions_curve

    # durations = [0.] + [get_distance(*pair) / velocity for pair in get_pairs(waypoints)]
    # durations = [0.] + [solve_multivariate_ramp(x1, x2, np.zeros(d), np.zeros(d), v_max, a_max)
    #                     for x1, x2 in get_pairs(waypoints)]
    # durations = [0.] + [max(spline_duration(opt_straight_line(x1[k], x2[k], v_max=v_max[k], a_max=a_max[k])) for k in range(d))
    #                    for x1, x2 in get_pairs(waypoints)] # min_linear_spline | opt_straight_line
    # times = np.cumsum(durations)

    #positions_curve = interp1d(times, waypoints, kind='quadratic', axis=0) # linear | slinear | quadratic | cubic
    #positions_curve = CubicSpline(times, waypoints, bc_type='clamped')
    #velocities = [np.zeros(len(waypoint)) for waypoint in waypoints]
    #positions_curve = CubicHermiteSpline(times, waypoints, dydx=velocities)

    #positions_curve = MultiPPoly.from_poly(positions_curve)
    #positions_curve = solve_multi_poly(times, waypoints, velocities, v_max, a_max)
    #positions_curve = positions_curve.spline()
    #positions_curve = positions_curve.hermite_spline()
    print('Position: t={:.3f}, error={:.3E}'.format(*analyze_continuity(positions_curve)))
    print('Velocity: t={:.3f}, error={:.3E}'.format(*analyze_continuity(positions_curve.derivative(nu=1))))
    print('Acceleration: t={:.3f}, error={:.3E}'.format(*analyze_continuity(positions_curve.derivative(nu=2))))

    # t1, t2 = np.random.uniform(positions_curve.x[0], positions_curve.x[-1], 2)
    # if t1 > t2:
    #     t1, t2 = t2, t1
    # print(t1, t2)
    # print([positions_curve(t) for t in [t1, t2]])
    # positions_curve = trim(positions_curve, t1, t2) # trim | trim_start | trim_end
    # print(positions_curve.x)
    # print([positions_curve(t) for t in [t1, t2]])

    curve_collision_fn = get_curve_collision_fn(collision_fn, max_velocities=v_max, max_accelerations=a_max)
    positions_curve = smooth_curve(positions_curve,
                                   #v_max=None, a_max=None,
                                   v_max=v_max,
                                   a_max=a_max,
                                   curve_collision_fn=curve_collision_fn, **kwargs)
    print('Position: t={:.3f}, error={:.3E}'.format(*analyze_continuity(positions_curve)))
    print('Velocity: t={:.3f}, error={:.3E}'.format(*analyze_continuity(positions_curve.derivative(nu=1))))
    print('Acceleration: t={:.3f}, error={:.3E}'.format(*analyze_continuity(positions_curve.derivative(nu=2))))

    return positions_curve

##################################################

def problem1():
    obstacles = [
        create_box(center=(.35, .75), extents=(.25, .25)),
        create_box(center=(.75, .35), extents=(.25, .25)),
        #create_box(center=(.75, .35), extents=(.225, .225)),
        create_box(center=(.5, .5), extents=(.25, .25)),
        #create_box(center=(.5, .5), extents=(.225, .225)),

        create_cylinder(center=(.25, .25), radius=.1),
    ]

    # TODO: alternate sampling from a mix of regions
    regions = {
        'env': create_box(center=(.5, .5), extents=(1., 1.)),
        'green': create_box(center=(.8, .8), extents=(.1, .1)),
    }
    start = np.array([0., 0.])
    goal = 'green'

    return start, goal, regions, obstacles

##################################################

def main():
    """
    Creates and solves the 2D motion planning problem.
    """
    # https://github.com/caelan/pddlstream/blob/master/examples/motion/run.py
    # TODO: 3D workspace and CSpace
    # TODO: visualize just the tool frame of an end effector

    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='rrt_connect',
                        help='The algorithm to use.')
    parser.add_argument('-d', '--draw', action='store_true',
                        help='When enabled, draws the roadmap')
    parser.add_argument('-r', '--restarts', default=0, type=int,
                        help='The number of restarts.')
    parser.add_argument('-s', '--smooth', action='store_true',
                        help='When enabled, smooths paths.')
    parser.add_argument('-t', '--time', default=1., type=float,
                        help='The maximum runtime.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The random seed to use.')
    args = parser.parse_args()
    print(args)
    # TODO: matplotlib viewer

    seed = args.seed
    if seed is None:
        #seed = random.randint(0, sys.maxsize)
        seed = random.randint(0, 10**3-1)
    print('Seed:', seed)
    random.seed(seed)
    np.random.seed(seed)

    #########################

    start, goal, regions, obstacles = problem1()
    environment = regions['env']
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
            collision_fn, cfree = get_collision_fn(environment, obstacles)
            sample_fn, samples = get_sample_fn(environment, obstacles=[], use_halton=False) # obstacles
            extend_fn, roadmap = get_extend_fn(environment, obstacles=obstacles)  # obstacles | []

            path = solve(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                         max_time=args.time, max_iterations=INF, num_samples=200,
                         restarts=2, smooth=0, algorithm=args.algorithm)
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
                #path = path[:1] + path[-2:]
                path = waypoints_from_path(path)
                add_path(viewer, path, color='green')
                #curve = interpolate_path(path) # , collision_fn=collision_fn)
                curve = retime_path(path, collision_fn=collision_fn, smooth=True)
                times, path = time_discretize_curve(curve)
                times = [np.linalg.norm(curve(t, nu=1), ord=INF) for t in times]
                #add_points(viewer, [curve(t) for t in curve.x])
                #add_path(viewer, path, color='red')
                add_timed_path(viewer, times, path) # TODO: add curve

            if args.smooth:
                for path in paths:
                    extend_fn, roadmap = get_extend_fn(environment, obstacles=obstacles)  # obstacles | []
                    smoothed = smooth_path(path, extend_fn, collision_fn, max_iterations=INF, max_time=args.time)
                    print('Smoothed distance_fn: {:.3f}'.format(compute_path_cost(smoothed, distance_fn)))
                    add_path(viewer, smoothed, color='red')
            user_input('Finish?')

if __name__ == '__main__':
    main()
