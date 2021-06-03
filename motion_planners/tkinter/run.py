from __future__ import print_function

import numpy as np
import math
import argparse
import time

from .viewer import sample_box, is_collision_free, \
    create_box, draw_environment, point_collides, sample_line, add_points, \
    add_roadmap, get_box_center, add_path, get_distance_fn, create_cylinder
from ..utils import user_input, profiler, INF, compute_path_cost, get_distance, elapsed_time, interval_generator, \
    get_pairs, remove_redundant, waypoints_from_path
from ..prm import prm
from ..lazy_prm import lazy_prm
from ..rrt_connect import rrt_connect, birrt
from ..rrt import rrt
from ..rrt_star import rrt_star
from ..smoothing import smooth_path
from ..lattice import lattice
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

def solve_two_ramp(x1, x2, v1, v2, a_max, v_max=INF):
    solutions = np.roots([
        a_max,
        2 * v1,
        (v1**2 - v2**2) / (2 * a_max) + (x1 - x2),
    ])
    solutions = [t for t in solutions if not isinstance(t, complex) and (t >= 0.)]
    #solutions = [t for t in solutions if t <= (v2 - v1) / a_max] # TODO: this constraint is strange
    solutions = [t for t in solutions if abs(v1 + t*a_max) <= abs(v_max)]
    if not solutions:
        return None
    t = min(solutions)
    T = t + 2 * (v1 - v2) / a_max
    if T < 0:
        return None
    return T

def solve_three_ramp(x1, x2, v1, v2, v_max, a_max):
    # http://motion.pratt.duke.edu/papers/icra10-smoothing.pdf
    # https://github.com/Puttichai/parabint/blob/2662d4bf0fbd831cdefca48863b00d1ae087457a/parabint/optimization.py
    # TODO: minimum-switch-time constraint
    #assert np.positive(v_max).all() and np.positive(a_max).all()
    # P+L+P-
    tp1 = (v_max - v1) / a_max
    tp2 = (v2 - v_max) / a_max
    tl = (v2 ** 2 + v1 ** 2 - 2 * v_max ** 2) / (2 * v_max * a_max) + (x2 - x1) / v_max
    ts = [tp1, tl, tp2]
    if any(t < 0 for t in ts):
        return None
    T = sum(ts)
    return T

def solve_ramp(x1, x2, v1, v2, v_max, a_max):
    assert all(abs(v) <= v_max for v in [v1, v2])
    candidates = [
        solve_two_ramp(x1, x2, v1, v2, a_max, v_max=v_max),
        solve_two_ramp(x1, x2, v1, v2, -a_max, v_max=-v_max),
        solve_three_ramp(x1, x2, v1, v2, v_max, a_max),
        solve_three_ramp(x1, x2, v1, v2, -v_max, -a_max),
    ]
    candidates = [t for t in candidates if t is not None]
    if not candidates:
        return None
    return min(t for t in candidates)

def solve_multivariate_ramp(x1, x2, v1, v2, v_max, a_max):
    d = len(x1)
    durations = [solve_ramp(x1[i], x2[i], v1[i], v2[i], v_max[i], a_max[i]) for i in range(d)]
    # if any(t is None for t in durations):
    #     return None
    durations = [t for t in durations if t is not None]
    if not durations:
        return None
    return max(durations)

##################################################

def smooth(positions_curve, v_max, a_max, num=100):
    from scipy.interpolate import CubicHermiteSpline
    for _ in range(num):
        times = positions_curve.x
        velocities_curve = positions_curve.derivative()
        # ts = [times[0], times[-1]]
        # t1, t2 = positions_curve.x[0], positions_curve.x[-1]
        t1, t2 = np.random.uniform(times[0], times[-1], 2)
        if t1 > t2:
            t1, t2 = t2, t1
        ts = [t1, t2]

        x1, x2 = [positions_curve(t) for t in ts]
        v1, v2 = [velocities_curve(t) for t in ts]
        t = solve_multivariate_ramp(x1, x2, v1, v2, v_max, a_max)
        if t is None:
            continue
        #assert t is not None
        print(t, t2 - t1)

        i1, i2 = [min(range(len(times)), key=lambda i: times[i] >= t) for t in ts]
        new_times = np.concatenate([times[:i1+1], [t1, t2], times[i2:]])
        #new_times = [ts[0], ts[-1] + t]
        positions = [positions_curve(t) for t in new_times]
        velocities = [velocities_curve(t) for t in new_times]
        positions_curve = CubicHermiteSpline(new_times, positions, dydx=velocities)
    return positions_curve

def interpolate_path(path, velocity=1., kind='linear', **kwargs): # linear | slinear | quadratic | cubic
    from scipy.interpolate import interp1d, CubicHermiteSpline, make_interp_spline, CubicSpline
    #from numpy import polyfit
    waypoints = remove_redundant(path)
    waypoints = waypoints_from_path(waypoints)

    print(len(path), len(waypoints))
    differences = [0.] + [get_distance(*pair) / velocity for pair in get_pairs(waypoints)]
    times = np.cumsum(differences) / velocity
    print(times)
    #positions_curve = interp1d(times, waypoints, kind=kind, axis=0, **kwargs)
    #positions_curve = CubicSpline(times, waypoints, bc_type='clamped')
    velocities = [np.zeros(len(waypoint)) for waypoint in waypoints]
    positions_curve = CubicHermiteSpline(times, waypoints, dydx=velocities)
    velocities_curve = positions_curve.derivative()
    print([velocities_curve(t) for t in times])

    d = len(path[0])
    v_max = 5.*np.ones(d)
    a_max = v_max / 1.
    positions_curve = smooth(positions_curve, v_max, a_max, num=100)
    return positions_curve

def discretize_curve(positions_curve, time_step=1e-2):
    control_times = np.append(np.arange(
        positions_curve.x[0], positions_curve.x[-1], step=time_step), [positions_curve.x[-1]])
    #velocities_curve = positions_curve.derivative()
    control_positions = [positions_curve(control_time) for control_time in control_times]
    return control_times, control_positions

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
                #path = path[:1] + path[-2:]
                path = waypoints_from_path(path)
                add_path(viewer, path, color='green')
                curve = interpolate_path(path)
                _, path = discretize_curve(curve)
                add_path(viewer, path, color='red')

            if args.smooth:
                for path in paths:
                    extend_fn, roadmap = get_extend_fn(obstacles=obstacles)  # obstacles | []
                    smoothed = smooth_path(path, extend_fn, collision_fn, iterations=INF, max_time=args.time)
                    print('Smoothed distance_fn: {:.3f}'.format(compute_path_cost(smoothed, distance_fn)))
                    add_path(viewer, smoothed, color='red')
            user_input('Finish?')

if __name__ == '__main__':
    main()
