from __future__ import print_function

import numpy as np
import math
import argparse
import time
import random

from motion_planners.retime import check_time, parabolic_val, curve_from_controls
from motion_planners.tkinter.discretize import time_discretize_curve, V_MAX, A_MAX
from motion_planners.tkinter.limits import get_max_velocity
from motion_planners.tkinter.smooth import smooth
from .samplers import get_sample_fn, get_collision_fn, get_extend_fn, get_distance_fn
from .viewer import create_box, draw_environment, add_points, \
    add_roadmap, get_box_center, add_path, create_cylinder
from ..utils import user_input, profiler, INF, compute_path_cost, get_distance, elapsed_time, get_pairs, remove_redundant, waypoints_from_path
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
    # TODO: https://ompl.kavrakilab.org/planners.html
]

##################################################

def optimize_two_ramp(x1, x2, v1, v2, a_max, v_max=INF):
    from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

    a_max = abs(a_max)
    v_max = abs(v_max)
    d = x2 - x1
    def objective(args):
        t1, t2, vp = args
        return t1 + t2
    def equality(args):
        t1, t2, vp = args
        return t1*(v1 + vp) / 2. + t2*(vp + v2) / 2.
    constraints = [
        # TODO: sample value and then optimize
        LinearConstraint(A=[-a_max, 0, 1], lb=-np.inf, ub=v1, keep_feasible=False),
        LinearConstraint(A=[+a_max, 0, 1], lb=v1, ub=+np.inf, keep_feasible=False),
        LinearConstraint(A=[0, -a_max, 1], lb=-np.inf, ub=v2, keep_feasible=False),
        LinearConstraint(A=[0, +a_max, 1], lb=v2, ub=+np.inf, keep_feasible=False),
        NonlinearConstraint(fun=equality, lb=d-1e-2, ub=d+1e-2, keep_feasible=False),
    ]
    bounds = [
        # TODO: time bound based on moving to a stop
        (0, np.inf),
        (0, np.inf),
        (-v_max, +v_max),
    ]
    guess = np.zeros(3)
    result = minimize(objective, x0=guess, bounds=bounds, constraints=constraints)
    print(result)
    input()


def filter_times(times):
    valid_times = list(filter(check_time, times))
    if not valid_times:
        return None
    return min(valid_times)


def min_two_ramp(x1, x2, v1, v2, T, a_max, v_max=INF):
    # from numpy.linalg import solve
    # from scipy.optimize import linprog
    # result = linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
    #                  method='interior-point', callback=None, options=None, x0=None)

    print()
    print(x1, x2, v1, v2, T, a_max)
    sign = +1 if a_max >= 0 else -1
    eqn = np.poly1d([
        T ** 2, # a**2
        sign * (2 * T * (v1 + v2) + 4 * (x1 - x2)), # a
        -(v2 - v1) ** 2, # 1
    ])
    candidates = []
    for a in np.roots(eqn): # eqn.roots
        if isinstance(a, complex) or (a == 0):
            continue
        ts = (T + (v2 - v1) / a) / 2.
        if not (0 <= ts <= T):
            continue
        vs = v1 + a * ts
        if abs(vs) > abs(v_max):
            continue
        candidates.append(a)
    if not candidates:
        return None

    a = min(candidates)
    ts = (T + (v2 - v1) / a) / 2.
    durations = [ts, T - ts]
    accels = [sign*a, -sign*a]
    p_curve = curve_from_controls(durations, accels, t0=0., x0=x1, v0=v1)
    #return p_curve

    end_times = np.append(p_curve.x[:1], p_curve.x[-1:])
    v_curve = p_curve.derivative()
    # print([0., T], end_times)
    # print([x1, x2], [float(p_curve(t)) for t in end_times])
    # print([v1, v2], [float(v_curve(t)) for t in end_times])
    assert np.allclose([0., T], end_times)
    assert np.allclose([x1, x2], [float(p_curve(t)) for t in end_times])
    assert np.allclose([v1, v2], [float(v_curve(t)) for t in end_times])
    return p_curve

def solve_one_ramp(x1, x2, v_max=INF):
    return abs(x2 - x1) / abs(v_max)

def solve_zero_ramp(x1, x2, v_max=INF, a_max=INF):
    assert (v_max >= 0) and (a_max >= 0)
    #v_max = abs(x2 - x1) / abs(v_max)
    d = abs(x2 - x1)
    t_accel = math.sqrt(d / a_max) # 1/2.*a*t**2 = d/2.
    if a_max*t_accel <= v_max:
        T = 2*t_accel
        return T
    t1 = t3 = (v_max - 0) / a_max
    t2 = (d - 2 * parabolic_val(t1, a=a_max)) / v_max
    T = t1 + t2 + t3
    return T

def solve_two_ramp(x1, x2, v1, v2, a_max, v_max=INF):
    #optimize_two_ramp(x1, x2, v1, v2, a_max)
    solutions = np.roots([
        a_max, # t**2
        2 * v1, # t
        (v1**2 - v2**2) / (2 * a_max) + (x1 - x2), # 1
    ])
    print()
    print(x1, x2, v1, v2, a_max, v_max)
    solutions = [t for t in solutions if check_time(t)]
    print(solutions)
    solutions = [t for t in solutions if 0 <= t <= abs(v2 - v1) / abs(a_max)] # TODO: this constraint is strange
    print(solutions)
    solutions = [t for t in solutions if abs(v1 + t*a_max) <= abs(v_max)]
    print(solutions)
    if not solutions:
        return None
    t = min(solutions)
    T = t + 2 * (v1 - v2) / a_max
    if T < 0:
        return None
    min_two_ramp(x1, x2, v1, v2, T, a_max, v_max=v_max)
    return T

def min_three_ramp(x1, x2, v1, v2, v_max, a_max, T):
    #print(tp1, tp2, tl)
    a = (v_max**2 - abs(v_max)*(v1 + v2) + (v1**2 + v2**2)/2) / (T*abs(v_max) - (x2 - x1))
    tp1 = (v_max - v1) / a
    tp2 = (v2 - v_max) / a
    tl = (v2 ** 2 + v1 ** 2 - 2 * abs(v_max) ** 2) / (2 * v_max * a) + (x2 - x1) / v_max
    print(tp1, tp2, tl)
    input()

def solve_three_ramp(x1, x2, v1, v2, v_max, a_max):
    # http://motion.pratt.duke.edu/papers/icra10-smoothing.pdf
    # https://github.com/Puttichai/parabint/blob/2662d4bf0fbd831cdefca48863b00d1ae087457a/parabint/optimization.py
    # TODO: minimum-switch-time constraint
    #assert np.positive(v_max).all() and np.positive(a_max).all()
    # P+L+P-
    tp1 = (v_max - v1) / a_max
    tp2 = (v2 - v_max) / a_max
    tl = (v2 ** 2 + v1 ** 2 - 2 * abs(v_max) ** 2) / (2 * v_max * a_max) + (x2 - x1) / v_max
    ts = [tp1, tl, tp2]
    if any(t < 0 for t in ts):
        return None
    T = sum(ts)
    #min_three_ramp(x1, x2, v1, v2, v_max, a_max, T)
    return T

def solve_ramp(x1, x2, v1, v2, v_max=INF, a_max=INF, min_t=0.):
    # TODO: handle infinite acceleration
    assert (v_max >= 0.) and (a_max >= 0.)
    assert all(abs(v) <= v_max for v in [v1, v2])
    if (v_max == INF) and (a_max == INF):
        T = 0
        return min(min_t, T) # TODO: throw an error
    if a_max == INF:
        T = solve_one_ramp(x1, x2, v_max=v_max)
        return min(min_t, T)

    candidates = [
        solve_two_ramp(x1, x2, v1, v2, a_max, v_max=v_max),
        solve_two_ramp(x1, x2, v1, v2, -a_max, v_max=-v_max),
    ]
    if v_max != INF:
        candidates.extend([
            solve_three_ramp(x1, x2, v1, v2, v_max, a_max),
            solve_three_ramp(x1, x2, v1, v2, -v_max, -a_max),
        ])
    candidates = [t for t in candidates if t is not None]
    assert candidates
    if not candidates:
        return None
    T = min(t for t in candidates)
    return min(min_t, T)

def solve_multivariate_ramp(x1, x2, v1, v2, v_max, a_max):
    d = len(x1)
    durations = [solve_ramp(x1[i], x2[i], v1[i], v2[i], v_max[i], a_max[i]) for i in range(d)]
    # if any(t is None for t in durations):
    #     return None
    durations = [t for t in durations if t is not None]
    print(durations)
    input()
    if not durations:
        return None
    return max(durations)

##################################################


def ramp_planner():
    pass

def conservative(x1, x2, v_max, a_max, min_t=INF): # v1=0., v2=0.,
    # TODO: switch time
    #if x1 > x2:
    #   return conservative(x2, x1, v_max, a_max, min_t=min_t)
    assert (v_max >= 0) and (a_max >= 0) # and (x2 >= x1)
    sign = +1 if x2 >= x1 else -1
    v1 = v2 = 0.
    x_half = (x1 + x2) / 2.

    position_curve = np.poly1d([sign*0.5*a_max, v1, x1])
    velocity_curve = position_curve.deriv()
    t_half = filter_times((position_curve - np.poly1d([x_half])).roots)
    # solutions = np.roots([
    #     0.5*a_max,
    #     v1,
    #     x1 - x_half,
    # ])
    if (t_half is not None) and (abs(velocity_curve(t_half)) <= v_max):
        # TODO: could separate out
        durations = [t_half, t_half]
        accels = [sign * a_max, -sign * a_max]
        spline = curve_from_controls(durations, accels, t0=0., x0=x1, v0=v1)
        # T = 2*t_half
        # times = [0., t_half, T]
        # c = np.zeros([3, len(times) - 1])
        # c[:, 0] = list(position_curve)
        # c[:, 1] = [-0.5*accels[1], velocity_curve(t_half), position_curve(t_half)]
        # spline = PPoly(c=c, x=times)
        return spline.x[-1]

    t_ramp = filter_times((velocity_curve - np.poly1d([sign*v_max])).roots)
    assert t_ramp is not None
    x_ramp = position_curve(t_ramp)
    d = abs(x2 - x1)
    d_ramp = abs(x_ramp - x1)
    d_hold = d - 2*d_ramp
    t_hold = abs(d_hold / v_max)

    durations = [t_ramp, t_hold, t_ramp]
    accels = [sign * a_max, 0., -sign * a_max]
    spline = curve_from_controls(durations, accels, t0=0., x0=x1, v0=v1)

    # T = 2*t_ramp + t_hold
    # times = [0., t_ramp, t_ramp + t_hold, T]
    # c = np.zeros([3, len(times) - 1])
    # c[:, 0] = list(position_curve)
    # c[:, 1] = [0.5 * accels[1], velocity_curve(t_ramp), position_curve(t_ramp)]
    # c[:, 2] = [0.5 * accels[2], velocity_curve(t_ramp), position_curve(t_ramp) + velocity_curve(t_ramp)*t_hold]
    # spline = PPoly(c=c, x=times) # TODO: extend
    return spline.x[-1]


def retime_path(path, velocity=get_max_velocity(V_MAX), **kwargs):
    from scipy.interpolate import CubicHermiteSpline
    d = len(path[0])
    # v_max = 5.*np.ones(d)
    # a_max = v_max / 1.
    v_max, a_max = V_MAX, A_MAX

    waypoints = remove_redundant(path)
    waypoints = waypoints_from_path(waypoints)
    #durations = [0.] + [get_distance(*pair) / velocity for pair in get_pairs(waypoints)]
    durations = [0.] + [max(conservative(x1[k], x2[k], v_max=v_max[k], a_max=a_max[k]) for k in range(d))
                        for x1, x2 in get_pairs(waypoints)] # solve_zero_ramp | conservative
    print(durations)
    durations = [0.] + [max(solve_zero_ramp(x1[k], x2[k], v_max=v_max[k], a_max=a_max[k]) for k in range(d))
                        for x1, x2 in get_pairs(waypoints)] # solve_zero_ramp | conservative
    print(durations)
    #durations = [0.] + [solve_multivariate_ramp(x1, x2, np.zeros(d), np.zeros(d), v_max, a_max)
    #                     for x1, x2 in get_pairs(waypoints)]

    times = np.cumsum(durations)
    velocities = [np.zeros(len(waypoint)) for waypoint in waypoints]
    positions_curve = CubicHermiteSpline(times, waypoints, dydx=velocities)
    #positions_curve = interp1d(times, waypoints, kind='quadratic', axis=0) # Cannot differentiate

    positions_curve = smooth(positions_curve,
                             #v_max=None, a_max=None,
                             v_max=v_max, a_max=a_max,
                             **kwargs)
    return positions_curve

def interpolate_path(path, velocity=1., kind='linear', **kwargs): # linear | slinear | quadratic | cubic
    from scipy.interpolate import CubicHermiteSpline
    #from numpy import polyfit
    waypoints = remove_redundant(path)
    waypoints = waypoints_from_path(waypoints)

    #print(len(path), len(waypoints))
    differences = [0.] + [get_distance(*pair) / velocity for pair in get_pairs(waypoints)]
    times = np.cumsum(differences) / velocity
    #positions_curve = interp1d(times, waypoints, kind=kind, axis=0, **kwargs)
    #positions_curve = CubicSpline(times, waypoints, bc_type='clamped')
    velocities = [np.zeros(len(waypoint)) for waypoint in waypoints]
    positions_curve = CubicHermiteSpline(times, waypoints, dydx=velocities)
    #velocities_curve = positions_curve.derivative()
    #print([velocities_curve(t) for t in times])

    d = len(path[0])
    v_max = 5.*np.ones(d)
    a_max = v_max / 1.
    positions_curve = smooth(positions_curve, v_max, a_max)
    return positions_curve


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

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

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
    environment = regions['env']

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
            collision_fn, cfree = get_collision_fn(environment, obstacles)
            sample_fn, samples = get_sample_fn(environment, obstacles=[], use_halton=False) # obstacles
            extend_fn, roadmap = get_extend_fn(environment, obstacles=obstacles)  # obstacles | []

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
                                   iterations=INF, max_time=args.time)
            elif args.algorithm == 'birrt':
                path = birrt(start, goal, distance_fn=distance_fn, sample_fn=sample_fn,
                             extend_fn=extend_fn, collision_fn=collision_fn,
                             restarts=2, iterations=INF, max_time=args.time, smooth=100)
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
                #curve = interpolate_path(path) # , collision_fn=collision_fn)
                curve = retime_path(path, collision_fn=collision_fn)
                _, path = time_discretize_curve(curve)
                #add_points(viewer, [curve(t) for t in curve.x])
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
