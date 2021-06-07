from __future__ import print_function

import numpy as np
import math
import argparse
import time
import random

from motion_planners.utils import get_delta
from .viewer import is_collision_free, \
    create_box, draw_environment, point_collides, sample_line, add_points, \
    add_roadmap, get_box_center, add_path, create_cylinder, contains
from ..utils import user_input, profiler, INF, compute_path_cost, get_distance, elapsed_time, interval_generator, \
    get_pairs, remove_redundant, waypoints_from_path, find
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

def get_sample_fn(region, obstacles=[], use_halton=True): #, check_collisions=False):
    # TODO: Gaussian sampling for narrow passages
    samples = []
    collision_fn, _ = get_collision_fn(region, obstacles)
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

def get_collision_fn(environment, obstacles):
    cfree = []

    def collision_fn(q):
        #time.sleep(1e-3)
        if not contains(q, environment):
            return True
        if point_collides(q, obstacles):
            return True
        cfree.append(q)
        return False

    return collision_fn, cfree

def get_extend_fn(environment, obstacles=[]):
    collision_fn, _ = get_collision_fn(environment, obstacles)
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

def get_distance_fn(weights):
    difference_fn = get_delta
    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn

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

def check_time(t):
    return not isinstance(t, complex) and (t >= 0.)

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

    sigma = +1 if a_max >= 0 else -1
    solutions = np.roots([
        T ** 2,
        sigma * (2 * T * (v1 + v2) + 4 * (x1 - x2)),
        -(v2 - v1) ** 2,
    ])
    a = solutions[0]
    t1 = (T + (v2 - v1) / a) / 2
    t2 = T - t1
    vp = v1 + a * t1

    print(solutions, a, t1, t2)
    print(x1 + v1 * t1 + 0.5 * a * t1 ** 2,  # + vp*t2 - 0.5*a*t2**2,
          x2 - v2 * t2 - 0.5 * a * t2 ** 2)
    print(v1 + a * t1,
          v2 + a * t2)
    input()

def solve_two_ramp(x1, x2, v1, v2, a_max, v_max=INF):
    #optimize_two_ramp(x1, x2, v1, v2, a_max)

    solutions = np.roots([
        a_max,
        2 * v1,
        (v1**2 - v2**2) / (2 * a_max) - (x2 - x1),
    ])
    solutions = [t for t in solutions if check_time(t)]
    #solutions = [t for t in solutions if t <= (v2 - v1) / a_max] # TODO: this constraint is strange
    solutions = [t for t in solutions if abs(v1 + t*a_max) <= abs(v_max)]
    if not solutions:
        return None
    t = min(solutions)
    T = t + 2 * (v1 - v2) / a_max
    if T < 0:
        return None
    return T
    #min_two_ramp(x1, x2, v1, v2, T, a_max, v_max=INF)

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

def solve_ramp(x1, x2, v1, v2, v_max, a_max):
    # TODO: handle infinite acceleration
    assert all(abs(v) <= v_max for v in [v1, v2])
    candidates = [
        solve_two_ramp(x1, x2, v1, v2, a_max, v_max=v_max),
        solve_two_ramp(x1, x2, v1, v2, -a_max, v_max=-v_max),
    ]
    if v_max < INF:
        candidates.extend([
            solve_three_ramp(x1, x2, v1, v2, v_max, a_max),
            solve_three_ramp(x1, x2, v1, v2, -v_max, -a_max),
        ])
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

def check_spline(spline, v_max=None, a_max=None, start=None, end=None):
    if (v_max is None) and (a_max is None):
        return True
    if start is None:
        start = 0
    if end is None:
        end = len(spline.x) - 1
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.html#scipy.interpolate.PPoly
    # polys = [np.poly1d([spline.c[c, 0, k] for c in range(spline.c.shape[0])]) # Decreasing order
    #         for k in range(spline.c.shape[-1])]
    # from numpy.polynomial.polynomial import Polynomial
    # polys = [Polynomial([spline.c[c, 0, k] for c in range(spline.c.shape[0])][-1])  # Increasing order
    #          for k in range(spline.c.shape[-1])]

    # TODO: take zersos of the velocity
    # _, v = find_max_velocity(spline, ord=INF)
    # _, a = find_max_acceleration(spline, ord=INF)
    # print(v, v_max, a, a_max)
    # input()

    signs = [+1, -1]
    for i in range(start, end):
        t0, t1 = spline.x[i], spline.x[i+1]
        t0, t1 = 0, (t1 - t0)
        boundary_ts = [t0, t1]
        for k in range(spline.c.shape[-1]):
            position_poly = np.poly1d([spline.c[c, i, k] for c in range(spline.c.shape[0])])
            # print([position_poly(t) for t in boundary_ts])
            # print([spline(t)[k] for t in boundary_ts])

            if v_max is not None:
                vel_poly = position_poly.deriv(m=1)
                if any(abs(vel_poly(t)) > v_max[k] for t in boundary_ts):
                    return False
                if any(not isinstance(r, complex) and (t0 <= r <= t1)
                       for s in signs for r in (vel_poly + s*np.poly1d([v_max[k]])).roots):
                    return False
            #a_max = None

            # TODO: reorder to check endpoints first
            if a_max is not None: # INF
                accel_poly = position_poly.deriv(m=2)
                #print([(accel_poly(t), a_max[k]) for t in boundary_ts])
                if any(abs(accel_poly(t)) > a_max[k] for t in boundary_ts):
                    return False
                # print([accel_poly(r) for s in signs for r in (accel_poly + s*np.poly1d([a_max[k]])).roots
                #        if isinstance(r, complex) and (t0 <= r <= t1)])
                if any(not isinstance(r, complex) and (t0 <= r <= t1)
                       for s in signs for r in (accel_poly + s*np.poly1d([a_max[k]])).roots):
                    return False
    return True

def smooth(start_positions_curve, v_max, a_max, collision_fn=lambda q: False, num=100, max_time=INF):
    start_time = time.time()
    if not check_spline(start_positions_curve, v_max, a_max):
        return None
    from scipy.interpolate import CubicHermiteSpline
    positions_curve = start_positions_curve
    for iteration in range(num):
        if elapsed_time(start_time) >= max_time:
            break
        times = positions_curve.x
        durations = [0.] + [t2 - t1 for t1, t2 in get_pairs(times)]
        positions = [positions_curve(t) for t in times]
        velocities_curve = positions_curve.derivative()
        velocities = [velocities_curve(t) for t in times]

        # ts = [times[0], times[-1]]
        # t1, t2 = positions_curve.x[0], positions_curve.x[-1]
        t1, t2 = np.random.uniform(times[0], times[-1], 2)
        if t1 > t2:
            t1, t2 = t2, t1

        ts = [t1, t2]
        i1 = find(lambda i: times[i] <= t1, reversed(range(len(times))))
        i2 = find(lambda i: times[i] >= t2, range(len(times)))
        assert i1 != i2

        x1, x2 = [positions_curve(t) for t in ts]
        v1, v2 = [velocities_curve(t) for t in ts]
        #assert all(abs(v) <= v_max for v in [v1, v2])
        new_positions = positions[:i1+1] + [x1, x2] + positions[i2:]
        new_velocities = velocities[:i1+1] + [v1, v2] + velocities[i2:]
        # if not all(np.less_equal(np.absolute(v), v_max).all() for v in new_velocities):
        #     continue

        max_t = t2 - t1
        #min_t = 0
        min_t = optimistic_time(x1, x2, v1, v2, v_max=v_max, a_max=a_max)
        #min_t = optimistic_time(x1, x2, v_max=v_max, a_max=a_max)
        # TODO: limit the distance/duration between these two points
        if min_t >= max_t:
            continue

        best_t = random.uniform(min_t, max_t)
        # best_t = solve_multivariate_ramp(x1, x2, v1, v2, v_max, a_max)
        # if best_t is None:
        #     continue
        #assert best_t is not None
        #print(best_t, t2 - t1)

        # current_t = t2 - t1
        # best_t = np.max(np.divide(np.absolute(x2 - x1), v_max))
        # #best_t = np.random.uniform(0, current_t)
        # best_t = np.random.uniform(best_t, current_t)

        # spline = CubicHermiteSpline([0, best_t], [x1, x2], dydx=[v1, v2])
        # if not check_spline(spline, v_max, a_max):
        #     continue

        # positions = [positions_curve(t) for t in times]
        # velocities = [velocities_curve(t) for t in times]
        new_durations = np.concatenate([
            durations[:i1+1], [t1 - times[i1], best_t, times[i2] - t2], durations[i2+1:]])
        assert len(new_durations) == (i1 + 1) + (len(durations) - i2) + 2
        print(new_durations)
        new_times = np.cumsum(new_durations)
        #new_times = [ts[0], ts[-1] + t]

        # TODO: splice in the new segment
        new_positions_curve = CubicHermiteSpline(new_times, new_positions, dydx=new_velocities)
        #new_positions_curve = CubicSpline(new_times, new_positions)
        print(iteration, new_positions_curve.x[-1], positions_curve.x[-1])
        if new_positions_curve.x[-1] >= positions_curve.x[-1]:
            continue

        #new_t1 = new_times[i1+1]
        #new_t2 = new_times[i1+2]
        #new_t2 = new_times[-(len(times) - i2 + 1)]
        # new_velocities_curve = new_positions_curve.derivative()
        # print(v2, new_velocities_curve(new_t2))

        if not check_spline(new_positions_curve, v_max, a_max):
            continue

        _, samples = discretize_curve(new_positions_curve, max_velocities=v_max)
        #_, samples = discretize_curve(new_positions_curve, start_t=new_times[i1+1], end_t=new_times[-(len(times) - i2 + 1)])
        if any(map(collision_fn, samples)):
            continue
        positions_curve = new_positions_curve
    print(start_positions_curve.x[-1], positions_curve.x[-1])

    return positions_curve

V_MAX = 5.*np.ones(2)
A_MAX = V_MAX / 2.

def optimistic_time(x1, x2, v1=None, v2=None, v_max=None, a_max=None):
    d = len(x1)
    if v_max is None:
        v_max = np.full(d, INF)
    if a_max is None:
        a_max = np.full(d, INF)
    lower_bounds = [
        # Instantaneously accelerate
        np.linalg.norm(np.divide(np.subtract(x2, x1), v_max), ord=INF),
    ]
    if (v1 is not None) and (v2 is not None):
        lower_bounds.extend([
            np.linalg.norm(np.divide(np.subtract(v2, v1), a_max), ord=INF),
        ])
    return max(lower_bounds)

def conservative(x1, x2, v_max, a_max, min_t=INF): # v1=0., v2=0.,
    from scipy.interpolate import PPoly
    # TODO: switch time
    if x1 > x2:
        return conservative(x2, x1, v_max, a_max, min_t=min_t)
    assert x2 >= x1 and (v_max >= 0) and (a_max >= 0)
    v1 = v2 = 0.
    d = (x2 - x1)
    x_half = (x1 + x2) / 2.

    position_curve = np.poly1d([0.5*a_max, v1, x1])
    velocity_curve = position_curve.deriv()
    t_half = filter_times((position_curve - np.poly1d([x_half])).roots)
    # solutions = np.roots([
    #     0.5*a_max,
    #     v1,
    #     x1 - x_half,
    # ])
    if (t_half is not None) and (velocity_curve(t_half) <= v_max):
        T = 2*t_half
        times = [0., t_half, T]
        c = np.zeros([3, len(times) - 1])
        c[:, 0] = list(position_curve)
        c[:, 1] = [-0.5*a_max, velocity_curve(t_half), position_curve(t_half)]
        spline = PPoly(c=c, x=times)
        return spline.x[-1]

    t_ramp = filter_times((velocity_curve - np.poly1d([v_max])).roots)
    assert t_ramp is not None
    x_ramp = position_curve(t_ramp)
    d_ramp = (x_ramp - x1)
    d_hold = d - 2*d_ramp
    t_hold = abs(d_hold / v_max)
    T = 2*t_ramp + t_hold

    times = [0., t_ramp, t_ramp + t_hold, T]
    c = np.zeros([3, len(times) - 1])
    c[:, 0] = list(position_curve)
    c[:, 1] = [0., velocity_curve(t_half), position_curve(t_half)]
    c[:, 2] = [-0.5 * a_max, v_max, x_ramp + d_hold]
    spline = PPoly(c=c, x=times) # TODO: extend
    return spline.x[-1]

def multivariate_conservative(x1, x2, v_max, a_max):
    d = len(x1)
    return max([conservative(x1[k], x2[k], v_max[k], a_max[k]) for k in range(d)])

def get_max_velocity(velocities, norm=INF):
    return np.linalg.norm(velocities, ord=norm)

def retime_path(path, velocity=get_max_velocity(V_MAX), **kwargs):
    from scipy.interpolate import CubicHermiteSpline
    d = len(path[0])
    # v_max = 5.*np.ones(d)
    # a_max = v_max / 1.
    v_max, a_max = V_MAX, A_MAX

    waypoints = remove_redundant(path)
    waypoints = waypoints_from_path(waypoints)
    differences = [0.] + [get_distance(*pair) / velocity for pair in get_pairs(waypoints)]
    # differences = [0.] + [solve_multivariate_ramp(x1, x2, np.zeros(d), np.zeros(d), v_max, a_max)
    #                      for x1, x2 in get_pairs(waypoints)]
    # differences = [0.] + [multivariate_conservative(x1, x2, v_max, a_max)
    #                       for x1, x2 in get_pairs(waypoints)]
    times = np.cumsum(differences)
    velocities = [np.zeros(len(waypoint)) for waypoint in waypoints]
    positions_curve = CubicHermiteSpline(times, waypoints, dydx=velocities)
    #positions_curve = interp1d(times, waypoints, kind='quadratic', axis=0) # Cannot differentiate

    # positions_curve = smooth(positions_curve,
    #                          v_max=None, a_max=None,
    #                          #v_max=v_max, a_max=a_max,
    #                          **kwargs)
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

def optimize(objective, lower, upper, num=10, max_time=INF, verbose=False, **kwargs):
    # https://www.cvxpy.org/examples/basic/socp.html
    from scipy.optimize import minimize #, line_search, brute, basinhopping, minimize_scalar
    start_time = time.time()
    best_t, best_f = None, INF
    bounds = list(zip(lower, upper))
    for iteration in range(num):
        if elapsed_time(start_time) >= max_time:
            break
        x0 = np.random.uniform(lower, upper)
        result = minimize(objective, x0=x0, bounds=bounds, **kwargs) # method=None, jac=None,
        if result.fun < best_f:
            best_t, best_f = result.x, result.fun
            if verbose:
                print(iteration, x0, result.x, result.fun) # objective(result.x)
    return best_t, best_f

def find_max_curve(curve, start_t=None, end_t=None, ord=2, **kwargs):
    if start_t is None:
        start_t = curve.x[0]
    if end_t is None:
        end_t = curve.x[-1]
    objective = lambda t: -np.linalg.norm(curve(t), ord=ord) # 2 | INF
    #objective = lambda t: -np.linalg.norm(curve(t), ord=2)**2 # t[0]
    #accelerations_curve = positions_curve.derivative() # TODO: ValueError: failed in converting 7th argument `g' of _lbfgsb.setulb to C/Fortran array
    #grad = lambda t: np.array([-2*sum(accelerations_curve(t))])
    #result = minimize_scalar(objective, method='bounded', bounds=(start_t, end_t)) #, options={'disp': False})

    #print(max(-objective(t) for t in curve.x))
    best_t, best_f = optimize(objective, lower=[start_t], upper=[end_t], **kwargs)
    best_t, best_f = best_t[0], -best_f
    return best_t, best_f

def find_max_velocity(positions_curve, **kwargs):
    velocities_curve = positions_curve.derivative(nu=1)
    #return find_max_curve(velocities_curve, **kwargs)
    accelerations_curve = velocities_curve.derivative(nu=1)
    times = list(velocities_curve.x)
    critical = accelerations_curve.roots(discontinuity=True)
    for k in range(critical.shape[0]):
        times.extend(critical[k])
    fn = lambda t: np.linalg.norm(velocities_curve(t), ord=INF)
    max_time = max(times, key=fn)
    return max_time, fn(max_time)

def find_max_acceleration(positions_curve, **kwargs):
    accelerations_curve = positions_curve.derivative(nu=2)
    return find_max_curve(accelerations_curve, **kwargs)

def filter_proximity(times, positions, resolution=0.):
    assert len(times) == len(positions)
    new_times = []
    new_positions = []
    for t, position in zip(times, positions):
        if not new_positions or (get_distance(new_positions[-1], position) >= resolution): # TODO: add first before exceeding
            new_times.append(t)
            new_positions.append(position)
    # new_times.append(times[-1])
    # new_positions.append(positions_curve(new_times[-1]))
    return new_times, new_positions

def time_discretize_curve(positions_curve, start_t=None, end_t=None, max_velocities=None, time_step=1e-2):
    if start_t is None:
        start_t = positions_curve.x[0]
    if end_t is None:
        end_t = positions_curve.x[-1]
    assert start_t < end_t

    ord = INF
    d = len(positions_curve(start_t))
    resolution = 2e-2
    resolutions = resolution*np.ones(d)
    if max_velocities is None:
        max_t, max_v = find_max_velocity(positions_curve, start_t=start_t, end_t=end_t, ord=ord, num=100)
        #max_t, max_v = INF, np.linalg.norm(V_MAX)
        time_step = resolution / max_v
        print('Max velocity: {:.3f} (at time {:.3f}) | Limit: {:.3f} | Step: {:.3f} | Duration: {:.3f}'.format(
            max_v, max_t, np.linalg.norm(V_MAX, ord=ord), time_step, positions_curve.x[-1])) # 2 | INF
        #input()
    else:
        time_step = np.min(np.divide(resolutions, max_velocities))

    times = np.append(np.arange(start_t, end_t, step=time_step), [end_t])
    #times = positions_curve.x
    #velocities_curve = positions_curve.derivative()
    positions = [positions_curve(t) for t in times]
    times, positions = filter_proximity(times, positions, resolution)

    return times, positions

discretize_curve = time_discretize_curve

def derivative_discretize_curve(positions_curve, start_t=None, end_t=None, resolution=1e-2, time_step=1e-3):
    d = positions_curve.c.shape[-1]
    resolutions = resolution*np.ones(d)
    if start_t is None:
        start_t = positions_curve.x[0]
    if end_t is None:
        end_t = positions_curve.x[-1]
    assert start_t < end_t
    velocities_curve = positions_curve.derivative()
    #acceleration_curve = velocities_curve.derivative()
    times = [start_t]
    while True:
        velocities = velocities_curve(times[-1])
        dt = min(np.divide(resolutions, np.absolute(velocities)))
        dt = min(dt, time_step)
        new_time = times[-1] + dt
        if new_time > end_t:
            break
        times.append(new_time)
    times.append(end_t)
    positions = [positions_curve(control_time) for control_time in times]
    # TODO: distance between adjacent positions
    return times, positions

def integral_discretize_curve(positions_curve, start_t=None, end_t=None, resolution=1e-2):
    #from scipy.integrate import quad
    if start_t is None:
        start_t = positions_curve.x[0]
    if end_t is None:
        end_t = positions_curve.x[-1]
    assert start_t < end_t
    distance_curve = positions_curve.antiderivative()
    #distance = positions_curve.integrate(a, b)
    # TODO: compute a total distance curve
    raise NotImplementedError()

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
                _, path = discretize_curve(curve)
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
