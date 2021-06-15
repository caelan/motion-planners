import random
import time

import numpy as np

from .linear import find_lower_bound
from .limits import check_spline
from .discretize import time_discretize_curve
from .parabolic import solve_multi_poly, solve_multivariate_ramp
from .retime import EPSILON, trim, spline_duration, append_polys
from ..utils import INF, elapsed_time, get_pairs, find, default_selector


def get_curve_collision_fn(collision_fn=lambda q: False, max_velocities=None, max_accelerations=None): # a_max

    def curve_collision_fn(curve, t0=None, t1=None):
        # TODO: stage the function to check all the easy things like joint limits first
        if curve is None:
            return True
        # TODO: can exactly compute limit violations
        # if not check_spline(curve, v_max=max_velocities, a_max=None, verbose=False,
        #                     #start_t=t0, end_t=t1,
        #                     ):
        #     return True
        _, samples = time_discretize_curve(curve, verbose=False,
                                           start_t=t0, end_t=t1,
                                           #max_velocities=v_max,
                                           )
        if any(map(collision_fn, default_selector(samples))):
           return True
        return False
    return curve_collision_fn

##################################################

def smooth_curve(start_curve, v_max, a_max, curve_collision_fn,
                 sample=True, intermediate=True, cubic=True, refit=True, num=1000, min_improve=0., max_time=INF):
    # TODO: rename smoothing.py to shortcutting.py
    # TODO: default v_max and a_max
    assert (num < INF) or (max_time < INF)
    assert refit or intermediate
    # TODO: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    #from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline, LSQBivariateSpline
    from scipy.interpolate import CubicHermiteSpline
    start_time = time.time()

    if curve_collision_fn(start_curve, t0=None, t1=None):
        #return None
        return start_curve
    curve = start_curve
    for iteration in range(num):
        if elapsed_time(start_time) >= max_time:
            break
        times = curve.x
        durations = [0.] + [t2 - t1 for t1, t2 in get_pairs(times)] # includes start
        positions = [curve(t) for t in times]
        velocities = [curve(t, nu=1) for t in times]

        # ts = [times[0], times[-1]]
        # t1, t2 = curve.x[0], curve.x[-1]
        t1, t2 = np.random.uniform(times[0], times[-1], 2) # TODO: sample based on position
        if t1 > t2: # TODO: minimum distance from a knot
            t1, t2 = t2, t1

        ts = [t1, t2]
        i1 = find(lambda i: times[i] <= t1, reversed(range(len(times)))) # index before t1
        i2 = find(lambda i: times[i] >= t2, range(len(times))) # index after t2
        assert i1 != i2

        local_positions = [curve(t) for t in ts]
        local_velocities = [curve(t, nu=1) for t in ts]
        #print(local_velocities, v_max)
        assert all(np.less_equal(np.absolute(v), v_max + EPSILON).all() for v in local_velocities)
        #if any(np.greater(np.absolute(v), v_max).any() for v in local_velocities):
        #    continue # TODO: do the same with collisions
        x1, x2 = local_positions
        v1, v2 = local_velocities

        #min_t = 0
        min_t = find_lower_bound(x1, x2, v1, v2, v_max=v_max, a_max=a_max)
        #min_t = optimistic_time(x1, x2, v_max=v_max, a_max=a_max)
        current_t = (t2 - t1) - min_improve
        if min_t >= current_t: # TODO: also limit the distance/duration between these two points
            continue

        #best_t = min_t
        if sample:
            max_t = current_t
            ramp_t = solve_multivariate_ramp(x1, x2, v1, v2, v_max, a_max)
            ramp_t = INF if ramp_t is None else ramp_t
            max_t = min(max_t, ramp_t)
            best_t = random.uniform(min_t, max_t)
        else:
            best_t = solve_multivariate_ramp(x1, x2, v1, v2, v_max, a_max)
        if (best_t is None) or (best_t >= current_t):
            continue
        #best_t += 1e-3
        #print(min_t, best_t, current_t)
        local_durations = [t1 - times[i1], best_t, times[i2] - t2]
        #local_times = [0, best_t]
        local_times = [t1, (t1 + best_t)] # Good if the collision function is time sensitive

        if intermediate:
            if cubic:
                local_curve = CubicHermiteSpline(local_times, local_positions, dydx=local_velocities)
            else:
                local_curve = solve_multi_poly(times=local_times, positions=local_positions,
                                               velocities=local_velocities,
                                               v_max=v_max, a_max=a_max)
            if (local_curve is None) or (spline_duration(local_curve) >= current_t) \
                    or curve_collision_fn(local_curve, t0=None, t1=None):
                continue
            # print(new_curve.hermite_spline().c[0,...])
            local_positions = [local_curve(x) for x in local_curve.x]
            local_velocities = [local_curve(x, nu=1) for x in local_curve.x]
            local_durations = [t1 - times[i1]] + [x - local_curve.x[0]
                                                    for x in local_curve.x[1:]] + [times[i2] - t2]

        if refit:
            new_durations = np.concatenate([
                durations[:i1 + 1], local_durations, durations[i2 + 1:]])
            # assert len(new_durations) == (i1 + 1) + (len(durations) - i2) + 2
            new_times = np.cumsum(new_durations)
            # new_times = [new_times[0]] + [t2 for t1, t2 in get_pairs(new_times) if t2 > t1]
            new_positions = positions[:i1 + 1] + local_positions + positions[i2:]
            new_velocities = velocities[:i1 + 1] + local_velocities + velocities[i2:]
            # if not all(np.less_equal(np.absolute(v), v_max).all() for v in new_velocities):
            #    continue
            if cubic:
                # new_curve = CubicSpline(new_times, new_positions)
                new_curve = CubicHermiteSpline(new_times, new_positions, dydx=new_velocities)
            else:
                new_curve = solve_multi_poly(new_times, new_positions, new_velocities, v_max, a_max)
            if (new_curve is None) or (spline_duration(new_curve) >= spline_duration(curve)) \
                    or not check_spline(new_curve, v_max, a_max) or \
                    (not intermediate and curve_collision_fn(new_curve, t0=None, t1=None)):
                continue
        else:
            assert intermediate
            # print(curve.x)
            # print(curve.c[...,0])
            # pre_curve = trim(curve, end=t1)
            # post_curve = trim(curve, start=t1)
            # curve = append_polys(pre_curve, post_curve)
            # print(curve.x)
            # print(curve.c[...,0])

            # print(new_curve.x)
            # print(new_curve.c[...,0])
            pre_curve = trim(curve, end=t1)
            post_curve = trim(curve, start=t2)
            new_curve = append_polys(pre_curve, local_curve, post_curve) # TODO: the numerics are throwing this off?
            # print(new_curve.x)
            # print(new_curve.c[...,0])
            #assert(not curve_collision_fn(new_curve, t0=None, t1=None))
            if (spline_duration(new_curve) >= spline_duration(curve)) or \
                    not check_spline(new_curve, v_max, a_max):
                continue
        print('Iterations: {} | Current time: {:.3f} | New time: {:.3f} | Elapsed time: {:.3f}'.format(
            iteration, spline_duration(curve), spline_duration(new_curve), elapsed_time(start_time)))
        curve = new_curve
    print('Iterations: {} | Start time: {:.3f} | End time: {:.3f} | Elapsed time: {:.3f}'.format(
        num, spline_duration(start_curve), spline_duration(curve), elapsed_time(start_time)))
    check_spline(curve, v_max, a_max)
    return curve
