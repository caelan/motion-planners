import random
import time

import numpy as np

from motion_planners.tkinter.limits import check_spline
from motion_planners.tkinter.discretize import time_discretize_curve
from motion_planners.parabolic import solve_multi_poly
from motion_planners.utils import INF, elapsed_time, get_pairs, find

def find_lower_bound(x1, x2, v1=None, v2=None, v_max=None, a_max=None):
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

##################################################

def smooth_curve(start_positions_curve, v_max, a_max, collision_fn=lambda q: False, num=100, max_time=INF):
    # TODO: rename smoothing.py to shortcutting.py
    start_time = time.time()
    if not check_spline(start_positions_curve, v_max, a_max):
        #return None
        return start_positions_curve
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
        min_t = find_lower_bound(x1, x2, v1, v2, v_max=v_max, a_max=a_max)
        #min_t = optimistic_time(x1, x2, v_max=v_max, a_max=a_max)
        if min_t >= max_t: # TODO: limit the distance/duration between these two points
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
        new_times = np.cumsum(new_durations)
        #new_times = [ts[0], ts[-1] + t]

        # TODO: splice in the new segment
        #new_positions_curve = CubicSpline(new_times, new_positions)
        #new_positions_curve = CubicHermiteSpline(new_times, new_positions, dydx=new_velocities)
        new_positions_curve = solve_multi_poly(new_times, new_positions, new_velocities, v_max, a_max)
        if new_positions_curve is None:
            continue
        print(iteration, new_positions_curve.x[-1], positions_curve.x[-1])
        if new_positions_curve.x[-1] >= positions_curve.x[-1]:
            continue

        #new_t1 = new_times[i1+1]
        #new_t2 = new_times[i1+2]
        #new_t2 = new_times[-(len(times) - i2 + 1)]
        # new_velocities_curve = new_positions_curve.derivative()
        # print(v2, new_velocities_curve(new_t2))

        # if not check_spline(new_positions_curve, v_max, a_max):
        #     continue

        _, samples = time_discretize_curve(new_positions_curve, max_velocities=v_max)
        #_, samples = time_discretize_curve(new_positions_curve, start_t=new_times[i1+1], end_t=new_times[-(len(times) - i2 + 1)])
        if any(map(collision_fn, samples)):
            continue
        positions_curve = new_positions_curve
    print('Start time: {:.3f} | End time: {:.3f} | Iterations: {} | Elapsed time: {:.3f}'.format(
        start_positions_curve.x[-1], positions_curve.x[-1], num, elapsed_time(start_time)))
    check_spline(positions_curve, v_max, a_max)
    return positions_curve
