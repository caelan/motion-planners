import numpy as np

from motion_planners.tkinter.limits import find_max_velocity, find_max_acceleration
from motion_planners.retime import spline_start, spline_end

from motion_planners.utils import get_distance, INF


V_MAX = 1.*np.ones(2)
A_MAX = (V_MAX - 0.) / (0.25 - 0.)
#V_MAX = INF*np.ones(2)
#A_MAX = 1e6*np.ones(2)

##################################################

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

##################################################

def time_discretize_curve(positions_curve, start_t=None, end_t=None, max_velocities=None, verbose=True, time_step=1e-2):
    if start_t is None:
        start_t = spline_start(positions_curve)
    if end_t is None:
        end_t = spline_end(positions_curve)
    start_t = max(start_t, spline_start(positions_curve))
    end_t = min(end_t, spline_end(positions_curve))
    assert start_t <= end_t

    norm = INF
    d = len(positions_curve(start_t))
    resolution = 2e-2
    resolutions = resolution*np.ones(d)
    if max_velocities is None:
        # TODO: adjust per trajectory segment
        v_max_t, max_v = find_max_velocity(positions_curve, start_t=start_t, end_t=end_t, norm=norm)
        a_max_t, max_a = find_max_acceleration(positions_curve, start_t=start_t, end_t=end_t, norm=norm)
        #v_max_t, max_v = INF, np.linalg.norm(V_MAX)
        time_step = resolution / max_v
        if verbose:
            print('Max velocity: {:.3f}/{:.3f} (at time {:.3f}) | Max accel: {:.3f}/{:.3f} (at time {:.3f}) | '
                  'Step: {:.3f} | Duration: {:.3f}'.format(
                max_v, np.linalg.norm(V_MAX, ord=norm), v_max_t, max_a, np.linalg.norm(A_MAX, ord=norm), a_max_t,
                time_step, positions_curve.x[-1])) # 2 | INF
    else:
        time_step = np.min(np.divide(resolutions, max_velocities))

    times = np.append(np.arange(start_t, end_t, step=time_step), [end_t])
    #times = positions_curve.x
    #velocities_curve = positions_curve.derivative()
    positions = [positions_curve(t) for t in times]
    times, positions = filter_proximity(times, positions, resolution)
    return times, positions

    # TODO: bug here (just use knot points instead?)
    times.extend(np.hstack(positions_curve.derivative().roots(discontinuity=True))) # TODO: make these points special within filter proximity
    times = sorted(set(times))
    positions = [positions_curve(t) for t in times]
    return times, positions


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
