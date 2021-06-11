import math

import numpy as np

from .retime import curve_from_controls, parabolic_val, check_time, spline_duration, append_polys, \
    MultiPPoly
from .limits import maximize_curve, EPSILON
from ..utils import INF, get_sign, get_pairs, waypoints_from_path

def check_curve(p_curve, x1, x2, v1, v2, T, v_max=INF, a_max=INF):
    assert p_curve is not None
    end_times = np.append(p_curve.x[:1], p_curve.x[-1:])
    v_curve = p_curve.derivative()
    # print()
    #print(x1, x2, v1, v2, T, v_max, a_max)
    # print([x1, x2], [float(p_curve(t)) for t in end_times])
    # print([v1, v2], [float(v_curve(t)) for t in end_times])
    if not np.allclose([0., T], end_times):
        raise RuntimeError([0., T], end_times)
    if not np.allclose([x1, x2], [float(p_curve(t)) for t in end_times]):
        raise RuntimeError([x1, x2], [float(p_curve(t)) for t in end_times])
    if not np.allclose([v1, v2], [float(v_curve(t)) for t in end_times]):
        raise RuntimeError([v1, v2], [float(v_curve(t)) for t in end_times])
    all_times = p_curve.x
    if not all(abs(v_curve(t)) <= abs(v_max) + EPSILON for t in all_times):
        raise RuntimeError(abs(v_max), [abs(v_curve(t)) for t in all_times])
    a_curve = v_curve.derivative()
    if not all(abs(a_curve(t)) <= abs(a_max) + EPSILON for t in all_times):
        raise RuntimeError(abs(a_max), [abs(a_curve(t)) for t in all_times])
    # TODO: check continuity

##################################################

def acceleration_cost(p_curve, **kwargs):
    # TODO: minimize max acceleration
    # TODO: minimize sum of squared
    # TODO: minimize absolute value
    # total_accel = p_curve.integrate(p_curve.x[0], p_curve.x[-1]) # TODO: square or abs
    a_curve = p_curve.derivative(nu=2)
    max_t, max_a = maximize_curve(a_curve, **kwargs)
    return max_a

def zero_one_fixed(x1, x2, T, v_max=INF):
    from scipy.interpolate import PPoly
    assert 0 < v_max < INF
    sign = get_sign(x2 - x1)
    d = abs(x2 - x1)
    v = d / T
    if v > v_max + EPSILON:
        return None
    dt = EPSILON
    t_hold = T - 2*dt
    v = d / t_hold
    #return zero_three_stage(x1, x2, T, v_max=v_max, a_max=1e6) # NOTE: approximation
    coeffs = [[0., x1], [sign*v, x1], [0., x2]]
    times = [0., dt, dt + t_hold, T]
    p_curve = PPoly(c=np.array(coeffs).T, x=times) # Not differentiable
    return p_curve

def zero_two_ramp(x1, x2, T, v_max=INF, a_max=INF):
    sign = get_sign(x2 - x1)
    d = abs(x2 - x1)
    t_accel = T / 2.
    a = d / t_accel ** 2 # Lower accel
    if a > a_max + EPSILON:
        return None
    if a*t_accel > v_max + EPSILON:
        return None
    a = min(a, a_max) # Numerical error
    durations = [t_accel, t_accel]
    accels = [sign * a, -sign * a]
    p_curve = curve_from_controls(durations, accels, x0=x1)
    return p_curve

def zero_three_stage(x1, x2, T, v_max=INF, a_max=INF):
    sign = get_sign(x2 - x1)
    d = abs(x2 - x1)
    solutions = np.roots([
        a_max,
        -a_max*T,
        d,
    ])
    solutions = filter(check_time, solutions)
    solutions = [t for t in solutions if (T - 2*t) >= 0]
    if not solutions:
        return None
    t1 = min(solutions)
    if t1*a_max > v_max + EPSILON:
        return None
    #t1 = min(t1, v_max / a_max)
    t3 = t1
    t2 = T - t1 - t3 # Lower velocity
    durations = [t1, t2, t3]
    accels = [sign * a_max, 0., -sign * a_max]
    p_curve = curve_from_controls(durations, accels, x0=x1)
    return p_curve

def opt_straight_line(x1, x2, v_max=INF, a_max=INF, T_min=0.):
    # TODO: solve for a given T which is higher than the min T
    # TODO: solve for all joints at once using a linear interpolator
    # Can always rest at the start of the trajectory if need be
    # Exploits symmetry
    assert (v_max > 0.) and (a_max > 0.)
    assert (v_max < INF) or (a_max < INF)
    #v_max = abs(x2 - x1) / abs(v_max)
    d = abs(x2 - x1)
    # if v_max == INF:
    #     raise NotImplementedError()

    if a_max == INF:
        T = d / v_max
        T += 2 * EPSILON
        p_curve = zero_one_fixed(x1, x2, T, v_max=v_max)
        check_curve(p_curve, x1, x2, v1=0., v2=0., T=T, v_max=v_max, a_max=a_max)
        return p_curve

    t_accel = math.sqrt(d / a_max) # 1/2.*a*t**2 = d/2.
    if a_max*t_accel <= v_max:
        T = 2.*t_accel
        #a = a_max
        assert T_min <= T
        T = max(T_min, T)
        p_curve = zero_two_ramp(x1, x2, T, v_max, a_max)
        check_curve(p_curve, x1, x2, v1=0., v2=0., T=T, v_max=v_max, a_max=a_max)
        return p_curve

    t1 = t3 = (v_max - 0.) / a_max
    t2 = (d - 2 * parabolic_val(t1, a=a_max)) / v_max
    T = t1 + t2 + t3

    assert T_min <= T
    T = max(T_min, T)
    p_curve = zero_three_stage(x1, x2, T, v_max=v_max, a_max=a_max)
    check_curve(p_curve, x1, x2, v1=0., v2=0., T=T, v_max=v_max, a_max=a_max)
    return p_curve


def quickest_inf_accel(x1, x2, v_max=INF):
    #return solve_zero_ramp(x1, x2, v_max=INF)
    return abs(x2 - x1) / abs(v_max)

##################################################

def min_two_ramp(x1, x2, v1, v2, T, a_max, v_max=INF):
    # from numpy.linalg import solve
    # from scipy.optimize import linprog
    # result = linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
    #                  method='interior-point', callback=None, options=None, x0=None)

    sign = +1 if a_max >= 0 else -1
    eqn = np.poly1d([
        T ** 2, # a**2
        sign * (2 * T * (v1 + v2) + 4 * (x1 - x2)),
        -(v2 - v1) ** 2, # 1
    ])
    candidates = []
    for a in np.roots(eqn): # eqn.roots
        if isinstance(a, complex) or (a == 0):
            continue
        if abs(a) > abs(a_max) + EPSILON:
            continue
        a = sign*a
        ts = (T + (v2 - v1) / a) / 2.
        if not (0 <= ts <= T):
            continue
        vs = v1 + a * ts
        if abs(vs) > abs(v_max) + EPSILON:
            continue
        # vs = a * (-ts) + v2
        # if abs(vs) > abs(v_max):
        #     continue
        candidates.append(a)
    if not candidates:
        return None

    a = min(candidates, key=lambda a: abs(a))
    ts = (T + (v2 - v1) / a) / 2.
    durations = [ts, T - ts]
    #accels = [sign*abs(a), -sign*abs(a)]
    accels = [a, -a]
    p_curve = curve_from_controls(durations, accels, t0=0., x0=x1, v0=v1)
    #return p_curve
    check_curve(p_curve, x1, x2, v1, v2, T, v_max=v_max, a_max=a_max)
    return p_curve

def quickest_two_ramp(x1, x2, v1, v2, a_max, v_max=INF):
    #optimize_two_ramp(x1, x2, v1, v2, a_max)
    solutions = np.roots([
        a_max, # t**2
        2 * v1, # t
        (v1**2 - v2**2) / (2 * a_max) + (x1 - x2), # 1
    ])
    solutions = [t for t in solutions if check_time(t)]
    #solutions = [t for t in solutions if t <= abs(v2 - v1) / abs(a_max)] # TODO: omitting this constraint from Hauser (abs(v2 - v1) might be 0)
    solutions = [t for t in solutions if t <= abs(v_max) / abs(a_max)] # Maybe this is what Hauser meant?
    solutions = [t for t in solutions if abs(v1 + t*a_max) <= abs(v_max) + EPSILON] # TODO: check v2
    if not solutions:
        return None
    t = min(solutions)
    T = 2*t + (v1 - v2) / a_max
    if T < 0:
        return None
    #min_two_ramp(x1, x2, v1, v2, T, a_max, v_max=v_max)
    return T

##################################################

def solve_three_stage(x1, x2, v1, v2, v_max, a):
    tp1 = (v_max - v1) / a
    tl = (v2 ** 2 + v1 ** 2 - 2 * v_max ** 2) / (2 * v_max * a) \
         + (x2 - x1) / v_max
    tp2 = (v2 - v_max) / -a # Difference from Hauser
    return tp1, tl, tp2


def min_three_stage(x1, x2, v1, v2, T, v_max, a_max=INF):
    #assert abs(v_max) < INF
    a = (v_max**2 - v_max*(v1 + v2) + (v1**2 + v2**2)/2) \
        / (T*v_max - (x2 - x1))
    if np.isnan(a) or (abs(a) > abs(a_max) + EPSILON) or (a == 0):
        return None
    durations = solve_three_stage(x1, x2, v1, v2, v_max, a)
    if any(t <= 0 for t in durations): # TODO: check T
        return None
    accels = [a, 0., -a]
    p_curve = curve_from_controls(durations, accels, t0=0., x0=x1, v0=v1)
    check_curve(p_curve, x1, x2, v1, v2, T, v_max=INF, a_max=INF)
    return p_curve


def quickest_three_stage(x1, x2, v1, v2, v_max, a_max):
    # http://motion.pratt.duke.edu/papers/icra10-smoothing.pdf
    # https://github.com/Puttichai/parabint/blob/2662d4bf0fbd831cdefca48863b00d1ae087457a/parabint/optimization.py
    # TODO: minimum-switch-time constraint
    #assert np.positive(v_max).all() and np.positive(a_max).all()
    # P+L+P-
    ts = solve_three_stage(x1, x2, v1, v2, v_max, a_max)
    if any(t < 0 for t in ts):
        return None
    T = sum(ts)
    #min_three_ramp(x1, x2, v1, v2, v_max, a_max, T)
    return T

##################################################

def min_stage(x1, x2, v1, v2, T, v_max=INF, a_max=INF):
    # if (v1 == 0.) and (v2 == 0.):
    #     # TODO: isn't necessarily a straight line in multi-dimensions
    #     candidates = [
    #         zero_two_ramp(x1, x2, T, v_max=v_max, a_max=a_max),
    #         zero_three_stage(x1, x2, T, v_max=v_max, a_max=a_max),
    #     ]
    # else:
    candidates = [
        min_two_ramp(x1, x2, v1, v2, T, a_max=a_max, v_max=v_max),
        min_two_ramp(x1, x2, v1, v2, T, a_max=-a_max, v_max=-v_max),
    ]
    #if v_max != INF:
    candidates.extend([
        min_three_stage(x1, x2, v1, v2, T, v_max, a_max),
        min_three_stage(x1, x2, v1, v2, T, -v_max, -a_max),
    ])
    candidates = [t for t in candidates if t is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda c: (spline_duration(c), acceleration_cost(c)))

def min_spline(times, positions, velocities, **kwargs):
    assert len(times) == len(positions) == len(velocities)
    # if not strictly_increasing(times):
    #     print(times)
    #     print(positions)
    #     input()
    splines = []
    for (t1, x1, v1), (t2, x2, v2) in get_pairs(list(zip(times, positions, velocities))):
        T = t2 - t1
        if T == 0: # TODO: check x1/x2 continuity etc.
            continue
        spline = min_stage(x1, x2, v1, v2, T, **kwargs)
        if spline is None:
            return None
        splines.append(spline)
    return append_polys(*splines)

def solve_multi_poly(times, positions, velocities, v_max, a_max, **kwargs):
    assert len(times) == len(positions) == len(velocities)
    d = len(positions[0])
    assert len(positions[0]) == len(velocities[0])
    positions = np.array(positions)
    velocities = np.array(velocities)
    positions_curves = [min_spline(times, positions[:, i], velocities[:, i],
                                   v_max=v_max[i], a_max=a_max[i], **kwargs) for i in range(d)]
    if any(position_curve is None for position_curve in positions_curves):
        return None
    return MultiPPoly(positions_curves)

def solve_multi_linear(positions, v_max, a_max, **kwargs):
    from scipy.interpolate import PPoly
    positions = waypoints_from_path(positions, **kwargs)
    d = len(positions[0])
    assert len(v_max) == len(a_max)
    splines = []
    for x1, x2 in get_pairs(positions):
        difference = np.subtract(x2, x1) # TODO: pass if too small
        unit_v_max = min(np.divide(v_max, np.absolute(difference)))
        unit_a_max = min(np.divide(a_max, np.absolute(difference)))
        curve = opt_straight_line(x1=0., x2=1., v_max=unit_v_max, a_max=unit_a_max)
        c = np.zeros(curve.c.shape + (d,))
        for k in range(d):
            c[:,:,k] = difference[k]*curve.c
            c[-1,:,k] += x1[k]
        splines.append(PPoly(c=c, x=curve.x))
    return append_polys(*splines)

##################################################

def quickest_stage(x1, x2, v1, v2, v_max=INF, a_max=INF, min_t=0.):
    # TODO: handle infinite acceleration
    assert (v_max > 0.) and (a_max > 0.)
    assert all(abs(v) <= v_max + EPSILON for v in [v1, v2])
    if (v_max == INF) and (a_max == INF):
        T = 0
        return min(min_t, T) # TODO: throw an error
    if a_max == INF:
        T = quickest_inf_accel(x1, x2, v_max=v_max)
        return min(min_t, T)

    # if (v1 == 0.) and (v2 == 0.):
    #     candidates = [opt_straight_line(x1, x2, v_max=v_max, a_max=a_max).x]

    candidates = [
        quickest_two_ramp(x1, x2, v1, v2, a_max, v_max=v_max),
        quickest_two_ramp(x1, x2, v1, v2, -a_max, v_max=-v_max),
    ]
    #if v_max != INF:
    candidates.extend([
        quickest_three_stage(x1, x2, v1, v2, v_max, a_max),
        quickest_three_stage(x1, x2, v1, v2, -v_max, -a_max),
    ])
    candidates = [t for t in candidates if t is not None]
    if not candidates:
        return None
    T = min(t for t in candidates)
    return max(min_t, T)

def solve_multivariate_ramp(x1, x2, v1, v2, v_max, a_max):
    d = len(x1)
    durations = [quickest_stage(x1[i], x2[i], v1[i], v2[i], v_max[i], a_max[i]) for i in range(d)]
    if any(duration is None for duration in durations):
        return None
    return max(durations)

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
    raise NotImplementedError()