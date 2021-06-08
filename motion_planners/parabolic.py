import math

import numpy as np

from motion_planners.retime import curve_from_controls, parabolic_val, check_time
from motion_planners.utils import INF

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


def shortest_inf_accel(x1, x2, v_max=INF):
    #return solve_zero_ramp(x1, x2, v_max=INF)
    return abs(x2 - x1) / abs(v_max)

##################################################

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

##################################################

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

##################################################

def solve_ramp(x1, x2, v1, v2, v_max=INF, a_max=INF, min_t=0.):
    # TODO: handle infinite acceleration
    assert (v_max >= 0.) and (a_max >= 0.)
    assert all(abs(v) <= v_max for v in [v1, v2])
    if (v_max == INF) and (a_max == INF):
        T = 0
        return min(min_t, T) # TODO: throw an error
    if a_max == INF:
        T = shortest_inf_accel(x1, x2, v_max=v_max)
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
