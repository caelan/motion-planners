import time

import numpy as np

from motion_planners.utils import INF, elapsed_time


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

##################################################

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

def find_max_curve(curve, start_t=None, end_t=None, norm=INF, **kwargs):
    if start_t is None:
        start_t = curve.x[0]
    if end_t is None:
        end_t = curve.x[-1]
    objective = lambda t: -np.linalg.norm(curve(t), ord=norm) # 2 | INF
    #objective = lambda t: -np.linalg.norm(curve(t), norm=2)**2 # t[0]
    #accelerations_curve = positions_curve.derivative() # TODO: ValueError: failed in converting 7th argument `g' of _lbfgsb.setulb to C/Fortran array
    #grad = lambda t: np.array([-2*sum(accelerations_curve(t))])
    #result = minimize_scalar(objective, method='bounded', bounds=(start_t, end_t)) #, options={'disp': False})

    #print(max(-objective(t) for t in curve.x))
    best_t, best_f = optimize(objective, lower=[start_t], upper=[end_t], **kwargs)
    best_f = objective(best_t)
    best_t, best_f = best_t[0], -best_f
    return best_t, best_f

##################################################

def optimize_curve(curve): # fn=None
    derivative = curve.derivative(nu=1)
    times = list(curve.x)
    critical = derivative.roots(discontinuity=True)
    for k in range(critical.shape[0]):
        times.extend(critical[k])
    #fn = lambda t: max(np.absolute(curve(t)))
    fn = lambda t: np.linalg.norm(curve(t), ord=INF)
    max_time = max(times, key=fn)
    return max_time, fn(max_time)

##################################################

def get_max_velocity(velocities, norm=INF):
    return np.linalg.norm(velocities, ord=norm)

def find_max_velocity(positions_curve, **kwargs):
    velocities_curve = positions_curve.derivative(nu=1)
    #return find_max_curve(velocities_curve, **kwargs)
    return optimize_curve(velocities_curve)


def find_max_acceleration(positions_curve, **kwargs):
    accelerations_curve = positions_curve.derivative(nu=2)
    #return find_max_curve(accelerations_curve, **kwargs)
    return optimize_curve(accelerations_curve)
