import numpy as np

from motion_planners.utils import INF


def check_time(t):
    return not isinstance(t, complex) and (t >= 0.)


def iterate_poly1d(poly1d):
    return enumerate(reversed(list(poly1d)))


def parabolic_val(t=0., t0=0., x0=0., v0=0., a=0.):
    return x0 + v0*(t - t0) + 1/2.*a*(t - t0)**2

##################################################

def poly_sum(p0, *polys):
    p_total = np.poly1d(p0)
    for p in polys:
        p_total = np.poly1d(np.polyadd(p_total, p))
    return p_total


def poly_prod(p0, *polys):
    p_total = np.poly1d(p0)
    for p in polys:
        p_total = np.poly1d(np.polymul(p_total, p))
    return p_total


def separate_poly(poly):
    from scipy.interpolate import PPoly
    k, m, d = poly.c.shape
    return [PPoly(c=poly.c[:,:,i], x=poly.x) for i in range(d)]

##################################################

def curve_from_controls(durations, accels, t0=0., x0=0., v0=0.):
    assert len(durations) == len(accels)
    #from numpy.polynomial import Polynomial
    #t = Polynomial.identity()
    times = [t0]
    positions = [x0]
    velocities = [v0]
    coeffs = []
    for duration, accel in zip(durations, accels):
        assert duration >= 0.
        coeff = [0.5*accel, 1.*velocities[-1], positions[-1]]
        coeffs.append(coeff)
        times.append(times[-1] + duration)
        p_curve = np.poly1d(coeff) # Not centered
        positions.append(p_curve(duration))
        v_curve = p_curve.deriv() # Not centered
        velocities.append(v_curve(duration))
    # print(positions)
    # print(velocities)

    #np.piecewise
    # max_order = max(p_curve.order for p_curve in p_curves)
    # coeffs = np.zeros([max_order + 1, len(p_curves), 1])
    # for i, p_curve in enumerate(p_curves):
    #     # TODO: need to center
    #     for k, c in iterate_poly1d(p_curve):
    #         coeffs[max_order - k, i] = c
    from scipy.interpolate import PPoly
    # TODO: check continuity
    return PPoly(c=np.array(coeffs).T, x=times) # TODO: spline.extend


def filter_times(times):
    valid_times = list(filter(check_time, times))
    if not valid_times:
        return None
    return min(valid_times)


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

    t_ramp = filter_times((velocity_curve - np.poly1d([sign * v_max])).roots)
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