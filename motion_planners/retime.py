import numpy as np

from motion_planners.utils import INF, get_pairs, find


def check_time(t):
    return not isinstance(t, complex) and (t >= 0.)


def filter_times(times):
    valid_times = list(filter(check_time, times))
    if not valid_times:
        return None
    return min(valid_times)


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


def append_polys(poly1, *polys):
    from scipy.interpolate import PPoly
    total_poly = poly1
    for poly in polys:
        new_xs = [total_poly.x[-1] + (x - poly.x[0]) for x in poly.x[1:]]
        total_poly = PPoly(c=np.append(total_poly.c, poly.c, axis=1),
                           x=np.append(total_poly.x, new_xs))
        #total_poly.extend()
    return total_poly


def spline_start(spline):
    return spline.x[0]


def spline_end(spline):
    return spline.x[-1]


def spline_duration(spline):
    return spline_end(spline) - spline_start(spline)

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
        coeff = [0.5*accel, 1.*velocities[-1], positions[-1]] # 0. jerk
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
    #from scipy.interpolate import CubicHermiteSpline
    return PPoly(c=np.array(coeffs).T, x=times) # TODO: spline.extend

##################################################

def min_linear_spline(x1, x2, v_max, a_max, t0=0.):
    #if x1 > x2:
    #   return conservative(x2, x1, v_max, a_max)
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
        spline = curve_from_controls(durations, accels, t0=t0, x0=x1, v0=v1)
        # T = 2*t_half
        # times = [0., t_half, T]
        # c = np.zeros([3, len(times) - 1])
        # c[:, 0] = list(position_curve)
        # c[:, 1] = [-0.5*accels[1], velocity_curve(t_half), position_curve(t_half)]
        # spline = PPoly(c=c, x=times)
        return spline

    t_ramp = filter_times((velocity_curve - np.poly1d([sign * v_max])).roots)
    assert t_ramp is not None
    x_ramp = position_curve(t_ramp)
    d = abs(x2 - x1)
    d_ramp = abs(x_ramp - x1)
    d_hold = d - 2*d_ramp
    t_hold = abs(d_hold / v_max)

    durations = [t_ramp, t_hold, t_ramp]
    accels = [sign * a_max, 0., -sign * a_max]
    spline = curve_from_controls(durations, accels, t0=t0, x0=x1, v0=v1)

    # T = 2*t_ramp + t_hold
    # times = [0., t_ramp, t_ramp + t_hold, T]
    # c = np.zeros([3, len(times) - 1])
    # c[:, 0] = list(position_curve)
    # c[:, 1] = [0.5 * accels[1], velocity_curve(t_ramp), position_curve(t_ramp)]
    # c[:, 2] = [0.5 * accels[2], velocity_curve(t_ramp), position_curve(t_ramp) + velocity_curve(t_ramp)*t_hold]
    # spline = PPoly(c=c, x=times) # TODO: extend
    return spline

def crop_poly(poly, t1=None, t2=None):
    from scipy.interpolate import PPoly
    print(t1, t2)
    if t1 is None:
        t1 = spline_start(poly)
    if t2 is None:
        t2 = spline_end(poly)
    assert t1 <= t2
    times = poly.x
    i1 = find(lambda i: times[i] >= t1, range(len(times)))
    i2 = find(lambda i: times[i] <= t2, reversed(range(len(times))))
    print(i1, i2)
    print(([t1] + poly.x[i1+1:i2] + [t2]).shape)
    print(poly.c[:,i1-1:i2+1,...].shape)
    # TODO: the adjusting of the center is a headache
    return PPoly(c=poly.c[:,i1:i2+1,...],
                 x=[t1] + poly.x[i1+1:i2] + [t2])

class MultiPPoly(object):
    def __init__(self, polys):
        self.polys = list(polys)
        self.x = sorted(np.concatenate([poly.x for poly in self.polys]))
        self.x = [self.x[0]] + [x2 for x1, x2 in get_pairs(self.x) if x2 > x1]
        # TODO: cache derivatives
    @property
    def d(self):
        return len(self.polys)
    @property
    def start_x(self):
        return spline_start(self)
    @property
    def end_(self):
        return spline_end(self)
    def __call__(self, *args, **kwargs):
        return np.array([poly(*args, **kwargs) for poly in self.polys])
    @staticmethod
    def concatenate(self, polys):
        raise NotImplementedError()
        #return MultiPPoly()
    # def slice(self, start, stop=None):
    #     raise NotImplementedError()
    # TODO: extend
    def derivative(self, *args, **kwargs):
        return MultiPPoly([poly.derivative(*args, **kwargs) for poly in self.polys])
    def antiderivative(self, *args, **kwargs):
        return MultiPPoly([poly.antiderivative(*args, **kwargs) for poly in self.polys])
    def roots(self, *args, **kwargs):
        return np.array([poly.roots(*args, **kwargs) for poly in self.polys])
    def spline(self, **kwargs):
        from scipy.interpolate import CubicSpline
        times = self.x
        positions = [self(x) for x in times]
        return CubicSpline(times, positions, bc_type='clamped', **kwargs)
    def hermite_spline(self, **kwargs):
        from scipy.interpolate import CubicHermiteSpline
        times = self.x
        positions = [self(x) for x in times]
        derivative = self.derivative()
        velocities = [derivative(x) for x in times]
        return CubicHermiteSpline(times, positions, dydx=velocities, **kwargs)
    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.polys)
