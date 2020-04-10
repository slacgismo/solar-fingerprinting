# -*- coding: utf-8 -*-
''' Pulses module
This module contains functions which parameterized functions which define
pulses of various shapes. The fingerprinting algorithm utilizes the Gaussian
pulse modified by a quadratic, or "gquad". All the functions translate and
scale the parameter inputs. The point of this is so that
scipy.optimize.curve_fit can find optimal parameter values for typical normalized
PV power signals.
'''

import numpy as np
from scipy.special import erfc

def gaussian(x, loc, scale, beta, amplitude):
    x = x.astype(np.float)
    a_p = np.exp(amplitude)
    loc_p = loc + 12
    scale_p = scale + 4
    beta_p =  beta + 1
    return a_p * np.exp(-(np.abs(x - loc_p) / scale_p) ** beta_p)

def gpow(x, loc, scale, beta, amplitude, k):
    x = x.astype(np.float)
    a_p = np.exp(amplitude)
    loc_p = loc + 12
    scale_p = scale + 4
    beta_p =  beta + 1
    k_p = k + 1
    return a_p * (x ** (k_p - 1)) * np.exp(-(np.abs(x - loc_p) / scale_p) ** beta_p)

def glin(x, loc, scale, beta, m, b):
    x = x.astype(np.float)
    loc_p = loc + 12
    scale_p = scale + 4
    beta_p =  beta + 1
    b_p = b + 1
    return (m * x + b_p) * np.exp(-(np.abs(x - loc_p) / scale_p) ** beta_p)

def gquad(x, loc, scale, beta, a, b, c):
    x = x.astype(np.float)
    loc_p = loc + 12
    scale_p = scale + 4
    beta_p =  3 * beta + 6
    # The following three parameters control the quadratic functions. The "raw"
    # parameters are highly correlated, so we sequentially orthogonalize the
    # dimensions. The magnitudes are selected based on values that successfully
    # orthagonalized a randomized set of daily signals.
    a_p = a / 10
    b_p = b / 10 - 2.4 * a
    c_p = c + 14 * a - 1.2 * b
    t1 = np.exp(a_p * x ** 2 + b_p * x + c_p)
    t2 = np.exp(-(np.abs(x - loc_p) / scale_p) ** beta_p)
    return t1 * t2

def log_gquad(x, loc, scale, beta, a, b, c):
    x = x.astype(np.float)
    loc_p = loc + 12
    scale_p = scale + 4
    beta_p =  3 * beta + 6
    # The following three parameters control the quadratic functions. The "raw"
    # parameters are highly correlated, so we sequentially orthogonalize the
    # dimensions. The magnitudes are selected based on values that successfully
    # orthagonalized a randomized set of daily signals.
    a_p = a / 10
    b_p = b / 10 - 2.4 * a
    c_p = c + 14 * a - 1.2 * b
    t1 = (a_p * x ** 2 + b_p * x + c_p)
    t2 = (-(np.abs(x - loc_p) / scale_p) ** beta_p)
    return t1 + t2

def gquad2(x, alpha, beta, a, b, c, k):
    x_tilde = x - len(x) / 2
    expr = alpha * x_tilde ** beta + a * x_tilde ** 2 + b * x_tilde + c
    f = k * np.exp(expr)
    return f

def gatan(x, loc, scale, beta, a, s, u, v):
    x = x.astype(np.float)
    loc_p = loc + 12
    scale_p = scale + 4
    beta_p =  beta + 1
    a_p = a
    s_p = s + 1
    u_p = u # 5 * u + 145
    v_p = v + 1
    t1 = (a_p * np.arctan(x * s_p - u_p) + v_p)
    t2 = np.exp(-(np.abs(x - loc_p) / scale_p) ** beta_p)
    return t1 * t2

def g2(x, loc1, scale1, beta1, m1, b1, loc2, scale2, beta2, m2, b2):
    p1 = glin(x, loc1, scale1, beta1, m1, b1)
    p2 = glin(x, loc2, scale2, beta2, m2, b2)
    return p1 + p2

def emg(x, amplitude, mu, sigma, lmbda):
    p1 = lmbda / 2
    p2 = np.exp(p1 * (2 * mu + lmbda * sigma ** 2 - 2 * x))
    p3 = erfc((mu + lmbda * sigma ** 2 - x) / (np.sqrt(2) * sigma))
    return amplitude * p2 * p3