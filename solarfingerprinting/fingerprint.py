# -*- coding: utf-8 -*-
''' Fingerprinting algorithm module
This module contains the fingerprinting algorithm for daily PV power signals
'''

import numpy as np
from scipy import optimize
from inspect import signature
from solarfingerprinting.pulses import gaussian, gpow, glin, gquad, gatan, g2
from solarfingerprinting.transform import forward_transform

FUNCTIONS = {
    'gauss': gaussian,
    'gauss_power': gpow,
    'gauss_linear': glin,
    'gauss_quad': gquad,
    'gauss_atan2': gatan,
    'gaus_lin_mixture': g2
}

def fingerprint(data, function='gauss_quad', residuals=None, reweight=False,
                normalize=True):
    max_val = np.max(data)
    num_meas_per_hour = len(data) / 24
    x = np.arange(0, 24, 1. / num_meas_per_hour)
    f = FUNCTIONS[function]
    n_args = len(signature(f).parameters) - 1
    init = np.zeros(n_args)
    if residuals is None:
        residuals = np.ones_like(data)
    try:
        optimal_params, _ = optimize.curve_fit(f, x[1:],
                                               data[1:] / max_val,
                                               p0=init,
                                               sigma=residuals[1:],
                                               maxfev=100000)
    except RuntimeError:
        encoding = None
        fit = None
    else:
        fit = np.zeros_like(x)
        fit[1:] = f(x[1:], *optimal_params) * max_val
        residual = data - fit
        # Using L-inf norm on normalized error rather than RMSE
        # rmse = np.linalg.norm(residual / max_val) / np.sqrt(len(data))
        inf_error = np.max(np.abs(residual / max_val))
        encoding = np.r_[optimal_params, inf_error]
        if reweight:
            encoding, fit = fingerprint(data, function=function,
                                        residuals=(residual + 1e-3),
                                        normalize=False)
        if normalize and function == 'gauss_quad':
            encoding = forward_transform(encoding)
    return encoding, fit