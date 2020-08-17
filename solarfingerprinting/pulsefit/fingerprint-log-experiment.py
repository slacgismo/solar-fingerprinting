# -*- coding: utf-8 -*-
''' Fingerprinting algorithm module
This module contains the fingerprinting algorithm for daily PV power signals
'''

import numpy as np
from scipy import optimize
from inspect import signature
from solarfingerprinting.pulsefit.pulses import log_gquad
from solarfingerprinting.pulsefit.transform import forward_transform


def fingerprint(data, return_fit=True, return_rmse=True, residuals=None,
                reweight=False, normalize=True):
    max_val = np.max(data)
    mask = data > 0
    num_meas_per_hour = len(data) / 24
    x = np.arange(0, 24, 1. / num_meas_per_hour)
    y = np.zeros_like(x)
    y[mask] = np.log(data[mask] / max_val)
    mask = data > 0
    f = log_gquad
    n_args = len(signature(f).parameters) - 1
    init = np.zeros(n_args)
    if residuals is None:
        residuals = np.ones_like(data)
    try:
        optimal_params, _ = optimize.curve_fit(f, x[mask], y[mask],
                                               p0=init,
                                               sigma=residuals[mask],
                                               maxfev=100000)
    except RuntimeError:
        optimal_params = None
        fit = None
        rmse = None
    else:
        fit = np.exp(f(x, *optimal_params)) * max_val
        residual = data - fit
        rmse = np.linalg.norm(residual) / np.sqrt(len(data))
        if reweight:
            optimal_params, fit, rmse = fingerprint(data, function=function,
                                                    residuals=residual + 1e-3,
                                                    normalize=False)
        if normalize:
            optimal_params = forward_transform(optimal_params)
    result = [optimal_params]
    if return_fit == True:
        result.append(fit)
    if return_rmse == True:
        result.append(rmse)
    return optimal_params, fit, rmse