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

def fingerprint(data, function='gauss_quad', return_fit=True, return_rmse=True,
                residuals=None, reweight=False, normalize=True):
    num_meas_per_hour = len(data) / 24
    x = np.arange(0, 24, 1. / num_meas_per_hour)
    f = FUNCTIONS[function]
    n_args = len(signature(f).parameters) - 1
    init = np.zeros(n_args)
    if residuals is None:
        residuals = np.ones_like(data)
    try:
        optimal_params, _ = optimize.curve_fit(f, x[1:],
                                               data[1:],
                                               p0=init,
                                               sigma=residuals[1:],
                                               maxfev=100000)
    except RuntimeError:
        optimal_params = init
    fit = f(x, *optimal_params)
    residual = data - fit
    rmse = np.linalg.norm(residual) / np.sqrt(len(data))
    if reweight:
        optimal_params, fit, rmse = fingerprint(data, function=function,
                                                residuals=residual + 1e-3,
                                                normalize=False)
    if normalize and function == 'gauss_quad':
        optimal_params = forward_transform(optimal_params)
    result = [optimal_params]
    if return_fit == True:
        result.append(fit)
    if return_rmse == True:
        result.append(rmse)
    return optimal_params, fit, rmse