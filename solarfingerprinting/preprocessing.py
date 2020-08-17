""" Preprocessing module

"""

import numpy as np
from scipy.interpolate import interp1d
from solardatatools.algorithms import SunriseSunset

def batch_process(data, mask, power=10):
    """ Process an entire PV power matrix at once

    :return:
    """
    N = 2 ** power
    output = np.zeros((N, data.shape[1]))
    xs_new = np.linspace(0, 1, N)
    for col_ix in range(data.shape[1]):
        y = data[:, col_ix]
        msk = mask[:, col_ix]
        xs = np.linspace(0, 1, int(np.sum(msk)))
        interp_f = interp1d(xs, y[msk])
        resampled_signal = interp_f(xs_new)
        output[:, col_ix] = resampled_signal / np.max(resampled_signal)
    return output
