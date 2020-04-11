# -*- coding: utf-8 -*-
''' Fingerprinting study module
This module contains a Python class to handle the execution of a fingerprint
study on a data set with more than one day of data.
'''

from time import time
import numpy as np
from solarfingerprinting.fingerprint import fingerprint
from solarfingerprinting.utilities import progress
import matplotlib as plt

class StudyHandler():
    def __init__(self, data, use_days=None):
        self.data = data
        if use_days is not None:
            self.use_days = use_days
        else:
            self.use_days = np.ones(data.shape[1], dtype=np.bool)
        self.encoding = None
        self.fits = None
        self.fail_list = None

    def execute(self, reweight=False, normalize=True, verbose=True):
        n_signals = self.data.shape[1]
        n_args = 7
        encoding = np.empty((n_signals, n_args))
        encoding[:] = np.nan
        fail_list = []
        fits = np.empty_like(self.data)
        fits[:] = np.nan
        ti = time()
        for ix in np.arange(n_signals)[self.use_days]:
            if verbose:
                progress(ix, n_signals,
                         ' {:.2f} minutes'.format((time() - ti) / 60.))
            signal = self.data[:, ix]
            optimal_params, fit = fingerprint(signal,
                                                    normalize=normalize,
                                                    reweight=reweight)
            if optimal_params is None:
                fail_list.append(ix)
            else:
                encoding[ix] = optimal_params
                fits[:, ix] = fit
        if verbose:
            progress(n_signals, n_signals,
                     ' {:.2f} minutes'.format((time() - ti) / 60.))
        self.encoding = encoding
        self.fits = fits
        if len(fail_list) > 0:
            self.fail_list = fail_list
        else:
            self.fail_list = None

    def plot_day(self, index):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        return