# -*- coding: utf-8 -*-
''' Fingerprinting study module
This module contains a Python class to handle the execution of a fingerprint
study on a data set with more than one day of data.
'''

from time import time
import numpy as np
from functools import partial, update_wrapper
from solarfingerprinting.fingerprint import fingerprint
from solarfingerprinting.utilities import progress
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, interact_manual, interactive, fixed
from IPython.display import display

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

    def plot_day(self, index, figsize=(10, 8)):
        data = self.data[:, index]
        fit = self.fits[:, index]
        num_meas_per_hour = len(data) / 24
        x = np.arange(0, 24, 1. / num_meas_per_hour)

        with sns.axes_style("white"):
            max_val = np.max(self.data)
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
            mat1 = self.data
            mat2 = np.nan_to_num(self.fits)
            ax[0, 0].imshow(mat1, cmap='hot', interpolation='none', aspect='auto',
                            vmin=0)
            ax[0, 0].set_title('power signal')
            ax[0, 1].imshow(mat2, cmap='hot', interpolation='none', aspect='auto',
                            vmin=0)
            ax[0, 1].set_title('fit pulses')
            for j in range(2):
                ax[0, j].set_xlabel('Day number')
                ax[0, j].set_yticks([])
                ax[0, j].set_ylabel('(sunset)        Time of day        (sunrise)')
                ax[0, j].axvline(index, color='red', ls='--')

            ax[1, 0].plot(x, data, label='power signal')
            ax[1, 0].plot(x, fit, label='fit pulse')
            ax[1, 0].legend()
            ax[1, 0].set_xlabel('time (5-minute increments)')
            ax[1, 0].set_ylabel('power')
            ax[1, 0].set_title('signals, day number {}'at(index))
            ax[1, 0].set_ylim(-0.05 * max_val, 1.05 * max_val)

            popt_t = self.encoding[index]
            ax[1, 1].stem(popt_t, use_line_collection=True)
            ax[1, 1].set_title('signal fingerprint')
            ax[1, 1].set_xlabel('parameter index')
            ax[1, 1].set_ylabel('parameter value')
            ax[1, 1].set_title('fingerprint, day number {}'.format(index))
            ax[1, 1].set_ylim(-5, 5)

        plt.tight_layout()
        plt.show()

    def inspect_days(self, figsize=(10, 8)):
        num_days = self.data.shape[1]
        max_index = num_days - 1
        w = interactive(self.plot_day, index=(0, max_index, 1),
                        figsize=fixed(figsize))
        display(w)