"""

"""
# Standard Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ipywidgets import interactive, fixed
from IPython.display import display
# Solar Data Tools imports
from solardatatools.algorithms import SunriseSunset
from solardatatools.utilities import progress
# Internal imports
from solarfingerprinting import batch_process, make_dictionaries,\
    estimate_parameters
from solarfingerprinting.featurize import freq_domain_features

class ClusterAnalysis():
    def __init__(self, data_handler=None, normed_data=None, envelope='saved',
                 detect_sunup_matrix='raw'):
        if data_handler is not None:
            self.dh = data_handler
            if not self.dh._ran_pipeline:
                try:
                    self.dh.run_pipeline()
                except:
                    msg = 'Auto-run of DataHandler pipeline failed!'
                    return
            self.data = self.dh.filled_data_matrix[:, self.dh.daily_flags.clear]
            self.normed_data = self.normalize_data(
                detect_sunup_matrix=detect_sunup_matrix
            )
        elif normed_data is not None:
            self.dh = None
            self.data = None
            self.normed_data = normed_data
        if envelope == 'saved':
            env = None
        elif envelope == 'average':
            env = np.average(self.normed_data, axis=1)
        elif isinstance(envelope, np.ndarray):
            env = envelope
        self.W1, self.W2 = make_dictionaries(
            envelope=env, normalize=True, reduce_level=2
        )
        # Declare other attributes
        self.signal1 = None
        self.signal2 = None
        self.features = None

    def normalize_data(self, detect_sunup_matrix='raw'):
        dh = self.dh
        ss = SunriseSunset()
        if detect_sunup_matrix == 'raw':
            ss.run_optimizer(dh.raw_data_matrix)
        elif detect_sunup_matrix == 'filled':
            ss.run_optimizer(dh.filled_data_matrix)
        else:
            print("Valid options for 'detect_sunup_matrix' are 'raw' and 'filled'")
        mask = ss.sunup_mask_estimated[:, dh.daily_flags.clear]
        data = self.data
        normed_data = batch_process(data, mask)
        return normed_data

    def signal_separation(self, verbose=True):
        normed_data = self.normed_data
        W1 = self.W1
        W2 = self.W2
        ti = time()
        counter = 0
        total = normed_data.shape[1]
        signal1 = np.zeros((normed_data.shape[1], W1.shape[1]))
        signal2 = np.zeros((normed_data.shape[1], W2.shape[1]))
        for y in list(normed_data.T):
            if verbose:
                progress(counter, total)
            r1, r2 = estimate_parameters(y, mat1=W1, mat2=W2)
            signal1[counter, :] = r1
            signal2[counter, :] = r2
            counter += 1
        if verbose:
            progress(counter, total, 'total_time: {:.2f} minutes'.format(
                (time() - ti) / 60
            ))
        self.signal1 = signal1
        self.signal2 = signal2

    def featurize_days(self, stat_funcs=None, normalize='internal'):
        if self.signal1 is None or self.signal2 is None:
            self.signal_separation()
        wvlt_features = freq_domain_features(self.signal2, stat_funcs=stat_funcs)
        # raw_features = np.c_[self.signal1, wvlt_features]
        self.scaler1 = StandardScaler()
        self.scaler2 = MinMaxScaler(feature_range=(0, 5))
        if normalize == 'internal':
            self.features = np.c_[
                self.scaler1.fit_transform(self.signal1),
                self.scaler2.fit_transform(wvlt_features)
            ]
        elif normalize == 'external':
            import os
            import pickle
            filepath = __file__.split('/')[:-1]
            fn1 = os.path.join('/', *filepath,
                              'fixtures/saved_rooftop_data_scaler1.pkl')
            fn2 = os.path.join('/', *filepath,
                               'fixtures/saved_rooftop_data_scaler2.pkl')
            with open(fn1, 'rb') as f:
                save_dict1 = pickle.load(f)
            with open(fn2, 'rb') as f:
                save_dict2 = pickle.load(f)
            self.scaler1.scale_ = np.array(save_dict1['scale'])
            self.scaler1.var_ = np.array(save_dict1['var'])
            self.scaler1.mean_ = np.array(save_dict1['mean'])
            self.scaler2.min_ = np.array(save_dict2['min'])
            self.scaler2.scale_ = np.array(save_dict2['scale'])
            self.features = np.c_[
                self.scaler1.transform(self.signal1),
                self.scaler2.transform(wvlt_features)
            ]
        else:
            self.features = np.c_[self.signal1, wvlt_features]

    def plot_day(self, day_ix, figsize=(10, 8)):
        signal = self.normed_data[:, day_ix]
        th1 = self.signal1[day_ix]
        th2 = self.signal2[day_ix]
        smooth_basis_fit = self.W1.dot(th1)
        complete_basis_fit = self.W1.dot(th1) + self.W2.dot(th2)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        ax[0, 0].plot(signal, label='normalized signal', linewidth=1)
        ax[0, 0].plot(smooth_basis_fit, linewidth=1, label='smooth basis')
        ax[0, 0].plot(complete_basis_fit, linewidth=1,
                      label='both bases')
        ax[0, 0].set_title('Time Domain')
        ax[0, 1].stem(self.features[day_ix], use_line_collection=True)
        ax[0, 1].set_title('Feature Representation')
        ax[0, 1].set_ylim(np.quantile(self.features, 0.01),
                          np.quantile(self.features, 0.99))
        ax[1, 0].stem(th1, use_line_collection=True)
        ax[1, 0].set_title('Smooth Basis Representation')
        ax[1, 1].stem(th2, use_line_collection=True)
        ax[1, 1].set_title('Sparse Wavelet Basis Representation')
        ax[0, 0].legend()
        plt.show()

    def inspect_days(self, figsize=(10, 8)):
        num_days = self.normed_data.shape[1]
        max_index = num_days - 1
        w = interactive(self.plot_day, day_ix=(0, max_index, 1),
                        figsize=fixed(figsize))
        display(w)
