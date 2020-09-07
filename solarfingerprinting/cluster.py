"""

"""
# Standard Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from time import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        self.N_ = None
        self.daily_distance = None
        self.site_score = None
        # Cluster Information
        self.outliers = None
        self.labels = None
        self.dbscan_history = None
        self.dbscan_min_samples = None

    def find_shapes(self, delta=0.01, min_samples=10, plot_history=True):
        epsilon = 0.01
        outlier_frac = 1.0
        ix = 0
        history = pd.DataFrame(columns=['epsilon', 'clusters', 'outliers'])
        while outlier_frac > 0.005:
            self.cluster(method='dbscan', eps=epsilon, min_samples=min_samples)
            clusters = len(set(self.labels))
            outlier_frac = np.sum(self.labels == -1) / len(self.labels)
            row = [epsilon, clusters, outlier_frac]
            history.loc[ix] = row
            epsilon += delta
            ix += 1
        query = history['outliers'] <= 0.4
        number_clusters = history[query]['clusters'].max()
        best_eps = history[history['clusters'] == number_clusters]['epsilon'].max()
        self.cluster(method='dbscan', eps=best_eps, min_samples=min_samples)
        self.dbscan_history = history
        self.dbscan_min_samples = min_samples
        if plot_history:
            fig, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(history['epsilon'], history['clusters'])
            ax[1].plot(history['epsilon'], history['outliers'])
            for a in ax:
                a.axvline(best_eps, color='red', ls='--', label='selected epsilon')
                a.set_xlabel('DBSCAN epsilon value')
            ax[0].set_ylabel('Number of Clusters')
            ax[1].set_ylabel('Outlier Fraction')
            ax[1].legend()
            return fig

    def set_dbscan_clusters(self, number_clusters):
        if self.dbscan_history is None:
            self.find_shapes(plot_history=False)
        history = self.dbscan_history
        best_eps = history[history['clusters'] == number_clusters][
            'epsilon'].max()
        min_samples = self.dbscan_min_samples
        self.cluster(method='dbscan', eps=best_eps, min_samples=min_samples)

    def cluster(self, method='dbscan', **kwargs):
        if self.features is None:
            print('please run featurize_days first')
            return
        if method == 'dbscan':
            clstr = DBSCAN(**kwargs)
        elif method == 'kmeans':
            clstr = KMeans(**kwargs)
        clstr.fit(self.features[self.outliers == 1])
        self.labels = np.zeros(self.data.shape[1])
        self.labels[self.outliers == -1] = -2
        self.labels[self.outliers == 1] = clstr.labels_

    def view_clusters(self, method='tsne', dim=2, groups=None, x_mark=None, **kwargs):
        if method == 'tsne':
            alg = TSNE(n_components=dim, **kwargs)
        elif method == 'pca':
            alg = PCA(n_components=dim)
        view = alg.fit_transform(self.features)
        fig = plt.figure()
        if dim == 3:
            ax = plt.axes(projection="3d")
        else:
            ax = plt.gca()
        if groups is None:
            ax.scatter(*view.T, marker='.')
        elif groups == 'outliers':
            for v in (1, -1):
                s = self.outliers == v
                ax.scatter(*view[s].T, marker='.', label=v)
            plt.legend()
        elif groups == 'clusters':
            for v in set(self.labels):
                s = self.labels == v
                if v == x_mark:
                    ax.scatter(*view[s].T, marker='x', label=v)
                else:
                    ax.scatter(*view[s].T, marker='.', label=v)
            plt.legend()
        ax.set_xlabel('Comp_1')
        ax.set_ylabel('Comp_2')
        if dim == 3:
            ax.set_zlabel('Comp_3')
        return fig

    def plot_clusters_time_domain(self, ncols=3):
        nrows =int(np.ceil(len(set(self.labels)) / ncols))
        figsize = (2.5 * ncols, 2.5 * nrows)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        labels = list(set(self.labels))
        counter = 0
        for i in range(3):
            for j in range(3):
                try:
                    self.plot_normed_signals_in_cluster(labels[counter],
                                                        axis=ax[i, j])
                    ax[i, j].set_title(
                        'Cluster {}'.format(int(labels[counter])))
                except:
                    pass
                counter += 1
        plt.tight_layout()

    def plot_cluster_labels_in_time(self, show_outliers=False):
        if self.dh is not None:
            xs = self.dh.day_index[self.dh.daily_flags.clear]
        else:
            xs = np.arange(len(self.daily_distance))
        fig = plt.figure()
        if show_outliers:
            plt.plot(xs, self.labels, linewidth=1, marker='.')
        else:
            plt.plot(xs[self.labels >= 0], self.labels[self.labels >= 0],
                     linewidth=1, marker='.')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.xlabel('Time [days]')
        plt.ylabel('Cluster Label')
        return fig

    def plot_normed_signals_in_cluster(self, label, axis=None):
        slct = self.labels == label
        if axis is None:
            fig = plt.figure()
            axis = plt.gca()
            return_fig = True
        else:
            return_fig = False
        subset = self.normed_data[:, slct]
        axis.fill_between(np.arange(2 ** 10),
                         np.percentile(subset, 97.5, axis=1),
                         np.percentile(subset, 2.5, axis=1),
                         color='orange', alpha=0.3)
        axis.plot(np.median(subset, axis=1), linewidth=1, color='blue',
                 alpha=0.5)
        if return_fig:
            return fig
        else:
            return

    def plot_daily_summary(self):
        fig = plt.figure()
        if self.dh is not None:
            xs = self.dh.day_index[self.dh.daily_flags.clear]
        else:
            xs = np.arange(len(self.daily_distance))
        s_tot = self.daily_distance
        plt.plot(xs[self.outliers == 1], s_tot[self.outliers == 1], linewidth=1,
                 marker='.', alpha=0.5, label='daily shade metric')
        plt.plot(xs[self.outliers == -1], s_tot[self.outliers == -1], ls='none',
                 marker='.', alpha=0.5, label='detected outliers')
        plt.xticks(rotation=45)
        plt.legend()
        plt.xlabel('Time [days]')
        plt.ylabel('Shade Metric')
        plt.tight_layout()
        return fig

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
        self.N_ = self.signal1.shape[1]

    def featurize_days(self, stat_funcs='mae', normalize='internal',
                       contamination='auto'):
        if self.signal1 is None or self.signal2 is None:
            self.signal_separation()
        self.N_ = self.signal1.shape[1]
        wvlt_features = freq_domain_features(self.signal2, stat_funcs=stat_funcs)
        self.M_ = wvlt_features.shape[1]
        # raw_features = np.c_[self.signal1, wvlt_features]
        self.scaler1 = StandardScaler()
        self.scaler2 = MinMaxScaler(feature_range=(0, 4))
        if normalize == 'internal':
            self.features = np.c_[
                self.scaler1.fit_transform(self.signal1),
                self.scaler2.fit_transform(wvlt_features)
            ]
        elif normalize == 'external':
            import os
            import pickle
            filepath = __file__.split('/')[:-1]
            if stat_funcs == 'rmse':
                fn1 = os.path.join('/', *filepath,
                                  'fixtures/saved_rooftop_data_scaler1_rmse.pkl')
                fn2 = os.path.join('/', *filepath,
                                   'fixtures/saved_rooftop_data_scaler2_rmse.pkl')
                with open(fn1, 'rb') as f:
                    save_dict1 = pickle.load(f)
                with open(fn2, 'rb') as f:
                    save_dict2 = pickle.load(f)
            elif stat_funcs == 'mae':
                fn1 = os.path.join('/', *filepath,
                                  'fixtures/saved_rooftop_data_scaler1_mae.pkl')
                fn2 = os.path.join('/', *filepath,
                                   'fixtures/saved_rooftop_data_scaler2_mae.pkl')
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
        distance = lambda u, v: (np.linalg.norm((u - v)[:self.N_], 2)
                                 + np.linalg.norm((u - v)[self.N_:], 1))
        clf = LocalOutlierFactor(n_neighbors=50, metric=distance,
                                 contamination=contamination)
        self.outliers = clf.fit_predict(self.features)
        N = self.N_
        s1 = np.sqrt(np.average(np.power(self.features[:, 1:N], 2), axis=1))
        s2 = np.average(np.abs(self.features[:, N:]), axis=1)
        self.daily_distance = s1 + s2
        condition = np.logical_and(self.outliers ==1, self.daily_distance >= 1)
        metric = np.abs(self.daily_distance - 1)
        metric[~condition] = 0
        self.site_score = np.average(metric)

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

    def plot_heatmap(self, normed=True):
        if normed:
            data = self.normed_data
        else:
            data = self.data
        with sns.axes_style('white'):
            fig = plt.figure()
            plt.imshow(data, aspect='auto', interpolation='none',
                       cmap='cividis')
        return fig
