"""

"""

import numpy as np

def featurize(smooth_params, wavelet_params):
    pass

def time_domain_features(fit_parameters, basis_matrix, levels=3,
                         penalty_function=None):
    if penalty_function is None:
        penalty_function = lambda x: np.sum(np.abs(x), axis=0)
    W2 = basis_matrix
    wvlt_features = []
    wvlt_data = W2.dot(fit_parameters.T)
    for j in range(levels):
        num_segments = int(2 ** j)
        len_segments = wvlt_data.shape[0] / (2 ** j)
        for seg in range(num_segments):
            i_start = int(seg * len_segments)
            i_end = int((seg + 1) * len_segments)
            view = wvlt_data[i_start:i_end]
            wvlt_features.append(penalty_function(view))
    wvlt_features = np.r_[wvlt_features].T
    return wvlt_features

def freq_domain_features(fit_parameters, stat_funcs=None):
    if stat_funcs is None:
        stat_funcs = [
            # lambda x: np.sqrt(np.average(np.power(x, 2), axis=1))
            lambda x: np.average(np.abs(x), axis=1)
        ]
    elif stat_funcs == 'rmse':
        stat_funcs = [
            lambda x: np.sqrt(np.average(np.power(x, 2), axis=1))
            # lambda x: np.average(np.abs(x), axis=1)
        ]
    elif stat_funcs == 'mae':
        stat_funcs = [
            # lambda x: np.sqrt(np.average(np.power(x, 2), axis=1))
            lambda x: np.average(np.abs(x), axis=1)
        ]
    ix_start = 0
    subbands = []
    p = fit_parameters.shape[1]
    for j in range(int(np.log2(p)), 0, -1):
        ix_end = ix_start + 2 ** (j - 1)
        if 2 ** (j - 1) >= 4:
            subbands.append(np.s_[ix_start:ix_end])
        else:
            subbands.append(np.s_[ix_start:ix_start + 4])
            break
        ix_start = ix_end
    wvlt_features = []
    for subb in subbands:
        data_view = fit_parameters[:, subb]
        summaries = [f(data_view) for f in stat_funcs]
        wvlt_features.extend(summaries)
    wvlt_features = np.r_[wvlt_features].T
    return wvlt_features