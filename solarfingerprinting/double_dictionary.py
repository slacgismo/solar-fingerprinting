"""Dictionary Generation Module

This module contains functions for generating two incomplete orthonormal basis
matrices.

"""

import numpy as np
import pywt
import os
from pathlib import Path

filepath = Path(__file__).parent
file_open = filepath / 'fixtures' / 'envelope.npy'
ENVELOPE = np.load(file_open)
os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         'fixtures', 'envelope.npy'))
WVLT = pywt.Wavelet('sym2').wavefun

def make_dictionaries(envelope=None, max_n=5, wavefun=WVLT, J=10,
                      normalize=True, reduce_level=None):
    """
    Convenience function for generating both the smooth dictionary "W1" and the
    sparse wavelet dictionary "W2".

    :param envelope: The reference signal shape
    :param max_n: Order of sines and cosines to include, produces 2*n vectors
    :param wavefun: A wavelet function object from PyWavelets
    :param J: Number of measurements after resampling: 2 ** J
    :param normalize: If true, enforce orthonormality of basis matrices
    :param reduce_level: Number of hierarchical levels to remove from wavelet basis
    :return: Two basis metrices, W1 and W2
    """
    if envelope is None:
        envelope = ENVELOPE
    W1 = make_smooth_basis(envelope, max_n=max_n, normalize=normalize)
    W2 = make_wavelet_basis(wavefun, J=J, normalize=normalize,
                            reduce_level=reduce_level)
    return W1, W2

def make_wavelet_basis(wavefun=WVLT, J=10, normalize=True, reduce_level=None):
    """
    Function to construct an orthonormal wavelet basis matrix from PyWavelets
    wavelet function object. Any discrete wavelet function from this package
    may be used. Dimension of the domain is n = (2 ** J). If reduce_level is
    not used, this returns a complete orthonormal basis, a matrix
    of size (n ⨉ n). If reduce_level is used, then the levels corresponding to
    the smallest time scales are removed. For instance, if J=10 and
    reduce_level=1, then a (1024 x 512) matrix is returned. J=10 and
    reduce_level=2, than a (1024 X 256) matrix is returned.

    The vectors produced by PyWavelets are approximations, and Gramm-Schmitt
    may be used to "polish" the basis and make it truly orthonormal by setting
    normalize=True.


    :param wavefun: A wavelet function object from PyWavelets
    :param J: Number of measurements after resampling: 2 ** J
    :param normalize: If true, enforce orthonormality of basis matrix
    :param reduce_level: Number of hierarchical levels to remove from wavelet basis
    :return: Wavelet basis matrix
    """
    n = 2 ** J
    W = np.zeros((n, n))
    column_ix = 0
    for level in range(1, J + 1):
#         print(level)
        vec = np.zeros(n)
        _, wf, _ = wavefun(level=level)
        wf = wf[~np.isclose(wf, 0, atol=1e-3)]
        m = len(wf)
#         print(m)
        if m <= n:
            vec[:m] = wf
        else:
            level -= 1
            break
#             vec[:] = wf[m//2-n//2:m//2+n//2]
        num_vecs = 2 ** (J - level)
        for ix in range(int(num_vecs)):
            roll_amt = (2 ** level) * ix
            W[:, column_ix] = np.roll(vec, roll_amt)
#             print('column', column_ix)
            column_ix += 1
    sf, _, _ = wavefun(level=level)
    sf = sf[~np.isclose(sf, 0, atol=1e-3)]
    m = len(sf)
    vec = np.zeros(n)
    if m <= n:
        vec[:m] = sf
    else:
        vec[:] = sf[m//2-n//2:m//2+n//2]
#     W[:, column_ix] = vec
    num_vecs = 2 ** (J - level)
    for ix in range(int(num_vecs)):
        roll_amt = (2 ** level) * ix
        W[:, column_ix] = np.roll(vec, roll_amt)
#             print('column', column_ix)
        column_ix += 1
    if reduce_level is not None:
        start_ix = np.sum([2 ** (J - k - 1) for k in range(reduce_level)])
        W = W[:, start_ix:]
    if normalize:
        q, r = np.linalg.qr(W)
        W = q
        W *= np.sign(np.diag(r))
        # Original wavelets are very sparse, but QR decomposition breaks the
        # sparsity. Sparsity helps speed up algorithm by orders of magnitude,
        # so we reintroduce the sparsity here
        W[np.abs(W) <= 1e-3] = 0
    # normvals = np.linalg.norm(W, axis=0)
    # W /= normvals
    return W

def make_smooth_basis(envelope=ENVELOPE, max_n=5, normalize=True):
    """
    Function to make a smooth, orthonormal basis, generated as sin and cos
    functions windowed by a parent "envelope". The parent envelope is typically
    the "average" response of a system

    :param envelope: The reference signal shape
    :param max_n: Order of sines and cosines to include, produces 2*n vectors
    :param normalize: If true, enforce orthonormality of basis matrix
    :return: smooth, orthonormal basis matrix
    """
    W = [envelope]
    n = len(envelope)
    xs = np.linspace(0, n, n)
    for i in range(1, max_n + 1):
        W.append(envelope * np.sin(xs * 2 * i * np.pi / n))
        W.append(envelope * np.cos(xs * 2 * i * np.pi / n))
    M1 = np.r_[W].T
    if normalize:
        q, r = np.linalg.qr(M1)
        M1 = q
        if np.average(M1[:, 0]) < 0:
            M1 *= -1
    return M1