"""

"""

import numpy as np
import pywt
import os

filepath = __file__.split('/')[:-1]
ENVELOPE = np.load(os.path.join('/', *filepath, 'fixtures/envelope.npy'))
WVLT = pywt.Wavelet('sym2').wavefun

def make_dictionaries(envelope=ENVELOPE, max_n=5, wavefun=WVLT, J=10,
                      normalize=True, reduce_level=None):
    W1 = make_smooth_basis(envelope, max_n=max_n, normalize=normalize)
    W2 = make_wavelet_basis(wavefun, J=J, normalize=normalize,
                            reduce_level=reduce_level)
    return W1, W2

def make_wavelet_basis(wavefun=WVLT, J=10, normalize=True, reduce_level=None):
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
        # sparsity. Sparsity helps speed up algorithm by orders of magnitude, \
        # so we reintroduce the sparsity here
        W[np.abs(W) <= 1e-3] = 0
    # normvals = np.linalg.norm(W, axis=0)
    # W /= normvals
    return W

def make_smooth_basis(envelope=ENVELOPE, max_n=5, normalize=True):
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