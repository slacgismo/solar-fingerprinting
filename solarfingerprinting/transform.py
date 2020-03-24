# -*- coding: utf-8 -*-
''' Transform module
This module contains the functions for transforming between the "raw" curve
fit parameters and the normalized fingerprint values. A quantile transform was
trained on a random selection of 100 clear days from 100 unique fixed-tilt
PV sites. This only works with the standard fingerprinting algorithm, using
the quadratically modified Gaussian pulse. All other functions do not have a
trained normal transform.
'''

import numpy as np
from sklearn.preprocessing import QuantileTransformer
import os

filepath = __file__.split('/')[:-1]
QUANTILES = np.loadtxt(os.path.join('/', *filepath, 'fixtures/frozen_quantiles.csv'))
REFERENCES = np.loadtxt(os.path.join('/', *filepath, 'fixtures/frozen_references.csv'))
QT = QuantileTransformer(output_distribution='normal')
QT.quantiles_ = QUANTILES
QT.references_ = REFERENCES

def forward_transform(data):
    data = np.asarray(data)
    try:
        data = data.reshape(-1, 6)
    except ValueError:
        print('Shape of the data should be N by 6 (six fit parameters)')
        return
    transformed = QT.transform(data)
    if transformed.shape[0] == 1:
        transformed = transformed[0]
    return transformed

def inverse_transform(data):
    data = np.asarray(data)
    try:
        data = data.reshape(-1, 6)
    except ValueError:
        print('Shape of the data should be N by 6 (six fingerprint parameters)')
        return
    transformed = QT.inverse_transform(data)
    if transformed.shape[0] == 1:
        transformed = transformed[0]
    return transformed
