# ==============================================================================
# DISCLAIMER: This does not work! (for now!)
# ==============================================================================
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
from numpy import histogram, interp
import theano
import lasagne
import pylab
from scipy.ndimage.filters import gaussian_filter
from skimage import feature
from skimage.filters import roberts, sobel, scharr, threshold_otsu
from scipy.io import loadmat, savemat
from pylearn2.datasets import preprocessing
import os
from numpy import float64


a = loadmat('lcn_transform.mat')

x = a['X']
y = a['y'].T

print x.shape
print y.shape


items = 10000
# reduce the set to process for now
x_greyscale = x[:items]
y = y[:items]

# Remove inner array
y = np.reshape(y, len(y))
# Output index must zero-indexed for python
y = y - 1

net2 = NeuralNet(
    layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('output', layers.DenseLayer),
            ],

            input_shape=(None, 3, 32, 32),
            conv1_num_filters=8, conv1_filter_size=(5,5), pool1_ds=(3, 3),
            conv2_num_filters=16, conv2_filter_size=(7,7), pool2_ds=(3, 3),
            output_num_units=10, output_nonlinearity=lasagne.nonlinearities.sigmoid,
            update_learning_rate=0.04,
            update_momentum=0.2,
            regression=False,
            max_epochs=100,
            verbose=1,
    )

net2.fit(x_greyscale.astype(float64)/256, y)