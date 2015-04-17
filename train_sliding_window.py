# ==============================================================================
# DISCLAIMER: This does not work! (for now!)
# ==============================================================================
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano
import lasagne
import pylab

from scipy.io import loadmat

import os


a = loadmat('sliding_window.mat')
x = a['img_array'][:50000]
y = a['is_dt'].T[:50000]

items = x.shape[0]

x_new = np.zeros((x.shape[0], 1, 32, 32))
for item in range(items):
    a = np.zeros((1, 32, 32))
    a[0] = x[item]
    x_new[item] = a / 256.0


print(x.shape)
print(y.shape)


# Remove inner array
y = np.reshape(y, len(y))

net2 = NeuralNet(
    layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],

            input_shape=(None, 1, 32, 32),
            hidden3_num_units=300,
            conv1_num_filters=8, conv1_filter_size=(4,4), pool1_ds=(2,2),
            output_num_units=2, output_nonlinearity=lasagne.nonlinearities.softmax,

            update_learning_rate=0.04,
            update_momentum=0.2,

            regression=False,
            max_epochs=10,
            verbose=1,
    )

net2.fit(x_new, y)

# Save model for future use...
