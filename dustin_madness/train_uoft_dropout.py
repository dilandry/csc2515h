# ==============================================================================
# From http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
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

def load_train_data():
    """ Load training data from data folder

    'x' represents the images. We have 3 dimensions for 'x' for the 'rgb'.
    Each image in 'x' is represented as a '32x32' matrix

    To access the 5th image for example, use:
                x[:, :, :, 4]

    dim x: (32, 32, 3, 73257)
    dim y: (73257, 1)
    returns: x, y
    """
    #train_file = "data/train_32x32.mat"
    train_file = "../../data/train_32x32.mat"

    print "Loading training data from: " + train_file

    train = loadmat(train_file)
    y = train['y']
    x = train['X']

    print "Loading training data done!"

    return x, y

def normalize_input(x):
    # x is (samples, 32, 32)
    # TODO: normalize
    x_processed = x / 256.0
    return x_processed



x, y = load_train_data()

x = x.T # samples x 32 x 32

print x.shape
print y.shape

x_processed = normalize_input(x)

print(x_processed.shape)
print(y.shape)

# Remove inner array
y = np.reshape(y, len(y))
# Output index must zero-indexed for python
y = y-1

uoft = NeuralNet(
    layers=[
            ('input', layers.InputLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('dropout6', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],

            input_shape=(None, 1, 32, 32),
            conv1_num_filters=96, conv1_filter_size=(5,5), pool1_ds=(3, 3),
            conv2_num_filters=128, conv2_filter_size=(5,5), pool2_ds=(3, 3),
            conv3_num_filters=256, conv3_filter_size=(5,5), pool3_ds=(3, 3),
            hidden1_num_units=2048, hidden2_num_units=2048,
            dropout1_p=0.9, dropout2_p=0.75, dropout3_p=0.75, dropout4_p=0.5, dropout5_p=0.5, dropout6_p=0.5,
            output_num_units=10, output_nonlinearity=lasagne.nonlinearities.softmax,

            update_learning_rate=0.04,
            update_momentum=0.95,

            regression=False,
            max_epochs=100,
            verbose=1,
    )

uoft.fit(x_processed, y)

# Save model for future use...
uoft.save_weights_to('./uoft.dat')

import cPickle as pickle
with open('uoft.pickle', 'wb') as f:
    pickle.dump(uoft, f, -1)
