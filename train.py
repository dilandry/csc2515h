# ==============================================================================
# DISCLAIMER: This does not work! (for now!)
# ==============================================================================
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

from scipy.io import loadmat

import os

def rgb2gray(rgb):
    """ Obtained from: http://stackoverflow.com/questions/12201577/convert-rgb-image-to-grayscale-in-python
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def greyscale_array(array):
    """ Expect array to be (samples x 3 x y-axis x x-axis)

    Returns: (samples x y-axis x x-axis)
    """
    print "Greyscaling filter on the images..."

    samples = array.shape[0]
    y_axis = array.shape[2]
    x_axis = array.shape[3]

    result = np.zeros((samples, 1, y_axis, x_axis))
    for sample in range(samples):
        image = array[sample].T # dim is (32, 32, 3)
        result[sample][0] = (rgb2gray(image))

    print "Greyscaling filter done!"

    return result

def show_image(array):
    """ Array has to be of dimension (a x b x 3), where 3 represents the 'rgb'
    channels

    TODO: does this work?
    """
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    plt.imshow(array, cmap=cm.Greys_r)
    plt.show()

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
    train_file = "data/train_32x32.mat"

    print "Loading training data from: " + train_file

    train = loadmat(train_file)
    y = train['y']
    x = train['X']

    print "Loading training data done!"

    return x, y


x, y = load_train_data()

x = x.T # samples x 32 x 32

x_greyscale = greyscale_array(x)
print(x_greyscale.shape)

net2 = NeuralNet(
    layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],

            input_shape=(None, 1, 32, 32),
            conv1_num_filters=22, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
            hidden5_num_units=100,
            output_num_units=1, output_nonlinearity=None,

            update_learning_rate=0.01,
            update_momentum=0.2,

            regression=False,
            max_epochs=20,
            verbose=1,
    )

y.shape = (y.shape[0], )
net2.fit(x_greyscale, y)
