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
from scipy.io import loadmat

import os

def rgb2gray(rgb):
    """ Obtained from: http://stackoverflow.com/questions/12201577/convert-rgb-image-to-grayscale-in-python
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def histeq(im,nbr_bins=255):

   #get image histogram
   imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
   cdf = imhist.cumsum()        # cumulative distribution function
   cdf = 255 * cdf / cdf[-1]    # normalize

   # use linear interpolation of cdf to find new pixel values
   im2 = interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape)

def greyscale_array(array):
    """ Expect array to be (samples x 3 x y-axis x x-axis)

    Returns: (samples x 1 x y-axis x x-axis)
    """
    print "Greyscaling filter on the images..."

    samples = array.shape[0]
    y_axis = array.shape[2]
    x_axis = array.shape[3]

    result = np.zeros((samples, 1, y_axis, x_axis))
    for sample in range(samples):
        image = array[sample].T # dim is (32, 32, 3)
        result[sample][0] = histeq(rgb2gray(image))

    print "Greyscaling filter done!"
    result = result / 255

    #return rgb2gray(array.T).T
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

def edgy(gray):
    binary = threshold_otsu(gray)
    edges1 = gray > binary
    return edges1


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
    train_file = "../data/train_32x32.mat"

    print "Loading training data from: " + train_file

    train = loadmat(train_file)
    y = train['y']
    x = train['X']

    print "Loading training data done!"

    return x, y


x, y = load_train_data()

x = x.T # samples x 32 x 32


# reduce the set to process for now
x = x[:20000]
y = y[:20000]

x_greyscale = greyscale_array(x)

print(x_greyscale.shape)
print(y.shape)

# Remove inner array
y = np.reshape(y, len(y))
# Output index must zero-indexed for python
y = y-1

net2 = NeuralNet(
    layers=[
            ('input', layers.InputLayer),
            ('local1', layers.LocalResponseNormalization2DLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout', layers.dropout),
            ('output', layers.DenseLayer),
            ],

            input_shape=(None, 1, 32, 32),
            conv1_num_filters=8, conv1_filter_size=(4,4), pool1_ds=(2, 2),
            conv2_num_filters=16, conv2_filter_size=(3,3), pool2_ds=(2, 2),
            output_num_units=10, output_nonlinearity=lasagne.nonlinearities.softmax,
            dropout_p=0.1,
            dropout2_p=0.1,
            update_learning_rate=0.04,
            update_momentum=0.2,
            regression=False,
            max_epochs=100,
            verbose=1,
    )

net2.fit(x_greyscale, y)
