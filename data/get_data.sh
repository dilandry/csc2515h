#!/bin/bash
# ##############################################################################
# Download the data required to run the program
# For now we are going to download the Format 2 data, which are easier to train
#
# Website: http://ufldl.stanford.edu/housenumbers/
# ##############################################################################

# Get Format 2 data
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
