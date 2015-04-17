## csc2515h class project housenumber

### General Structure
Python code for the digit recognition is in folder "recognition". 
MATLAB and Python code for the digit segmentation is in folder "segmentation"

### Libraries Used
For now, we'll be using _Python_ and the
[Lasagne](https://github.com/benanne/Lasagne) library for developing our Neural
Network, together with Theano, numpy, and scipy.

MATLAB was also used.

### Setup
If your `data` folder is empty, please run `get_data.sh` from the `data` folder
to download the training and test set for Format 2 of the housenumber stuff.

```
# now let's install lasagne
# lasagne is not yet on PyPi, have to install manually
git clone https://github.com/benanne/Lasagne.git
cd Lasagne
pip install -r requirements.txt
python setup.py install
cd ..

# nolearn is a wrapper among a long of python libraries
pip install nolearn

# install matplotlib to plot stuff
pip install matplotlib
```

Also, install pylearn2.

# Lecun Transform
TO use the dataset for lecun, download the mat file from https://www.dropbox.com/s/povwqf7mewf3yza/lcn_transform.mat?dl=0
