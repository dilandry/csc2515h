## csc2515h class project housenumber


### Libraries Used
For now, we'll be using _Python_ and the
[Lasagne](https://github.com/benanne/Lasagne) library for developing our Neural
Network, together with Theano, numpy, and scipy.

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
