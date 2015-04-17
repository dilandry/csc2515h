# ==============================================================================
# code used to do lecun_lcn
# ==============================================================================
from scipy.io import loadmat, savemat
from pylearn2.datasets import preprocessing
import numpy as np
from numpy import *

train_file = "../data/train_32x32.mat"
print "Finished loading data"
train = loadmat(train_file)

y = train['y'].T
x = train['X'].T


items = x.shape[0]
subset_y = y[:items]
subset_x = x[:items]

# need to do the astype to avoid size errors in the end
# it initially considers it as float64
result = np.zeros((items, 3, subset_x.shape[3], subset_x.shape[2])).astype(uint8)

print result.shape

for sample in range(items):
	print sample + 1
	image = subset_x[sample].T # 32 x 32 x 3
	# cast to reduce size of final array
	result[sample] = (preprocessing.lecun_lcn(image.T.astype(float64), (32, 32), 7) * 255).astype(uint8)


print "saving data"
savemat("lcn_transform.mat", {'X': result.astype(uint8), 'y': subset_y})
