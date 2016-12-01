import os
import numpy as np
from sklearn.externals import joblib
import scipy.io as sio

totalData = np.load('../features/feature_resnet_test.npy')
compressor = joblib.load('compressors/resnet_train_256.pkl')
outData = compressor.transform(totalData)

print(outData.shape)
sio.savemat('compressedFeatures/pcaCompressed_resnet_test_256.mat', {'pca256':outData})

