import numpy as np
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
import scipy.io
import numpy as np

'''
featureFile = 'feature_resnet_train.npy'
componentNumber = 256
compressorName = 'resnet_train_256.pkl'
outFile = 'pcaCompressed_resnet_train_256.mat'



totalData = np.load('features/' + featureFile)
print(totalData.shape)

svd = TruncatedSVD(n_components=componentNumber, algorithm='arpack')
svd.fit(totalData)
outData = svd.transform(totalData)
joblib.dump(svd, 'PCACompress/compressors/' + compressorName)

scipy.io.savemat('PCACompress/compressedFeatures/' + outFile, {'M':outData})


'''

trainData = np.load('../features/feature_resnet_train.npy')
testData =  np.load('../features/feature_resnet_test.npy')

print(trainData.shape)
print(testData.shape)

totalData = np.vstack((trainData, testData))

svd = TruncatedSVD(n_components=256, algorithm='arpack')
svd.fit(totalData)
trainOut = svd.transform(trainData)
testOut  = svd.transform(testData)

#joblib.dump(svd, '.pkl')

scipy.io.savemat('compressedFeatures/pcaTogether_resnet_256.mat',
                 {'train':trainOut, 'test':testOut})





'''
Sample code to save and load Matlab files

import scipy.io
totalData = scipy.io.loadmat('features/test_photo_index.mat')
print(totalData['M'].shape)

vect = np.arange(10)
vect.shape
scipy.io.savemat('np_vector.mat', {'vect':vect})
'''


'''
Convert csv file to .npy file

inFile = open('features/feature_googlenet_test.csv', 'r')
outMatrix = np.empty([236416, 1024], np.float32)
for index, line in enumerate(inFile):
    line = line[0:-1]
    temp = line.split(',')
    temp = map(float, temp)
    outMatrix[index, :] = temp
inFile.close()

np.save('features/feature_googlenet_test.npy', outMatrix)


'''



