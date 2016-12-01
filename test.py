import numpy as np
import os
print os.getcwd()

import scipy.io


'''
y_1 = scipy.io.loadmat('temp/feat_resnet_test1_50000.mat')
y_2 = scipy.io.loadmat('temp/feat_resnet_test50001_100000.mat')
y_3 = scipy.io.loadmat('temp/feat_resnet_test100001_150000.mat')
y_4 = scipy.io.loadmat('temp/feat_resnet_test150001_200000.mat')
y_5 = scipy.io.loadmat('temp/feat_resnet_test200001_236416.mat')
y_6 = scipy.io.loadmat('temp/feat_resnet_test_more.mat')

output = np.vstack((y_1['feat'], y_2['feat'], y_3['feat'], y_4['feat'], y_5['feat'], y_6['feat']))
np.save('features/feature_resnet_test.npy', output)


'''
x = np.genfromtxt('raw_data/test_photo_to_biz.csv',dtype=None, delimiter=',')
print(x[:4,:])
