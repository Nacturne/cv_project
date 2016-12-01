import numpy as np
'''
logistic = np.load('OneHot/logisticPredictions.npy')
svm = np.load('OneHot/svmPredictions.npy')
row, col = svm.shape
result = np.empty([row, col], dtype=np.int8)

for i in [0, 1, 2, 3, 8]:
    result[:,i] = logistic[:,i]

for i in [4, 5, 6, 7]:
    result[:,i] = svm[:,i]

np.save('OneHot/logisticSvmCombine.npy', result)
'''

previouse = np.load('OneHot/logisticSvmCombine.npy')
googlenet = np.load('OneHot/logisticPredictions_googlenet.npy')
row, col = googlenet.shape

previouse[:,3] = googlenet[:,3]
#previouse[:,8] = googlenet[:,8]

np.save('OneHot/logisticSvmGooglenetCombine.npy', previouse)