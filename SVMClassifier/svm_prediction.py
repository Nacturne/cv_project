from sklearn import datasets, metrics
import numpy as np
from utils import binToLabel





X = np.genfromtxt('../BizFeatures/resnet/BizFeature_resnet_together_test_pca256_cluster64.csv',
                  dtype=np.float32,
                  delimiter=',')
row, col = X.shape
#Y = np.genfromtxt('biz_label.csv', dtype=np.float32, delimiter=',')
#Y = Y[0:521,1:10] # take col 2 to col 9
#print(X.shape)
#print(Y.shape)



from sklearn.externals import joblib

predictions = np.empty([row,9])

for i in range(9):
    svm = joblib.load('models/svm_'+str(i)+'.pkl')
    prediction = svm.predict(X)
    predictions[:,i] = prediction
    #print(metrics.f1_score(Y[400:521,i],prediction))

np.save('../LabelConvert/OneHot/svmPredictions.npy', predictions)

print(predictions.shape)

#label_predict = binToLabel(predictions)
#label_true = binToLabel(Y[400:521,:])
#
#
#
#from utils import mean_f1
#
#out = mean_f1(label_true,label_predict)
#
#print('*' * 20)
#print('\n' * 3)
#print(out)
#
'''
svm modes: 0.740608106311
'''
