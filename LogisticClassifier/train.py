import sklearn.linear_model
import sklearn.metrics
from utils import binToLabel, mean_f1
import time
import numpy as np

startTime = time.time()


valAmount = -1


classNum = 9
logisticModels = ['']*classNum # place holder for the 9 svm models. note that the model is string

for i in range(classNum):
    logisticModels[i] = sklearn.linear_model.LogisticRegression()

X = np.genfromtxt('../BizFeatures/googlenet/BizFeature_googlenet_train_1024_cluster64.csv',
                  dtype=np.float32,
                  delimiter=',')
Y = np.genfromtxt('../BizFeatures/labels/biz_labels.csv',
                  dtype=np.float32,
                  delimiter=',')

row, col = X.shape
end = row - 1 - valAmount

print(X.shape)
print(Y.shape)


for i in range(classNum):
    logisticModels[i].fit(X[:end,:], Y[:end,i])
    endTime = time.time()
    print 'Model d% is done. Time elapsed: 4.2f% s' % (i, endTime - startTime)

# Store the models
from sklearn.externals import joblib

for i in range(len(logisticModels)):
    joblib.dump(logisticModels[i], 'models/logistic_googlenet_'+str(i)+'.pkl')


#valX = X[end+1:row,:]
#valY = Y[end+1:row,:]
#
#print(valX.shape)
#print(valY.shape)
#
#predictions = np.empty([valAmount,9])
#
#for i in range(len(logisticModels)):
#    prediction = logisticModels[i].predict(valX)
#    predictions[:,i] = prediction
#    print(sklearn.metrics.f1_score(valY[:,i],prediction))
#
#
#label_predict = binToLabel(predictions)
#label_true = binToLabel(valY)
#
#
#out = mean_f1(label_true,label_predict)
#
#print('*' * 20)
#print('\n' * 3)
#print(out)

'''
0.721649484536
0.847682119205
0.910344827586
0.622222222222
0.58064516129
0.893617021277
0.938388625592
0.626865671642
0.844221105528
'''