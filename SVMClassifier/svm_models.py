from sklearn import svm, metrics
from utils import binToLabel, mean_f1
import time
import numpy as np

startTime = time.time()


valAmount = 150


classNum = 9
svm_models = ['']*classNum # place holder for the 9 svm models. note that the model is string

for i in range(classNum):
    svm_models[i] = svm.LinearSVC()

X = np.genfromtxt('../BizFeatures/resnet/BizFeature_resnet_together_train_pca256_cluster64.csv',
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
    svm_models[i].fit(X[:end,:], Y[:end,i])
    endTime = time.time()
    print 'Model d% is done. Time elapsed: 4.2f% s' % (i, endTime - startTime)

# Store the models
#from sklearn.externals import joblib
#
#for i in range(len(svm_models)):
#    joblib.dump(svm_models[i], 'models/svm_'+str(i)+'.pkl')



valX = X[end+1:row,:]
valY = Y[end+1:row,:]

print(valX.shape)
print(valY.shape)

predictions = np.empty([valAmount,9])

for i in range(len(svm_models)):
    prediction = svm_models[i].predict(valX)
    predictions[:,i] = prediction
    print(metrics.f1_score(valY[:,i],prediction))


label_predict = binToLabel(predictions)
label_true = binToLabel(valY)


out = mean_f1(label_true,label_predict)

print('*' * 20)
print('\n' * 3)
print(out)


'''
0.704761904762
0.838709677419
0.868965517241
0.602941176471
0.676470588235
0.913978494624
0.946859903382
0.701298701299
0.825396825397
'''