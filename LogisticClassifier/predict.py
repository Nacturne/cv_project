import numpy as np


inFile = open('../BizFeatures/googlenet/BizFeature_googlenet_test_1024_cluster64.csv', 'r')
out_1 = open('../BizFeatures/googlenet/googlenet_test_1.csv', 'w')
out_2 = open('../BizFeatures/googlenet/googlenet_test_2.csv', 'w')
out_3 = open('../BizFeatures/googlenet/googlenet_test_3.csv', 'w')


for index, line in enumerate(inFile):
    if index < 3000:
        out_1.write(line)
    elif index < 6000:
        out_2.write(line)
    else:
        out_3.write(line)


out_1.close()
out_2.close()
out_3.close()
inFile.close()



from sklearn.externals import joblib


X = np.genfromtxt('../BizFeatures/googlenet/googlenet_test_1.csv',
                    dtype=np.float32,
                    delimiter=',')
row, col = X.shape
print(X.shape)
predictions_1 = np.empty([row,9])

for i in range(9):
    svm = joblib.load('models/logistic_googlenet_'+str(i)+'.pkl')
    prediction = svm.predict(X)
    predictions_1[:,i] = prediction





X = np.genfromtxt('../BizFeatures/googlenet/googlenet_test_2.csv',
                    dtype=np.float32,
                    delimiter=',')
row, col = X.shape
print(X.shape)
predictions_2 = np.empty([row,9])

for i in range(9):
    svm = joblib.load('models/logistic_googlenet_'+str(i)+'.pkl')
    prediction = svm.predict(X)
    predictions_2[:,i] = prediction



X = np.genfromtxt('../BizFeatures/googlenet/googlenet_test_3.csv',
                    dtype=np.float32,
                    delimiter=',')
row, col = X.shape
print(X.shape)
predictions_3 = np.empty([row,9])

for i in range(9):
    svm = joblib.load('models/logistic_googlenet_'+str(i)+'.pkl')
    prediction = svm.predict(X)
    predictions_3[:,i] = prediction

del X

predictions = np.vstack((predictions_1, predictions_2, predictions_3))
np.save('../LabelConvert/OneHot/logisticPredictions_googlenet.npy', predictions)

print(predictions.shape)


