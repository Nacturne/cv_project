import numpy as np


inFile = '../BizFeatures/resnet/BizFeature_resnet_together_test_pca256_cluster64.csv'
outFile = '../Ensemble/data/logisticPredictionsProb.npy'

X = np.genfromtxt(inFile,
                  dtype=np.float32,
                  delimiter=',')
row, col = X.shape


from sklearn.externals import joblib

predictions = np.empty([row,9])

for i in range(9):
    svm = joblib.load('models/logistic_'+str(i)+'.pkl')
    prediction = svm.predict_proba(X)
    predictions[:,i] = prediction[:,1]

np.save(outFile, predictions)

print(predictions.shape)


