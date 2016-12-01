import nn
import numpy as np

predictions = nn.train( xFile='../BizFeatures/resnet/BizFeature_resnet_together_train_pca256_cluster64.csv',
                        yFile='../BizFeatures/labels/biz_labels.csv',
                        testFile='../BizFeatures/resnet/BizFeature_resnet_together_test_pca256_cluster64.csv',
                        colNumber = 256*64,
                        classNumber = 9,
                        learningRate = 1e-4,
                        training_epochs = 700,
                        batch_size = 100,
                        display_step = 5,
                        labelThreshold = 0.3,
                        returnProb = True)

print(predictions.shape)
#np.save('../LabelConvert/OneHot/nnPredictions.npy', predictions)
np.save('../Ensemble/data/nnPredictionsProb.npy', predictions)