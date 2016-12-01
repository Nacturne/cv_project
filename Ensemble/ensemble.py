import numpy as np



def ensemble(pred_1, pred_2, weights=[0.5, 0.5], threshold=0.5):
    result = pred_1 * weights[0] + pred_2 * weights[1]
    result[result < threshold] = 0
    result[result != 0] = 1
    return result


logistic = np.load('data/logisticPredictionsProb.npy')
nn = np.load('data/nnPredictionsProb.npy')
threshold = 0.5

for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    j = 1 - i
    result = ensemble(pred_1=logistic, pred_2=nn, weights=[i, j], threshold=threshold)
    outFile = '../LabelConvert/Ensemble/ensemble_logistic_nn_' + \
                str(int(10*i)) + '_' + str(int(10*j)) + '_' + str(int(10*threshold)) + '.npy'
    np.save(outFile, result)