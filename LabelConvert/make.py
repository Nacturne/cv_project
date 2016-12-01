import numpy as np

def binToLabel(v):
    output = np.where(v==1)[0]
    output = output.tolist()
    return output

def labelConvert(pred_file, submission_file):
    #pred_file = 'OneHot/logisticPredictions.npy'
    #submission_file = 'submissions/logisticSubmission.csv'
    bizId = open('test_biz_id.csv', 'r')
    idList = bizId.readlines()
    bizId.close()
    pred = np.load(pred_file)
    row, col = pred.shape
    submission = open(submission_file, 'w')
    submission.write('business_id,labels\n')
    for i in range(row):
        id = idList[i]
        id = id[:-1]
        labels = binToLabel(pred[i,:])
        labels = ' '.join(str(label) for label in labels)
        submission.write(id + ',' + labels + '\n')
    submission.close()

#import os
#files = os.listdir('./Ensemble/')
#
#for file in files:
#    inName = 'Ensemble/' + file
#    outName = 'submissions/' + file[:-4] + '.csv'
#    labelConvert(inName, outName)
#
#labelConvert('OneHot/logisticSvmGooglenetCombine.npy',
#             'submissions/logisticSvmGooglenetCombine.csv')

labelConvert('OneHot/logisticPredictions_googlenet.npy',
             'submissions/logisticPredictions_googlenet.csv')