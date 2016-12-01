import tensorflow as tf
import numpy as np
from utils import binToLabel
from utils import mean_f1

def weightsIniti(shape):
    weightsMatrix = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weightsMatrix)

def biasIniti(shape):
    biasMatrix = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(biasMatrix)

def train(  xFile,
            yFile,
            testFile,
            colNumber,
            classNumber,
            learningRate,
            training_epochs,
            batch_size,
            display_step,
            labelThreshold,
            returnProb):
    #################################
    # Get the training data:
    #################################
    input = np.genfromtxt(xFile,delimiter=',',dtype=None)
    output = np.genfromtxt(yFile,delimiter=',',dtype=None)

    totalNumber = input.shape[0]


    #################################
    # Training Data placeholder
    #################################
    images = tf.placeholder(dtype=tf.float32, shape=[None, colNumber])
    labelsTrue = tf.placeholder(dtype=tf.float32, shape=[None, classNumber])


    #################################
    # Weights and Bias for different layers
    #################################
    weights = {
        'h0': weightsIniti([colNumber, 32]),
        'h1': weightsIniti([32, 16]),
        'h2': weightsIniti([16, classNumber])
    }

    biases = {
        'b0': biasIniti([32]),
        'b1': biasIniti([16]),
        'b2': biasIniti([classNumber])
    }

    #################################
    # Construct the model
    #################################
    layer_0 = tf.matmul(images, weights['h0']) + biases['b0']
    out_0 = tf.nn.tanh(layer_0)

    layer_1 = tf.matmul(out_0, weights['h1']) + biases['b1']
    out_1 = tf.nn.tanh(layer_1)


    layer_final = tf.matmul(out_1, weights['h2']) + biases['b2']
    out_final = tf.nn.sigmoid(layer_final)


    #################################
    # Define Cost Function:
    #################################
    costFunction = tf.reduce_sum(tf.square(out_final - labelsTrue))
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(costFunction)

    # Initializing the variables
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for epoch in range(training_epochs):
        total_batch = int(totalNumber / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = input[i*batch_size:(i+1)*batch_size-1, :]
            batch_y = output[i*batch_size:(i+1)*batch_size-1, :]
            sess.run(optimizer, feed_dict={images: batch_x, labelsTrue: batch_y})

        if epoch % display_step == 0:
            yPredicted = sess.run(out_final, feed_dict={images: input, labelsTrue: output})
            yPredicted[yPredicted < labelThreshold] = 0
            yPredicted[yPredicted != 0] = 1

            yPredicted = binToLabel(yPredicted.astype(int))
            yTrue = binToLabel(output)
            meanF1 = mean_f1(yTrue, yPredicted)
            print('Epoch: {0}\nThe Mean F1 Score is {1:.4f} '.format(epoch, meanF1))

    print("Optimization Finished!")
    print('%'*80 + '\n'*3)


    testIn = np.genfromtxt(testFile, delimiter=',', dtype=None)
    predictedLabels = sess.run(out_final, feed_dict={images: testIn})
    if not returnProb:
        predictedLabels[predictedLabels < labelThreshold] = 0
        predictedLabels[predictedLabels != 0] = 1

    sess.close()
    return predictedLabels


'''
testIn = input[1900:1999,:]
    testOut = output[1900:1999,:]
    yPredicted = sess.run(out_final, feed_dict={images: testIn, labelsTrue: testOut})
    yPredicted[yPredicted < labelThreshold] = 0
    yPredicted[yPredicted != 0] = 1

    yPredicted = binToLabel(yPredicted.astype(int))
    yTrue = binToLabel(testOut)
    meanF1 = mean_f1(yTrue, yPredicted)
    print('The Validation Mean F1 Score is {:.4f} '.format(meanF1))

'''