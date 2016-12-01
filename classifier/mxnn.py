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
            colNumber,
            classNumber,
            learningRate,
            training_epochs,
            batch_size,
            display_step,
            labelThreshold,
            keepProb):
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
        'h0': weightsIniti([colNumber, 2500]),
        'h1': weightsIniti([2500, 1024]),
        'h2': weightsIniti([1024, 365]),
        'h3': weightsIniti([365, classNumber])
    }

    biases = {
        'b0': biasIniti([2500]),
        'b1': biasIniti([1024]),
        'b2': biasIniti([365]),
        'b3': biasIniti([classNumber])
    }

    keepProbability = tf.placeholder("float")

    #################################
    # Construct the model
    #################################
    layer_0 = tf.matmul(images, weights['h0']) + biases['b0']
    out_0 = tf.nn.relu(layer_0)
    out_0 = tf.nn.dropout(out_0, keepProbability)

    layer_1 = tf.matmul(out_0, weights['h1']) + biases['b1']
    out_1 = tf.nn.relu(layer_1)
    out_1 = tf.nn.dropout(out_1, keepProbability)

    layer_2 = tf.matmul(out_1, weights['h2']) + biases['b2']
    out_2 = tf.nn.relu(layer_2)
    out_2 = tf.nn.dropout(out_2, keepProbability)

    layer_final = tf.matmul(out_2, weights['h3']) + biases['b3']
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
        for i in range(total_batch-1):
            batch_x = input[i*batch_size:(i+1)*batch_size-1, :]
            batch_y = output[i*batch_size:(i+1)*batch_size-1, :]
            sess.run(optimizer, feed_dict={images: batch_x, labelsTrue: batch_y, keepProbability: keepProb})

        if epoch % display_step == 0:
            yPredicted = sess.run(out_final, feed_dict={images: input, labelsTrue: output, keepProbability: 1.0})
            yPredicted[yPredicted < labelThreshold] = 0
            yPredicted[yPredicted != 0] = 1

            yPredicted = binToLabel(yPredicted.astype(int))
            yTrue = binToLabel(output)
            meanF1 = mean_f1(yTrue, yPredicted)
            print('Epoch: {0}\nThe Mean F1 Score is {1:.4f} '.format(epoch, meanF1))

    print("Optimization Finished!")
    print('%'*80 + '\n'*3)

    testIn = input[1900:1999,:]
    testOut = output[1900:1999,:]
    yPredicted = sess.run(out_final, feed_dict={images: testIn, labelsTrue: testOut, keepProbability: 1.0})
    yPredicted[yPredicted < labelThreshold] = 0
    yPredicted[yPredicted != 0] = 1

    yPredicted = binToLabel(yPredicted.astype(int))
    yTrue = binToLabel(testOut)
    meanF1 = mean_f1(yTrue, yPredicted)
    print('The Validation Mean F1 Score is {:.4f} '.format(meanF1))

    sess.close()
    return meanF1


