import tensorflow as tf
import numpy as np

def weightsIniti(shape):
    weightsMatrix = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weightsMatrix)

def biasIniti(shape):
    biasMatrix = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(biasMatrix)



rowNumber = 64
colNumber = 64
classNumber = 9
learningRate = 1e-4
training_epochs = 700
batch_size = 100
display_step = 1
labelThreshold = 0.3

#################################
# Get the training data:
#################################
input = np.genfromtxt('data/BizFeature_reset_train_pca64_cluster64.csv',delimiter=',',dtype=None)
output = np.genfromtxt('data/biz_labels.csv',delimiter=',',dtype=None)

totalNumber = input.shape[0]



#################################
# Training Data placeholder
#################################
images = tf.placeholder(dtype=tf.float32, shape=[None, rowNumber*colNumber])
imagesRec = tf.reshape(tensor=images, shape=[-1, rowNumber, colNumber, 1])
labelsTrue = tf.placeholder(dtype=tf.float32, shape=[None, classNumber])



#################################
# Weights and Bias for different layers
#################################
fullyRow = int(np.ceil(rowNumber/4))
fullyCol = int(np.ceil(colNumber/4))
weights = {
    'h0': weightsIniti([5, 5, 1, 32]),
    'h1': weightsIniti([5, 5, 32, 64]),
    'h2': weightsIniti([fullyRow*fullyCol*64, 1024]),
    'h3': weightsIniti([1024, classNumber])


}

biases = {
    'b0': biasIniti([32]),
    'b1': biasIniti([64]),
    'b2': biasIniti([1024]),
    'b3': biasIniti([classNumber])
}


#################################
# Construct the model
#################################
# Input layer:
convOut_0 = tf.nn.conv2d(input=imagesRec, filter=weights['h0'], strides=[1, 1, 1, 1], padding='SAME')
poolOut_0 = tf.nn.max_pool(value=convOut_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
out_0 = tf.nn.relu(tf.add(poolOut_0, biases['b0']))

# Layer 1:
convOut_1 = tf.nn.conv2d(input=out_0, filter=weights['h1'], strides=[1, 1, 1, 1], padding='SAME')
poolOut_1 = tf.nn.max_pool(value=convOut_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
out_1 = tf.nn.relu(tf.add(poolOut_1, biases['b1']))

# Layer 2: Fully connected
out_1 = tf.reshape(tensor=out_1, shape=[-1, fullyRow*fullyCol*64])
out_2 = tf.nn.relu(tf.matmul(out_1, weights['h2']) + biases['b2'])

# Layer 3: Output Layer
out_3 = tf.nn.sigmoid(tf.matmul(out_2, weights['h3']) + biases['b3'])


#################################
# Define Cost Function:
#################################
costFunction = tf.reduce_sum(tf.square(out_3 - labelsTrue))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(costFunction)



# Initializing the variables
init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)




from utils import binToLabel
from utils import mean_f1



for epoch in range(training_epochs):
    total_batch = int(totalNumber / batch_size)
    # Loop over all batches
    for i in range(total_batch-1):
        batch_x = input[i*batch_size:(i+1)*batch_size-1, :]
        batch_y = output[i*batch_size:(i+1)*batch_size-1, :]
        sess.run(optimizer, feed_dict={images: batch_x, labelsTrue: batch_y})

    if epoch % display_step == 0:
        yPredicted = sess.run(out_3, feed_dict={images: input, labelsTrue: output})
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
yPredicted = sess.run(out_3, feed_dict={images: testIn, labelsTrue: testOut})
yPredicted[yPredicted < labelThreshold] = 0
yPredicted[yPredicted != 0] = 1

yPredicted = binToLabel(yPredicted.astype(int))
yTrue = binToLabel(testOut)
meanF1 = mean_f1(yTrue, yPredicted)
print('The Validation Mean F1 Score is {:.4f} '.format(meanF1))


sess.close()
exit(0)


