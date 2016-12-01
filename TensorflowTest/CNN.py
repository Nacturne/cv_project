import tensorflow as tf
import numpy as np
import input_data

def weightsIniti(shape):
    weightsMatrix = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weightsMatrix)

def biasIniti(shape):
    biasMatrix = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(biasMatrix)


#################################
# Training Data placeholder
#################################
imagesBatch = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
inputData = tf.reshape(tensor=imagesBatch, shape=[-1, 28, 28, 1])
outputData = tf.placeholder(dtype=tf.float32, shape=[None, 10])


#################################
# Input Layer
#################################
W_0 = weightsIniti([5, 5, 1, 32])
b_0 = biasIniti([32])

convOut_0 = tf.nn.conv2d(input=inputData, filter=W_0, strides=[1, 1, 1, 1], padding='SAME')
# 1x28x28x32
poolOut_0 = tf.nn.max_pool(value=convOut_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 1x14x14x32
out_0 = tf.nn.relu(poolOut_0 + b_0)
# 1x14x14x32


#################################
# Layer 1:
#################################
W_1 = weightsIniti([5, 5, 32, 64])
b_1 = biasIniti([64])

convOut_1 = tf.nn.conv2d(input=out_0, filter=W_1, strides=[1, 1, 1, 1], padding='SAME')
# 1x14x14x64
poolOut_1 = tf.nn.max_pool(value=convOut_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 1x7x7x64
out_1 = tf.nn.relu(poolOut_1 + b_1)
# 1x7x7x64


#################################
# Layer 2: Fully connected
#################################
out_1 = tf.reshape(tensor=out_1, shape=[-1, 7*7*64])
# 1x(7*7*64)

W_2 = weightsIniti([7*7*64, 1024])
b_2 = biasIniti([1024])

out_2 = tf.nn.relu(tf.matmul(out_1, W_2) + b_2)
# 1x1024



#################################
# Layer 3:
#################################
W_3 = weightsIniti([1024, 10])
b_3 = biasIniti([10])

out_3 = tf.nn.softmax(tf.matmul(out_2, W_3) + b_3)
# 1x10




#################################
# Training:
#################################
#cross_entropy = -tf.reduce_sum(outputData * tf.log(out_3))
cross_entropy = tf.reduce_sum(tf.square(outputData-out_3))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(out_3,1), tf.argmax(outputData,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
for i in range(10000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = sess.run(accuracy, feed_dict={imagesBatch: batch[0], outputData: batch[1]})
    print "step %d, training accuracy %g"%(i,train_accuracy)
  sess.run(train_step, feed_dict={imagesBatch: batch[0], outputData: batch[1]})

test_accuracy = sess.run(accuracy, feed_dict={imagesBatch: mnist.test.images, outputData: mnist.test.labels})
print(test_accuracy)

sess.close()

# tensorflow will not stop by itself in Pycharm IDE
exit(0)


