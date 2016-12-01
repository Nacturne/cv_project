import tensorflow as tf
import input_data

def weightsIniti(shape):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
    return tf.Variable(initial)

def biasIniti(shape):
    initial = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial)



#################################
# Training Data placeholder
#################################

x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
yTrue = tf.placeholder(dtype=tf.float32, shape=[None, 10])


#################################
# Layer 0: Input Layer
#################################
W_0 = weightsIniti([28*28, 256])
b_0 = biasIniti([256])
out_0 = tf.nn.tanh(tf.matmul(x, W_0) + b_0)


#################################
# Layer 1: Hidden Layer
#################################
W_1 = weightsIniti([256, 128])
b_1 = biasIniti([128])
out_1 = tf.nn.tanh(tf.matmul(out_0, W_1) + b_1)


#################################
# Layer 2: Hidden Layer
#################################
W_2 = weightsIniti([128, 64])
b_2 = biasIniti([64])
out_2 = tf.nn.tanh(tf.matmul(out_1, W_2) + b_2)


#################################
# Layer 3: Output Layer
#################################
W_3 = weightsIniti([64, 10])
b_3 = biasIniti([10])
out_3 = tf.nn.sigmoid(tf.matmul(out_2, W_3) + b_3)


#################################
# Define the cost function
#################################
costFunction = tf.reduce_sum(tf.square(out_3 - yTrue))
trainStep = tf.train.AdamOptimizer(1e-4).minimize(costFunction)



#################################
# Define accuracy
#################################
correctness = tf.equal(tf.argmax(out_3, 1), tf.argmax(yTrue, 1))
accuracy = tf.reduce_mean(tf.cast(correctness, dtype=tf.float32))




#################################
# Start a session
#################################
sess = tf.Session()
sess.run(tf.initialize_all_variables())



#################################
# Train the model
#################################
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
for i in range(500):
    batch = mnist.train.next_batch(100)
    feedData = {x:batch[0], yTrue:batch[1]}
    sess.run(trainStep, feed_dict=feedData)
    if i%100 == 0:
        #outMatrix = sess.run(out_3, feed_dict=feedData)
        batch_accuracy = sess.run(accuracy, feed_dict=feedData)
        print('the accuracy for batch {0}: {1:.3f}'.format(i, batch_accuracy))

feedTest = {x: mnist.test.images, yTrue: mnist.test.labels}
test_accuracy = sess.run(accuracy, feed_dict=feedTest)
print('the accuracy for test-------------------------: {0:.3f}'.format(test_accuracy))


outVisual = mnist.train.next_batch(10)
feedVisual = {x:outVisual[0], yTrue:outVisual[1]}
outValue = sess.run(out_3, feed_dict=feedVisual)
outValue[outValue<.3] = 0
outValue[outValue!=0] = 1
outValue = outValue.astype(int)
print(outValue)
sess.close()
exit(0)