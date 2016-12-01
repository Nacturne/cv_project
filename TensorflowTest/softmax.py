import tensorflow as tf
import input_data





x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#W = tf.Variable(initial_value=tf.truncated_normal(shape=[28*28, 10], stddev=0.1))
#b = tf.Variable(initial_value=tf.constant(value=0.1, shape=[10]))

yPredicted = tf.nn.softmax(tf.matmul(x, W) + b)


yTrue = tf.placeholder("float", [None,10])
#yTrue = tf.placeholder(dtype=tf.float32, shape=[None, 10])

costFunction = -tf.reduce_sum(yTrue*tf.log(yPredicted))
#costFunction = tf.reduce_sum(tf.square(yTrue - yPredicted))

trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(costFunction)



correctness = tf.equal(tf.argmax(yTrue,1), tf.argmax(yPredicted,1))
accuracy = tf.reduce_mean(tf.cast(correctness,dtype=tf.float32))


sess = tf.Session()
sess.run(tf.initialize_all_variables())

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

for i in range(5000):
    batch = mnist.train.next_batch(50)
    feedData = {x:batch[0], yTrue:batch[1]}
    sess.run(trainStep, feed_dict=feedData)
    if i%100 == 0:
        batch_accuracy = sess.run(accuracy, feed_dict=feedData)
        print('the accuracy for batch {0}: {1:.3f}'.format(i, batch_accuracy))

feedTest = {x: mnist.test.images, yTrue: mnist.test.labels}
test_accuracy = sess.run(accuracy, feed_dict=feedTest)
print('the accuracy for test-------------------------: {0:.3f}'.format(test_accuracy))

sess.close()
exit(0)