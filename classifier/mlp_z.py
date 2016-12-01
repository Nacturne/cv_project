
from __future__ import print_function


import tensorflow as tf
import numpy as np
#from tensorflow import config
#import matplotlib.pyplot as plt

def numpy_floatX(data):
    return np.asarray(data)



inputdata = np.genfromtxt('BizFeature.csv',delimiter=',',dtype=None)
batch_x = numpy_floatX(inputdata)
output = np.genfromtxt('biz_label.csv',delimiter=',',dtype=None)
batch_y = numpy_floatX(output)

# Parameters
learning_rate = 0.0005
training_epochs = 500
batch_size = 521
display_step = 1
logs_path = ''

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input =  1024# MNIST data input (img shape: 32*32)
n_classes = 9 # total classes (0-8 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input],name='InputData')
y = tf.placeholder("float", [None, n_classes],name='LabelData')


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="weights_layer1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name="weights_layer2"),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]),name="weights_layer3")
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="biases_layer1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="biases_layer2"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="biases_layer3")
}


# Construct model
with tf.name_scope('Model'):
    pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))

with tf.name_scope('SGD'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.arg_max(pred, 1), tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(acc,tf.float32))


# Initializing the variables
init = tf.initialize_all_variables()

tf.scalar_summary("loss", cost)
tf.scalar_summary("accuracy",acc)
merged_summary_op = tf.merge_all_summaries()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(521/batch_size)
        # Loop over all batches
        for i in range(total_batch):
                     
	    # Run optimization op (backprop) and cost op (to get loss value)
            #for (x, y) in zip(batch_x, batch_y):
            _, c,summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_x,y: batch_y})
            
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")


    #plt.plot(train_X, train_Y, 'ro', label='Original data')
    #plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    #plt.legend()
    #plt.show()
    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
