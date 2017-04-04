# -*- coding: utf-8 -*-
"""
RNN by tensorflow basic op

@author: zhaoxm
"""

'''
Realize basic RNN by basic op and add ertra loss for MNIST classification
'''

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 300000
batch_size = 128
display_step = 100
loss_len = 3

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


### Define a lstm cell with tensorflow
# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, n_steps, n_input)
# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

## Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#x = tf.unstack(x, n_steps, 1)
#
#lstm_cell = rnn.BasicRNNCell(n_hidden)
##
### Get lstm cell output
#outputs, states = rnn.static_rnn(lstm_cell, x_input, dtype=tf.float32)

# Define a RNN cell with basic op
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [n_input + n_hidden, n_hidden])
    b = tf.get_variable('b', [n_hidden])

def RNNCell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [n_input + n_hidden, n_hidden])
        b = tf.get_variable('b', [n_hidden])
        output = tf.tanh(tf.add(tf.matmul(tf.concat([rnn_input, state],1), W), b))
    return output, output

# using tf.shape to realize batch_size lazy evaluation
state = tf.zeros([tf.shape(x)[0], n_hidden])
outputs = []
for i in xrange(n_steps):
    x_i = x[:,i,:]
    output, state = RNNCell(x_i, state)
    outputs.append(output)
    
    
# Linear activation, using rnn inner loop last output
preds = []
cost = []
for out_idx in xrange(n_steps - 1, n_steps - loss_len - 1, -1):
    pred = tf.matmul(outputs[out_idx], weights['out']) + biases['out']
    preds.append(pred)
    # Define loss and optimizer
    cost.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)))
cost = tf.reduce_sum(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
