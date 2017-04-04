# -*- coding: utf-8 -*-
"""
RNN by tensorflow basic op

@author: zhaoxm
"""

'''
Realize basic RNN by basic op and add ertra loss for MNIST classification
'''

import tensorflow as tf
#from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
To classify images using a basic RNN or basic RNN-RCN, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 300000
batch_size = 128
display_step = 100
loss_len = 1
useRCN = True

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 192 # hidden layer num of features
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

# Define 1D conv&pooling op
def conv1d(x, W, b, stride = 1, padding='SAME'):
    x = tf.nn.conv1d(x, W, stride, padding=padding)
    x = tf.add(x, b)
    return x

def maxpool1d(x, stride = 2):
    x = tf.expand_dims(x, axis=1)
    x = tf.nn.max_pool(x, [1, 1, stride, 1],
                        [1, 1, stride, 1], padding='SAME')
    x = tf.squeeze(x, axis=1)
    return x

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
def get_weights(name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable('W', [n_input + n_hidden, n_hidden])
        b = tf.get_variable('b', [n_hidden])
    return W, b
get_weights('rnn_cell', reuse=None)

def RNNCell(rnn_input, state):
    W, b = get_weights('rnn_cell', reuse=True)
    output = tf.tanh(tf.add(tf.matmul(tf.concat([rnn_input, state],1), W), b))
    return output, output
    
# Define a RNN convolution cell with basic op
n_filter1, n_filter2, n_filter3  = 16, 32, 64
def get_conv_weights(name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        Wxc1 = tf.get_variable('Wxc1', [5, 1, n_filter1])
        Wxc2 = tf.get_variable('Wxc2', [5, n_filter1, n_filter2])
        Wxc3 = tf.get_variable('Wxc3', [5, n_filter2, n_filter3])
        bxc1 = tf.get_variable('bxc1', [n_filter1])
        bxc2 = tf.get_variable('bxc2', [n_filter2])
        bxc3 = tf.get_variable('bxc3', [n_filter3])
        Wh1 = tf.get_variable('Wh1', [n_input/2*n_filter1, n_input/2*n_filter1])
        bh1 = tf.get_variable('bh1', [n_input/2*n_filter1])
        Wh2 = tf.get_variable('Wh2', [n_input/4*n_filter2, n_input/4*n_filter2])
        bh2 = tf.get_variable('bh2', [n_input/4*n_filter2])
        Wh3 = tf.get_variable('Wh3', [(n_input/4-4)*n_filter3, (n_input/4-4)*n_filter3])
        bh3 = tf.get_variable('bh3', [(n_input/4-4)*n_filter3])
    return Wxc1, Wxc2, Wxc3, bxc1, bxc2, bxc3, Wh1, bh1, Wh2, bh2, Wh3, bh3
get_conv_weights('rnn_conv_cell', reuse=None)

# Basic RNN-RCN
def RNNConvCell(rnn_input, state):
    # rnn_input: N x D
    # state[0]: N x D/2 x filter1, 
    # state[1]: N x D/4 x filter2, 
    # stage[2]: N x (D/4 - 4) x filter3,
    next_state = [0] * 3
    N = tf.shape(rnn_input)[0]
    Wxc1, Wxc2, Wxc3, bxc1, bxc2, bxc3, Wh1, bh1, Wh2, bh2, Wh3, bh3 = \
                                get_conv_weights('rnn_conv_cell', reuse=True)
    
    x = tf.expand_dims(rnn_input, axis=2) # N x D x 1
    conv1 = conv1d(x, Wxc1, bxc1)
    conv1 = maxpool1d(conv1, 2) #  N x D/2 x filter1
    
    conv1_r = tf.add(tf.matmul(state[0], Wh1), bh1)
    conv1_r = tf.reshape(conv1_r, [-1, n_input/2, n_filter1])
    conv1 = tf.add(conv1, conv1_r)
    next_state[0] = tf.reshape(tf.tanh(conv1), [N, -1])
    
    conv2 = conv1d(conv1, Wxc2, bxc2)
    conv2 = maxpool1d(conv2, 2)
    
    conv2_r = tf.add(tf.matmul(state[1], Wh2), bh2)
    conv2_r = tf.reshape(conv2_r, [-1, n_input/4, n_filter2])
    conv2 = tf.add(conv2, conv2_r)
    next_state[1] = tf.reshape(tf.tanh(conv2), [N, -1])
    
    # conv3: N x (D/4 - 4) x 64 x 1, D/4-4 == n_hidden
    conv3 = conv1d(conv2, Wxc3, bxc3, padding='VALID')
    conv3_r = tf.add(tf.matmul(state[2], Wh3), bh3)
    conv3_r = tf.reshape(conv3_r, [-1, n_input/4-4, n_filter3])
    conv3 = tf.add(conv3, conv3_r)
    next_state[2] = tf.reshape(tf.tanh(conv3), [N, -1])
    output = next_state[2]
    return output, next_state

# using tf.shape to realize batch_size lazy evaluation
if useRCN:
    state = [0] * 3
    state[0] = tf.zeros([tf.shape(x)[0], (n_input/2*n_filter1)])
    state[1] = tf.zeros([tf.shape(x)[0], (n_input/4*n_filter2)])
    state[2] = tf.zeros([tf.shape(x)[0], (n_input/4-4)*n_filter3])
else:
    state = tf.zeros([tf.shape(x)[0], n_hidden])
    
outputs = []
for i in xrange(n_steps):
    x_i = x[:,i,:]
    if useRCN:
        output, state = RNNConvCell(x_i, state)
    else:
        output, state = RNNCell(x_i, state)
    outputs.append(output)
    
    
# Linear activation, using rnn inner loop last output
preds = []
cost = []
for out_idx in xrange(n_steps - 1, n_steps - loss_len - 1, -1):
    pred = tf.matmul(outputs[out_idx], weights['out']) + biases['out']
    preds.append(pred)
    # Define loss and optimizer
    cost.append(tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)))
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
