# -*- coding: utf-8 -*-
"""
Convolution Recurrent Neural Network

@author: zhaoxm
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
policy = 'RNN' # RNN, MLP or CNN
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 512 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# Define 1D conv&pooling op
def conv1d(x, W, b, stride = 1):
    x = tf.nn.conv1d(x, W, stride, padding='SAME')
    x = tf.add(x, b)
    return x

def maxpool1d(x, stride = 2):
    x = tf.expand_dims(x, axis=1)
    x = tf.nn.max_pool(x, [1, 1, stride, 1],
                        [1, 1, stride, 1], padding='SAME')
    x = tf.squeeze(x, axis=1)
    return x
    
# Define weights
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 1, 32], stddev=0.01)),
    'wc2': tf.Variable(tf.random_normal([5, 32, 64], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
    # CNN params
    'wd1': tf.Variable(tf.random_normal([28*7*64, n_hidden/4], stddev=0.01)),
    'wd2': tf.Variable(tf.random_normal([n_hidden/4, n_classes])),
    # MLP params
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=0.01)),
    'wd3': tf.Variable(tf.random_normal([8*1*64, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32], stddev=0.01)),
    'bc2': tf.Variable(tf.random_normal([64], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes])),
    # CNN params
    'bc3': tf.Variable(tf.random_normal([64], stddev=0.01)),
    # MLP params
    'bd1': tf.Variable(tf.random_normal([n_hidden/4], stddev=0.01)), 
}

def convnet1d(x, W, b):
    with tf.variable_scope("convnet"):
        x_reshape = tf.reshape(x, shape=[-1, 28, 1])
    
        conv1 = conv1d(x_reshape, weights['wc1'], biases['bc1'])
        conv1 = maxpool1d(conv1, 2)
        
        conv2 = conv1d(conv1, weights['wc2'], biases['bc2'])
        conv2 = maxpool1d(conv2, 2)
    return conv2
    
# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Get convolution embedding feature
feature_list = []
for i in xrange(n_steps):
    x_step = x[:,i,:]
    feature_list.append(convnet1d(x_step, weights, biases))
feature = tf.stack(feature_list, axis=1)

def RNN(feature, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, feature_input_size)
    # Required shape: 'n_steps' tensors list of shape (batch_size, feature_input_size)
    
    # Permuting batch_size and n_steps
    feature_input = tf.transpose(feature, [1, 0, 2, 3])
    # Reshaping to (n_steps*batch_size, feature_input_size)
    feature_input = tf.reshape(feature_input, 
                            [-1, np.product(feature_input.shape.as_list()[2:])])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, feature_input_size)
    feature_input = tf.split(feature_input, n_steps, 0)
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, activation=tf.tanh)
    
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, feature_input, dtype=tf.float32)
    
    # Linear activation, using rnn inner loop last output
    pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return pred

if policy == 'RNN':
    logits = RNN(feature, weights, biases)
elif policy == 'MLP':
    feature = tf.reshape(feature, [-1, np.product(feature.shape.as_list()[1:])])
    fc1 = tf.nn.relu(tf.add(tf.matmul(feature, weights['wd1']), biases['bd1']))
    logits = tf.add(tf.matmul(fc1, weights['wd2']), biases['out'])
elif policy == 'CNN':
    conv_feature = tf.nn.conv2d(feature, weights['wc3'], [1,1,1,1], 'VALID')
    conv_feature = tf.add(conv_feature, biases['bc3'])
    conv_feature = tf.nn.max_pool(conv_feature, [1,3,3,1], [1,3,3,1], 'SAME')   
    conv_feature = tf.reshape(conv_feature, 
                             [-1, np.product(conv_feature.shape.as_list()[1:])])                   
    logits = tf.add(tf.matmul(conv_feature, weights['wd3']), biases['out'])
else:
    print 'Unkonw policy! Candidates are RNN, MLP and CNN.'    

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
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
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")

# Calculate accuracy for 128 mnist test images
test_data = mnist.test.images[:].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:]
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    
#for v in tf.global_variables():
#    print(v)