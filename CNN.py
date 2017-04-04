# -*- coding: utf-8 -*-
"""
Convolution Neural Network

@author: zhaoxm
"""

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden = 512 # num of embedding features
n_classes = 10 # MNIST total classes (0-9 digits)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, b, stride = 1):
    x = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding='SAME')
    x = tf.add(x, b)
    return x

def maxpool(x, stride = 2):
    x = tf.nn.max_pool(x, [1, stride, stride, 1],
                        [1, stride, stride, 1], padding='SAME')
    return x

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': weight_variable([5, 5, 1, 32]),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': weight_variable([5, 5, 32, 64]),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': weight_variable([7*7*64, n_hidden]),
    # 1024 inputs, 10 outputs (class prediction)
    'out': weight_variable([n_hidden, n_classes])
}

biases = {
    'bc1': bias_variable([32]),
    'bc2': bias_variable([64]),
    'bd1': bias_variable([n_hidden]),
    'out': bias_variable([n_classes])
}

# Create model
def convnet(x, weights, biases):
    x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x_reshape, weights['wc1'], biases['bc1'])
    conv1 = maxpool(conv1, 2)
    
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool(conv2, 2)
    
    conv2 = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(conv2, weights['wd1']), biases['bd1']))
    logits = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return logits

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, n_classes])

logits = convnet(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

accuracy = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:],
                                      y: mnist.test.labels[:]}))

    
