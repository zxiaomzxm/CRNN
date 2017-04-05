#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Binary add by lstm

@author: zhaoxm
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# training dataset generation
class BinaryAddDataset(object):
    def __init__(self, binary_dim=8):
        self.int2binary = {}
        self.binary_dim = binary_dim
        
        self.largest_number = pow(2,binary_dim)
        binary = np.unpackbits(
                np.array([range(self.largest_number)],dtype=np.uint8).T,axis=1)
        for i in range(self.largest_number):
            self.int2binary[i] = binary[i]
    
    def get_batch(self, batch_size=10):
        x_batch, y_batch = \
            np.zeros([batch_size, nsteps, 2]), np.zeros([batch_size, nsteps, 1])
        for i in xrange(batch_size):
            # generate a simple addition problem (a + b = c)
            a_int = np.random.randint(self.largest_number/2) # int version
            a = self.int2binary[a_int] # binary encoding
#            a = a[::-1]
            b_int = np.random.randint(self.largest_number/2) # int version
            b = self.int2binary[b_int] # binary encoding
#            b = b[::-1]
            # true answer
            c_int = a_int + b_int
            c = self.int2binary[c_int]
#            c = c[::-1]
            
            x_batch[i] = np.concatenate([a.reshape([1, nsteps, 1]), \
                             b.reshape([1, nsteps, 1])], \
                             axis=2)
            y_batch[i] = c.reshape([1, nsteps, 1])
        return x_batch, y_batch
    
binary_dim = 8 
Data = BinaryAddDataset(binary_dim=binary_dim)
# hyper params
alpha = 1e-1 
input_dim = 2 
hidden_dim = 16 
output_dim = 1
nsteps = binary_dim
display_step = 100
max_iter = 100000

# tf part
# forward pass
x = tf.placeholder('float', [None, nsteps, input_dim]) # batchsize x nsteps x input_dim
y = tf.placeholder('float', [None, nsteps, output_dim]) # batchsize x nsteps x output_dim

W = tf.Variable(tf.random_normal([2*hidden_dim, output_dim]), name='W_out')
b = tf.Variable(tf.random_normal([output_dim]), name='b_out')

x_tmp = tf.unstack(x, axis=1)

#cell = rnn.LSTMCell(hidden_dim, activation=tf.tanh)
#outputs, state = rnn.static_rnn(cell, x_tmp, dtype=tf.float32)

# Forward direction cell
lstm_fw_cell = rnn.BasicLSTMCell(hidden_dim, forget_bias=1.0)
# Backward direction cell
lstm_bw_cell = rnn.BasicLSTMCell(hidden_dim, forget_bias=1.0)
outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_tmp,
                                              dtype=tf.float32)

outputs = tf.stack(outputs)
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = tf.reshape(outputs, [-1, 2*hidden_dim])
y_pred = tf.sigmoid(tf.matmul(outputs, W) + b)
y_pred = tf.reshape(y_pred, [-1, nsteps, output_dim])
                  
# define cost func
batch_size = 2 
cost = tf.nn.l2_loss(y - y_pred) / batch_size
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in xrange(max_iter):
        x_data, y_data = Data.get_batch(batch_size)
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        
        if i % display_step == 0:
            loss = sess.run(cost, feed_dict={x: x_data, y: y_data})
            pred = sess.run(y_pred, feed_dict={x: x_data, y: y_data})
            print 'Pred: %s' % np.round(pred.squeeze()).astype(int)
            print 'True: %s' % y_data.squeeze().astype(int)
            print 'Iteration %d, loss = %f' % (i, loss)
        
for v in tf.global_variables():
    print("%s : %s") % (v.name,v.get_shape())
