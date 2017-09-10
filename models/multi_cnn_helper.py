
"""
Created on Tue Mar 22 10:43:29 2016
@author: Rob Romijnders
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import pdb

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable(name, initializer=initial)
  #return tf.Variable(initial, name = name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, pool_width):
  return tf.nn.max_pool(x, ksize=[1, pool_width, 1, 1],
                        strides=[1, 2, 1, 1], padding='SAME')


num_filt_1 = 16     #Number of filters in first conv layer
num_filt_2 = 8   #Number of filters in second conv layer
num_filt_3 = 4      #Number of filters in thirs conv layer
num_fc_1 = 40       #Number of neurons in fully connected layer
num_fc_2 = 20
max_iterations = 20000
plot_row = 5        #How many rows do you want to plot in the visualization
learning_rate = 2e-5
input_norm = False   # Do you want z-score input normalization?

def build_conv_net(x, bn_train, dropout, ts_length, num_classes, pool_width, layer_size_1=40, layer_size_2=20, reuse=False):
  initializer = tf.contrib.layers.xavier_initializer()
  """Build the graph"""
  # ewma is the decay for which we update the moving average of the
  # mean and variance in the batch-norm layers

  num_fc_1 = layer_size_1
  num_fc_2 = layer_size_2 

  # TODO: take away this
  x_image = tf.reshape(x, [-1,64, ts_length, 1])
  keep_prob = dropout

  W_conv1 = tf.get_variable("Conv_Layer_1", shape=[5,1, 1, num_filt_1],initializer=initializer)
  b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
  a_conv1 = conv2d(x_image, W_conv1) + b_conv1
  a_maxp1 = max_pool_2x2(a_conv1, pool_width)
  
  h_conv1 = tf.nn.relu(a_maxp1)

  """
  
  W_conv2 = tf.get_variable("Conv_Layer_2", shape=[4,1, num_filt_1, num_filt_2],initializer=initializer)
  b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
  a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
  a_maxp2 = max_pool_2x2(a_conv2, pool_width)


  h_conv2 = tf.nn.relu(a_maxp2)
  """
  W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[64*256*num_filt_1, num_fc_1],initializer=initializer)
  b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
  h_conv2_flat = tf.reshape(h_conv1, [-1, num_filt_1*64*256])

  h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  W_fc2 = tf.get_variable("W_fc2", shape=[num_fc_1, num_fc_2],initializer=initializer)
  b_fc2 = tf.get_variable('b_fc2', initializer=tf.constant(0.1, shape=[num_fc_2]))
  h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
  W_fc3 = tf.get_variable("W_fc3", shape=[num_fc_2, num_classes],initializer=initializer)
  b_fc3 = tf.get_variable('b_fc3', initializer=tf.constant(0.1, shape=[num_classes]))
  h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
  #filter_1_summary = tf.summary.image("Filter_1", W_conv1)
  #filter_2_summary = tf.summary.image("Filter_2", W_conv2)
  filters = [W_conv1, h_fc2]
  return h_fc3, filters
