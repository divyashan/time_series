'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 32 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)







# Create some wrappers for simplicity
def conv2d(x, W, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return x


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout, D):
    # Reshape input picture
    num_filt_1 = 32
    num_filt_2 = 64
    x = tf.reshape(x, shape=[-1, D, 1, 1])

    # Convolution Layer
    W_conv1 = weights['wc1']
    b_conv1 = weights['bc1']
    a_conv1 = conv2d(x, W_conv1) + b_conv1

    a_conv1 = tf.contrib.layers.batch_norm(a_conv1, is_training=bn_train,updates_collections=None)
    h_conv1 = tf.nn.relu(a_conv1)

    # Max Pooling (down-sampling)
    #conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    W_conv2 = weights['wc2']
    b_conv2 = weights['bc1']
    a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

    a_conv2 = tf.contrib.layers.batch_norm(a_conv2, is_training=bn_train,updates_collections=None)
    h_conv2 = tf.nn.relu(a_conv2)
    # Max Pooling (down-sampling)
    W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D*num_filt_2, num_fc_1],initializer=initializer)
    b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
    h_conv3_flat = tf.reshape(h_conv2, [-1, D*num_filt_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
initializer = tf.contrib.layers.xavier_initializer()

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.get_variable('wc1', shape=[5, 1, 1, 32], initializer=initializer),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.get_variable('wc2', shape=[4, 1, 32, 64], initializer=initializer),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.get_variable('wd1', shape=[10240, 1024], initializer=initializer),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.get_variable('out', [1024, n_classes], initializer=initializer)
}

biases = {
    'bc1': tf.Variable(tf.constant(.1, shape=[32])),
    'bc2': tf.Variable(tf.constant(.1, shape=[64])),
    'bd1': tf.Variable(tf.constant(.1, shape=[1024])),
    'out': tf.Variable(tf.constant(.1, shape=[nclasses]))
}
"""
# Construct model
D = 784
pred = conv_net(x, weights, biases, keep_prob, D)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
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
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
"""