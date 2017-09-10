import random
import numpy as np
import time
import tensorflow as tf
import math
import sys


from itertools import product
from multi_cnn_helper import build_conv_net
import matplotlib.pyplot as plt
from utils import plot_filters, normalize_rows, evaluate_test_embedding, classify_sample, next_batch, create_triplets, create_triplet_idxs, next_batch_from_idx

sys.path.insert(0, '../../')

# from time_series.tsne_python import tsne
from time_series.parse_dataset.readUcr import UCRDataset
from time_series.parse_dataset.readEEG import loadEEG


import pdb
batch_size = 128
embedding_size = 40



def regularizer(model1, model2, model3):
    r = 1
    norm1 = tf.norm(model1, axis=1)
    norm2 = tf.norm(model2, axis=1)
    norm3 = tf.norm(model3, axis=1)
    a1 = tf.square(tf.subtract(norm1, r))
    b1 = tf.square(norm2 - r)
    c1 = tf.square(norm3 - r)

    x = tf.reduce_sum(tf.add(a1, tf.add(b1, c1)))/(3*batch_size)
    return x


def triplet_loss(anchor, positive, negative, alpha=1):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        #"""
        e_pos_dist = tf.exp(pos_dist)
        e_neg_dist = tf.exp(neg_dist)

        d_plus = tf.divide(e_pos_dist, tf.add(e_pos_dist, e_neg_dist))
        d_neg = tf.divide(e_neg_dist, tf.add(e_pos_dist, e_neg_dist))-1
        loss = tf.reduce_mean(tf.add(tf.square(d_plus), tf.square(d_neg)))
        """
        basic_loss = tf.maximum(tf.add(tf.subtract(pos_dist,neg_dist), alpha),0)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        #"""

    return loss

def debug_loss(anchor, positive, negative, alpha=1., intra_class=.01, inter_class=1):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    pos_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1))
    neg_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1))

    #"""
    e_pos_dist = tf.exp(pos_dist)
    e_neg_dist = tf.exp(neg_dist)

    d_plus = tf.divide(e_pos_dist, tf.add(e_pos_dist, e_neg_dist))
    d_neg = tf.divide(e_neg_dist, tf.add(e_pos_dist, e_neg_dist))-1
    loss = tf.reduce_mean(tf.add(tf.square(d_plus), tf.square(d_neg)))
    """
    basic_loss = tf.maximum(tf.add(tf.subtract(pos_dist,neg_dist), alpha),0)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    """

    return e_pos_dist, e_neg_dist

def new_loss(anchor, positive, negative, alpha=1, l=.01):
    with tf.variable_scope('triplet_loss'):

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)/4.0
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)/4.0

        mu_plus = tf.reduce_mean(pos_dist)
        mu_neg = tf.reduce_mean(neg_dist)

        sigma_plus = tf.reduce_mean(tf.square(pos_dist - mu_plus))
        sigma_neg = tf.reduce_mean(tf.square(neg_dist - mu_neg))

        loss = sigma_plus + sigma_neg + l*(tf.maximum(0.0 , mu_plus - mu_neg + alpha))
    return loss

def new_new_loss(anchor, positive, negative, inter_class=-10, intra_class=.01):
    with tf.variable_scope('triplet_loss'):

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        pos_loss = tf.reduce_mean(tf.maximum(pos_dist, intra_class))
        neg_loss = tf.reduce_mean(tf.maximum(pos_dist-neg_dist, inter_class))

        loss = pos_loss + neg_loss
        #loss = -tf.reduce_mean(neg_dist)
    return loss


def compute_accuracy(triplets):
    predict_same=distance.eval(feed_dict={anchor:triplets[:,0],same:triplets[:,1],dropout_f:1.0})
    predict_diff=distance.eval(feed_dict={anchor:triplets[:,0],same:triplets[:,2],dropout_f:1.0})
    acc = float(len(np.where(predict_same-predict_diff < 0)[0]))/len(predict_same)
    return acc

def test_model(pool_pctg, layer_size_1, layer_size_2):
    tf.reset_default_graph()

    X_train, y_train, X_test, y_test = loadEEG()
    X_val = X_test[:2]
    y_val = y_test[:2]
    X_test = X_test[2:]
    y_test = y_test[2:]

    labels = np.array([0, 1, 2, 3, 4, 5])
    digit_indices = [np.where(y_train == i)[0] for i in labels]
    tr_trip_idxs = create_triplet_idxs(X_train, digit_indices, labels)
    digit_indices = [np.where(y_test == i)[0] for i in labels]
    te_trip_idxs = create_triplet_idxs(X_test, digit_indices, labels)
    print 'There are ', len(tr_trip_idxs), ' training examples!'
    #p = np.random.permutation(len(tr_trip_idxs))
    #tr_trip_idxs = tr_trip_idxs[p]


    # Initializing the variables
    # the data, shuffled and split between train and test sets
    #"""
    X_train = normalize_rows(X_train)
    X_test = normalize_rows(X_test)
    #"""
    D = X_train.shape[1]
    ts_length = X_train.shape[2]
    global_step = tf.Variable(0,trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
    pool_width = pool_pctg*ts_length

    # create training+test positive and negative pairs
    anchor = tf.placeholder(tf.float32,shape=([None, D, ts_length]),name='L')
    same = tf.placeholder(tf.float32,shape=([None,D, ts_length]),name='R')
    different = tf.placeholder(tf.float32,shape=([None,D, ts_length]),name='R')
    labels = tf.placeholder(tf.float32,shape=([None]),name='gt')

    dropout_f = tf.placeholder("float")
    bn_train = tf.placeholder(tf.bool)

    with tf.variable_scope("siamese") as scope:
        model1, filters= build_conv_net(anchor,bn_train,dropout_f, ts_length, embedding_size, pool_width, layer_size_1, layer_size_2)
        scope.reuse_variables()
        model2, _ = build_conv_net(same,bn_train,dropout_f, ts_length, embedding_size, pool_width, layer_size_1, layer_size_2)
        scope.reuse_variables()
        model3, _ = build_conv_net(different,bn_train,dropout_f, ts_length, embedding_size, pool_width, layer_size_1, layer_size_2)


    distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1,model2),2),1,keep_dims=True))
    loss = triplet_loss(model1, model2, model3) #+ regularizer(model1, model2, model3)
    #loss = new_new_loss(model1, model2, model3) + regularizer(model1, model2, model3)

    debug_val = debug_loss(model1, model2, model3)
    regularization = regularizer(model1, model2, model3)
    tr_loss = triplet_loss(model1, model2, model3)


    t_vars = tf.trainable_variables()
    d_vars  = [var for var in t_vars if 'l' in var.name]
    batch = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

    f1 = open('X_output.txt', 'w')
    f2 = open('X_labels.txt', 'w')
    f1_t = open('X_output_test.txt', 'w')
    f2_t = open('X_labels_test.txt', 'w')
    patience_window = 10
    early_stopping = False
    last_vals = [0 for i in range(patience_window)]
    skippable_batches = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Training cycle
        for epoch in range(400):
            avg_loss = 0.
            avg_r = 0.
            total_batch = int(np.ceil(tr_trip_idxs.shape[0]/float(batch_size)))
            start_time = time.time()
            # Loop over all batches
            loss_values = []
            avg_tr = 0.
            for i in range(total_batch):
                if i in skippable_batches:
                    continue
                s  = i * batch_size
                e = (i+1) *batch_size
                # Fit training using batch data
                input1,input2,input3 =next_batch_from_idx(s,e,tr_trip_idxs,X_train)
                #anchor_embedding=model1.eval(feed_dict={anchor:input1,dropout_f:1.0})
                #same_embedding=model1.eval(feed_dict={anchor:input2,dropout_f:1.0})
                #diff_embedding=model1.eval(feed_dict={anchor:input3,dropout_f:1.0})
                _,loss_value,predict, r_loss, tr_val, d=sess.run([optimizer,loss,distance, regularization, tr_loss, debug_val], feed_dict={anchor:input1,same:input2,different:input3,dropout_f:1.0})
                print loss_value
                if loss_value < .001:
                    skippable_batches.append(i)
                    if i % 30 == 0:
                        train_embedding=model1.eval(feed_dict={anchor:X_train,dropout_f:1.0})
                        test_embedding = model1.eval(feed_dict={anchor:X_test,dropout_f:1.0})
                        #val_embedding = model1.eval(feed_dict={anchor:X_val,dropout_f:1.0})
                        accuracy = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
                        print 'ACCURACY: ', accuracy, ' EPOCH: ', epoch

                #pdb.set_trace()
                if math.isnan(loss_value):
                    pdb.set_trace()
                avg_loss += loss_value
                loss_values.append(loss_value)
                avg_r += r_loss
                avg_tr += tr_val

            #print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
            print('epoch %d  time: %f loss %0.5f r_loss %0.5f tr_loss %0.5f' %(epoch,duration,avg_loss/(total_batch), avg_r/total_batch, tr_val))

            duration = time.time() - start_time
            if epoch % 10  == 0:
                tr_acc = compute_accuracy(tr_trip_idxs)
                print "Training accuracy: ", tr_acc

                print('epoch %d  time: %f loss %0.5f r_loss %0.5f tr_loss %0.5f' %(epoch,duration,avg_loss/(total_batch), avg_r/total_batch, tr_val))
                train_embedding=model1.eval(feed_dict={anchor:X_train,dropout_f:1.0})
                test_embedding = model1.eval(feed_dict={anchor:X_test,dropout_f:1.0})
                #val_embedding = model1.eval(feed_dict={anchor:X_val,dropout_f:1.0})
                accuracy = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
                #val_accuracy = evaluate_test_embedding(train_embedding, y_train, val_embedding, y_val)

                print('Accuracy given NN approach %0.2f' %(100*accuracy))
                #print('Val Accuracy given NN approach %0.2f' %(100*val_accuracy))


                last_vals[(epoch/100) % patience_window] = val_accuracy
                if last_vals.count(last_vals[0]) == len(last_vals):
                    early_stopping = True
            """
            if early_stopping:
                print 'Stopping early!'
                break
            """



        train_embedding=model1.eval(feed_dict={anchor:X_train,dropout_f:1.0})
        test_embedding = model1.eval(feed_dict={anchor:X_test,dropout_f:1.0})
        accuracy = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
        print('Accuracy given NN approach %0.2f' %(100*accuracy))

        filter1_weights = sess.run(filters[0])
        for coord, label in zip(train_embedding, y_train):
            f1.write(' '.join([str(a) for a in coord]) + "\n")
            f2.write(str(label) + "\n")

        for coord, label in zip(test_embedding, y_test):
            f1_t.write(' '.join([str(a) for a in coord]) + "\n")
            f2_t.write(str(label) + "\n")

    return accuracy
