import random
import numpy as np
import time
import tensorflow as tf 
import input_data
import math
import sys

from parse_dataset.readUcr import UCRDataset
from itertools import product
import matplotlib.pyplot as plt
sys.path.insert(0, '../../')
from time_series.tsne_python import tsne

from cnn_helper import build_conv_net
import scipy

import pdb

from utils import plot_filters, normalize_rows, evaluate_test_embedding, classify_sample, next_batch, dist


POOL_PCTG = .05

def create_pairs(x, digit_indices, labels):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    n_classes = len(labels)
    n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1
    pairs = []
    tags = []
    for j, d in enumerate(labels):
        eq_pairs = list(product(digit_indices[j], digit_indices[j]))
        eq_pairs = filter(lambda x: x[0] < x[1], eq_pairs)
        for pair in eq_pairs:
            pairs.append([x[pair[0]], x[pair[1]]])
            tags.append(SAME_LABEL)
        for i in range(j+1, len(labels)):
            diff_pairs = list(product(digit_indices[j], digit_indices[i]))
            for pair in diff_pairs:
                pairs.append([x[pair[0]], x[pair[1]]])
                tags.append(NEG_LABEL)
        """
        for i in range(n):
            z1, z2 = digit_indices[j][i], digit_indices[j][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(n_classes)
            dn_i = ((j + inc) % n_classes)
            z1, z2 = digit_indices[j][i], digit_indices[dn_i][i]
            pairs += [[x[z1], x[z2]]]
            tags += [1, 0]
        """
    return np.array(pairs), np.array(tags)      


def contrastive_loss(y,d):
    tmp= (.5*y + .5) *tf.square(d)
    #tmp= tf.mul(y,tf.square(d))
    tmp2 = (1-(.5*y + .5)) *tf.square(tf.maximum((1 - d),0))
    return tf.reduce_sum(tmp +tmp2)/batch_size/2

def c_loss(y, model1, model2):
    new_y = tf.scalar_mul(-1, y)
    b = tf.matmul(tf.diag(tf.squeeze(new_y)),model2)
    c = model1 + b
    return tf.reduce_sum(tf.norm(c, axis=1))/batch_size


def compute_accuracy(prediction,labels):
    new_labels = np.array([.5*g + .5 for g in labels])
    predict = np.array([int(y > 0) for y in np.squeeze(prediction)])
    return 1-np.abs(new_labels-predict).mean()
    #return labels[prediction.ravel() < 0.5].mean()
    #return tf.reduce_mean(labels[prediction.ravel() < 0.5])


def regularizer(model1, model2, r):
    norm1 = tf.norm(model1, axis=1)
    norm2 = tf.norm(model2, axis=1)
    a1 = tf.square(tf.subtract(norm1, r))
    b1 = tf.square(norm2 - r)
    x = .5*tf.reduce_sum(tf.add(a1, b1))/batch_size
    return x


tf.reset_default_graph()

ucr_dataset = UCRDataset("../../../ucr_data/" + sys.argv[1]) 
SAME_LABEL = 1
NEG_LABEL = -1

X_train = ucr_dataset.Xtrain
y_train = np.expand_dims(ucr_dataset.Ytrain,1)
X_val = ucr_dataset.Xtest[:2]
y_val = np.expand_dims(ucr_dataset.Ytest[:2], 1)
X_test = ucr_dataset.Xtest[2:]
y_test = np.expand_dims(ucr_dataset.Ytest[2:], 1)
#X_val, X_test = np.split(X_test, 2)
#y_val, y_test = np.split(y_test, 2)

labels = np.unique(y_train)
digit_indices = [np.where(y_train == i)[0] for i in labels]
tr_pairs, tr_y = create_pairs(X_train, digit_indices, labels)
pos_ind = np.where(tr_y == SAME_LABEL)[0]
neg_ind = np.where(tr_y == NEG_LABEL)[0]

#digit_indices = [np.where(y_val == i)[0] for i in labels]
#val_pairs, val_y = create_pairs(X_val, digit_indices, labels)
digit_indices = [np.where(y_test == i)[0] for i in labels]
te_pairs, te_y = create_pairs(X_test, digit_indices, labels)

# Initializing the variables
r=1
N = tr_pairs.shape[0]
# the data, shuffled and split between train and test sets
"""
X_train = normalize_rows(X_train)
X_test = normalize_rows(X_test)
"""
ts_length = len(X_train[0])
batch_size = 128
num_pos = 30
num_neg = batch_size - num_pos
global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
# create training+test positive and negative pairs
images_L = tf.placeholder(tf.float32,shape=([None,ts_length]),name='L')
images_R = tf.placeholder(tf.float32,shape=([None,ts_length]),name='R')
labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
embedding_size = 40
dropout_f = tf.placeholder("float")
bn_train = tf.placeholder(tf.bool) 

pool_width = POOL_PCTG*ts_length
with tf.variable_scope("siamese") as scope:
    """
    model1= build_model_mlp(images_L,dropout_f, ts_length)
    scope.reuse_variables()
    model2 = build_model_mlp(images_R,dropout_f, ts_length)

    model1= conv_net(images_L, weights, biases, dropout_f, ts_length)
    scope.reuse_variables()
    model2= conv_net(images_R, weights, biases, dropout_f, ts_length)
    """

    model1, filters = build_conv_net(images_L, bn_train, dropout_f, ts_length, embedding_size, pool_width)
    scope.reuse_variables()
    model2, _ = build_conv_net(images_R, bn_train, dropout_f, ts_length, embedding_size, pool_width)



normalize_model1 = tf.nn.l2_normalize(model1,0)
normalize_model2 = tf.nn.l2_normalize(model2,0)
cos_similarity = tf.reduce_sum(tf.multiply(normalize_model1, normalize_model1),1,keep_dims=True)

distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1,model2),2),1,keep_dims=True))
#distance = 1-scipy.spatial.distance.cosine(model1, model2)
#loss = contrastive_loss(labels,distance) + regularizer(model1, model2, r)
#loss = c_loss(labels, model1, model2) + regularizer(model1, model2, r)
loss = contrastive_loss(labels,distance) + regularizer(model1, model2, r)

#ugh = c_loss(labels, model1, model2)
ugh = contrastive_loss(labels,distance)
#loss = contrastive_loss(labels,distance) + regularizer(model1, model2, r)
regularization = regularizer(model1, model2, r)
#contrastice loss
t_vars = tf.trainable_variables()
d_vars  = [var for var in t_vars if 'l' in var.name]
batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(0.00001,momentum=0.9,epsilon=1e-6).minimize(loss)
    # Launch the graph

f1 = open('X_output.txt', 'w')
f2 = open('X_labels.txt', 'w')
f1_t = open('X_output_test.txt', 'w')
f2_t = open('X_labels_test.txt', 'w')

#filter2_summary = tf.summary.image("Filter_2", filters[1])
patience_window = 5
last_vals = [0 for i in range(patience_window)]
early_stopping = False
with tf.Session() as sess:

    tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter('/tmp/logs', sess.graph_def)
    #merged = tf.summary.merge_all()


    # Training cycle
    step = 0
    perf_collect = [[],[]]
    for epoch in range(700):

        total_c_loss = 0
        for i in range(max(int(np.ceil(N/float(batch_size))),1)):

            batch_ind = np.arange(i*batch_size, min((i+1)*batch_size,N-1))
            #pos_ind = np.arange(i*num_pos, min((i+1)*num_pos,N-1))
            #neg_ind = np.arange(i*num_neg, min((i+1)*num_neg,N-1))

            #pos_ind = np.random.choice( np.arange(N), num_pos)
            #neg_ind = np.random.choice( np.arange(N), num_neg)
            #batch_ind = np.concatenate((pos_ind, neg_ind))
                #print('VAL ACCURACY %0.2f' % perf_collect[1][-1])

            input1,input2,y = tr_pairs[batch_ind,[0]], tr_pairs[batch_ind,1], tr_y[batch_ind, np.newaxis] 
            _, loss_value,predict, r_loss, c_loss=sess.run([optimizer, loss,distance, regularization, ugh], feed_dict={images_L:input1,images_R:input2 ,labels:y,dropout_f:1.0, bn_train:True})
            total_c_loss += c_loss
            if math.isnan(c_loss):
                pdb.set_trace()

        if epoch%50 == 0:
            feature1=model1.eval(feed_dict={images_L:input1,dropout_f:1.0, bn_train:False})
            feature2=model2.eval(feed_dict={images_R:input2,dropout_f:1.0, bn_train:False})

            #tr_acc = compute_accuracy(predict,y)

            print('epoch %d loss %0.5f r_loss %0.5f c_loss %0.5f ' %(epoch,loss_value, r_loss, total_c_loss))



            if epoch%50 == 0:
                if r_loss < 0.00001 and c_loss < 0.00001:
                    pdb.set_trace()

                train_embedding=model1.eval(feed_dict={images_L:X_train,dropout_f:1.0, bn_train:False})
                test_embedding = model1.eval(feed_dict={images_L:X_test,dropout_f:1.0, bn_train:False})
                val_embedding = model1.eval(feed_dict={images_L:X_val, dropout_f:1.0, bn_train:False})
                accuracy = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
                val_accuracy = evaluate_test_embedding(train_embedding, y_train, val_embedding, y_val)
                last_vals[(epoch/100) % patience_window] = val_accuracy
                if last_vals.count(last_vals[0]) == len(last_vals) and i > 900:
                    early_stopping = True
                print('Accuracy given NN approach %0.2f' %(100*accuracy))
                print('Val Accuracy given NN approach %0.2f' %(100*val_accuracy))
        """
        if early_stopping:
            print 'Stopping early'
            break
        """


        #print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
    filter1_weights = sess.run(filters[0])
    feature1=model1.eval(feed_dict={images_L:tr_pairs[:,0],dropout_f:1.0, bn_train:False})
    feature2=model2.eval(feed_dict={images_R:tr_pairs[:,1],dropout_f:1.0, bn_train:False})

    # Test model
    """
    y = np.reshape(te_y,(te_y.shape[0],1))
    feature1=model1.eval(feed_dict={images_L:te_pairs[:,0],dropout_f:1.0, bn_train:False})
    feature2=model2.eval(feed_dict={images_R:te_pairs[:,1],dropout_f:1.0, bn_train:False})
    te_acc = compute_accuracy_features(feature1, feature2,te_y)
    print('Accuracy test set %0.2f' % (100 * te_acc))
    """
    train_embedding=model1.eval(feed_dict={images_L:X_train,dropout_f:1.0, bn_train:False})
    test_embedding = model1.eval(feed_dict={images_L:X_test,dropout_f:1.0, bn_train:False})

    accuracy = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
    print('Accuracy given NN approach %0.2f' %(100*accuracy))
    """
    filter1_weights = sess.run(filters[0])
    #plot_filters(1, 8, filter1_weights)
    for coord, label in zip(train_embedding, y_train):
        f1.write(' '.join([str(a) for a in coord]) + "\n")
        f2.write(str(label) + "\n")

    for coord, label in zip(test_embedding, y_test):
        f1_t.write(' '.join([str(a) for a in coord]) + "\n")
        f2_t.write(str(label) + "\n")
    """
