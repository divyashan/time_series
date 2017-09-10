import random
import numpy as np
import time
import tensorflow as tf 
import input_data
import math
import sys

from readUcr import UCRDataset

from utils import plot_filters, normalize_rows, evaluate_test_embedding, classify_sample, next_batch, dist, create_triplets, create_triplet_idxs, next_batch_from_idx



sys.path.insert(0, '../../')
from time_series.tsne_python import tsne


import pdb


def get_hardest_triplets(triplets, a, p, n, n_samples):

    def map_triplet(x, y, z):
        return [np.linalg.norm(x - y), np.linalg.norm(x-z)]

    distances = np.array([map_triplet(a[i], p[i], n[i]) for i in range(len(a))])
    sort_idx = np.argsort(distances, axis=0)[:,1]
    sorted_triplets = triplets[sort_idx]
    return sorted_triplets[:n_samples]



def mlp(input_,input_dim,output_dim,name="mlp"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001, stddev=0.05))
        return tf.nn.relu(tf.matmul(input_,w))
        
def build_model_mlp(X_,_dropout, input_length):

    model = mlpnet(X_,_dropout, input_length)
    return model

def mlpnet(image,_dropout, ts_length):
    l1 = mlp(image,ts_length,128,name='l1')
    l1 = tf.nn.dropout(l1,_dropout)
    l2 = mlp(l1,128,128,name='l2')
    l2 = tf.nn.dropout(l2,_dropout)
    l3 = mlp(l2,128,128,name='l3')
    #"""
    l3 = tf.nn.dropout(l3,_dropout)
    l4 = mlp(l3,128,128,name='l4')
    #"""
    return l4

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

def triplet_loss(anchor, positive, negative, alpha=5):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        
        pos_dist = dist(anchor, positive)
        neg_dist = dist(anchor, negative)
        
        #"""
        e_pos_dist = tf.exp(pos_dist)
        e_neg_dist = tf.exp(neg_dist)

        d_plus = tf.divide(e_pos_dist, tf.add(e_pos_dist, e_neg_dist))
        d_neg = tf.divide(e_neg_dist, tf.add(e_pos_dist, e_neg_dist))-1
        loss = tf.reduce_mean(tf.add(tf.square(d_plus), tf.square(d_neg)))
        """
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        #"""
      
    return loss

def new_triplet_loss(anchor, positive, negative, alpha=1, l=.03):

    phi = dist(anchor, positive) - .5*(dist(anchor, negative) + dist(anchor, positive)) + alpha
    psi = dist(anchor, positive) - alpha
    loss = tf.maximum(0.0, phi) + l*tf.maximum(0.0, psi)

    return tf.reduce_mean(loss)

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


def compute_accuracy(triplets):
    predict_same=distance.eval(feed_dict={anchor:triplets[:,0],same:triplets[:,1],dropout_f:1.0})
    predict_diff=distance.eval(feed_dict={anchor:triplets[:,0],same:triplets[:,2],dropout_f:1.0})
    acc = float(len(np.where(predict_same-predict_diff < 0)[0]))/len(predict_same)
    return acc

def test_model(dataset):
    tf.reset_default_graph()
    ucr_dataset = UCRDataset("../../../ucr_data/" + dataset) 

    X_train = ucr_dataset.Xtrain
    y_train = ucr_dataset.Ytrain 
    X_val = ucr_dataset.Xtest[:2]
    y_val = np.expand_dims(ucr_dataset.Ytest[:2], 1)
    X_test = ucr_dataset.Xtest[2:]
    y_test = np.expand_dims(ucr_dataset.Ytest[2:], 1)

    labels = np.unique(y_train)
    r=4
    digit_indices = [np.where(y_train == i)[0] for i in labels]
    tr_trips = create_triplets(X_train, digit_indices, labels)
    tr_trip_idxs = create_triplet_idxs(X_train, digit_indices, labels)

    digit_indices = [np.where(y_val == i)[0] for i in labels]
    val_trips = create_triplets(X_val, digit_indices, labels)
    val_trip_idxs = create_triplet_idxs(X_val, digit_indices, labels)

    digit_indices = [np.where(y_test == i)[0] for i in labels]
    te_trips = create_triplets(X_test, digit_indices, labels)

    # Epoch interval to evaluate early stopping
    es_epochs = 50

    #p = np.random.permutation(len(tr_trips))
    #tr_trips = tr_trips[p]
    #tr_trip_idxs = tr_trip_idxs[p]

    #X_train = normalize_rows(X_train)
    #X_test = normalize_rows(X_test)

    ts_length = len(X_train[0])
    batch_size = 24
    global_step = tf.Variable(0,trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
    # create training+test positive and negative pairs
    anchor = tf.placeholder(tf.float32,shape=([None,ts_length]),name='L')
    same = tf.placeholder(tf.float32,shape=([None,ts_length]),name='R')
    different = tf.placeholder(tf.float32,shape=([None,ts_length]),name='R')
    labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
    dropout_f = tf.placeholder("float")
    with tf.variable_scope("siamese") as scope:
        model1= build_model_mlp(anchor,dropout_f, ts_length)
        scope.reuse_variables()
        model2 = build_model_mlp(same,dropout_f, ts_length)
        scope.reuse_variables()
        model3 = build_model_mlp(different,dropout_f, ts_length)


    distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1,model2),2),1,keep_dims=True))
    loss = triplet_loss(model1, model2, model3) + regularizer(model1, model2, model3)
    regularization = regularizer(model1, model2, model3)


    t_vars = tf.trainable_variables()
    d_vars  = [var for var in t_vars if 'l' in var.name]
    batch = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.00010).minimize(loss)

    f1 = open('X_output.txt', 'w')
    f2 = open('X_labels.txt', 'w')
    f1_t = open('X_output_test.txt', 'w')
    f2_t = open('X_labels_test.txt', 'w')
    best_val_acc = 0
    early_stopping = False 
    best_epoch = 0
    with tf.Session() as sess:
    #sess.run(init)
        tf.global_variables_initializer().run()
        # Training cycle
        for epoch in range(10000):
            #if early_stopping:
            #    break 
            avg_loss = 0.
            avg_r = 0.
            total_batch = int(np.ceil(tr_trips.shape[0]/float(batch_size)))
            anchor_embedding = model1.eval(feed_dict={anchor:tr_trips[:,0],dropout_f:1.0})
            same_embedding = model1.eval(feed_dict={anchor:tr_trips[:,1],dropout_f:1.0})
            different_embedding = model1.eval(feed_dict={anchor:tr_trips[:,2],dropout_f:1.0})
            #hard_trips = get_hardest_triplets(tr_trips, anchor_embedding, same_embedding, different_embedding, tr_trips.shape[0]/2)
            hard_trips = tr_trips
            start_time = time.time()
            # Loop over all batches
            for i in range(total_batch):
                s  = i * batch_size
                e = (i+1) *batch_size
                # Fit training using batch data
                ainput1, ainput2, ainput3 = next_batch(s, e, tr_trips)
                input1,input2,input3 =next_batch_from_idx(s,e,tr_trip_idxs,X_train)
                pdb.set_trace()

                _,loss_value,predict, r_loss=sess.run([optimizer,loss,distance, regularization], feed_dict={anchor:input1,same:input2,different:input3,dropout_f:1.0})
                if math.isnan(loss_value):
                    pdb.set_trace()
                avg_loss += loss_value
                avg_r += r_loss
            duration = time.time() - start_time


            if epoch % 500 == 0:
                train_embedding=model1.eval(feed_dict={anchor:X_train,dropout_f:1.0})
                test_embedding = model1.eval(feed_dict={anchor:X_test,dropout_f:1.0})
                val_embedding = model1.eval(feed_dict={anchor:X_val,dropout_f:1.0})

                anchor_embedding=model1.eval(feed_dict={anchor:tr_trips[:,0],dropout_f:1.0})
                same_embedding=model1.eval(feed_dict={anchor:tr_trips[:,1],dropout_f:1.0})
                diff_embedding=model1.eval(feed_dict={anchor:tr_trips[:,2],dropout_f:1.0})
                accuracy = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
                other_acc = compute_accuracy(hard_trips)
                te_other_acc = compute_accuracy(te_trips)
                print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
                print('Accuracy given NN approach %0.2f' %(100*accuracy))
                #accuracy = compute_accuracy(val_trips)
                #print 'Validation Accuracy: ', accuracy

            # Early stopping
            if epoch % es_epochs == 0:
                train_embedding=model1.eval(feed_dict={anchor:X_train,dropout_f:1.0})
                val_embedding = model1.eval(feed_dict={anchor:X_val,dropout_f:1.0})
                accuracy = evaluate_test_embedding(train_embedding, y_train, val_embedding, y_val)
                #accuracy = compute_accuracy(val_trips)
                if accuracy > best_val_acc:
                    best_val_acc = accuracy
                    best_epoch = epoch 
                    if best_val_acc == 1.0:
                        early_stopping = True




        predict_same=distance.eval(feed_dict={anchor:tr_trips[:,0],same:tr_trips[:,1],dropout_f:1.0})
        predict_diff=distance.eval(feed_dict={anchor:tr_trips[:,0],same:tr_trips[:,2],dropout_f:1.0})
        tr_acc = float(len(np.where(predict_same-predict_diff < 0)[0]))/len(predict_same)
        print "Training accuracy: ", tr_acc

        

        train_embedding=model1.eval(feed_dict={anchor:X_train,dropout_f:1.0})
        test_embedding = model1.eval(feed_dict={anchor:X_test,dropout_f:1.0})
        accuracy = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
        print('Accuracy given NN approach %0.2f' %(100*accuracy))
        print 'Best Epoch: ', best_epoch

        for coord, label in zip(train_embedding, y_train):
            f1.write(' '.join([str(a) for a in coord]) + "\n")
            f2.write(str(label) + "\n")

        for coord, label in zip(test_embedding, y_test):
            f1_t.write(' '.join([str(a) for a in coord]) + "\n")
            f2_t.write(str(label) + "\n")