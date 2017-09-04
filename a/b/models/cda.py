from readUcr import UCRDataset
import numpy as np
import time
import tensorflow as tf 
import sys 
from fastdtw import fastdtw
from scipy.spatial.distance import squareform, pdist

import pdb

def gen_pairwise_dtw(X_train):
    dtw_dists = np.zeros((len(X_train), len(X_train)))
    for i in range(len(X_train)):
    	for j in range(i+1, len(X_train)):
	        d , _ = fastdtw(X_train[i],X_train[j])
	        dtw_dists[i][j] = d
	        dtw_dists[j][i] = d
    return dtw_dists

def dtw_to_training(sample, X_train):
	return [fastdtw(sample, x)[0] for x in X_train]

def alpha(alpha_0, t):
	return alpha_0

def weight(Y_ij, lambda_y):
	return 1

ucr_dataset = UCRDataset("../../../time-series/ucr_data/" + sys.argv[1]) 

X_train = ucr_dataset.Xtrain
y_train = ucr_dataset.Ytrain 
X_test = ucr_dataset.Xtest
y_test = ucr_dataset.Ytest

dtw_dists = gen_pairwise_dtw(X_train)
alpha_0 = .3


X = X_train
Y = X_train

x_dist = dtw_dists


n_steps = 100000
n_samples = len(Y)
lambda_y = 0

for t in range(n_steps):
	D = squareform(pdist(Y))

	# Choose a random vector to stay fixed 
	i = np.random.choice(np.arange(n_samples), 1)[0]

	# Choose the vector that's furthest from having the correct
	# curvilinear distance
	#i = np.argmax(np.sum(np.abs(x_dist-D), axis=1))

	fixed_pt = Y[i]
	for j in range(n_samples):
		if i == j:
			continue
		
		adjust_pt = Y[j]
		Y_ij = D[i][j]
		scale = alpha(alpha_0, t)*weight(Y_ij, lambda_y)*(x_dist[i][j] - Y_ij)*(1.0/Y_ij)
		adjust_delta = scale*(adjust_pt - fixed_pt)
		Y[j] = adjust_pt + adjust_delta
	
	if t % 500 == 0:
		D = squareform(pdist(Y))
		print "Loss: ", np.linalg.norm(x_dist-D)
		#pdb.set_trace()


alpha_0 = .95
n_steps = 1000
correct = 0

for idx, new_sample in enumerate(X_test):
	dtw_dist_i = dtw_to_training(new_sample, X_train)
	adjust_pt = new_sample

	euclid_dist_before = [np.linalg.norm(Y[j] - adjust_pt) for j in range(n_samples)]
	for i in range(n_steps):
		for j in range(n_samples):
			fixed_pt = Y[j]
			Y_ij = np.linalg.norm(fixed_pt-adjust_pt)
			scale = alpha(alpha_0, t)*weight(Y_ij, lambda_y)*(dtw_dist_i[j] - Y_ij)*(1.0/Y_ij)
			adjust_delta = scale*(adjust_pt - fixed_pt)
			adjust_pt = adjust_pt + adjust_delta
	euclid_dist_after = [np.linalg.norm(Y[j] - adjust_pt) for j in range(n_samples)]
	loss = np.linalg.norm(np.array(dtw_dist_i)-np.array(euclid_dist_after))
	guess = y_train[np.argmin(euclid_dist_after)]

	if guess == y_test[j]:
		correct += 1
	print 'Guess: ', y_train[np.argmin(euclid_dist_after)], '\tActual: ', y_test[idx], np.argmin(euclid_dist_after), loss

print float(correct)/len(y_test)


"""
def mlp(input_,input_dim,output_dim,name="mlp"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = .1, stddev=0.05))
        return tf.nn.relu(tf.matmul(input_,w))
        

def mlpnet(image,_dropout, ts_length, ):
    l1 = mlp(image,ts_length,128,name='l1')
    l1 = tf.nn.dropout(l1,_dropout)
    #l2 = mlp(l1,128,128,name='l2')
    #l2 = tf.nn.dropout(l2,_dropout)
    #l3 = mlp(l2,128,128,name='l3')
    return l1

def build_model_mlp(X_,_dropout, input_length):

    model = mlpnet(X_,_dropout, input_length)
    return model

ucr_dataset = UCRDataset("../../../time-series/ucr_data/" + sys.argv[1]) 

X_train = ucr_dataset.Xtrain
y_train = ucr_dataset.Ytrain 
X_test = ucr_dataset.Xtest
y_test = ucr_dataset.Ytest


dtw_dists = gen_pairwise_dtw(X_train)
ts_length = len(X_train[0])
dropout_f = tf.placeholder("float")
embedding_length = ts_length
n_samples = len(X_train)


X_in = tf.placeholder(tf.float32,shape=([None,ts_length]),name='input_ts')
X_out = build_model_mlp(X_in,dropout_f, ts_length)


r = tf.reduce_sum(X_out*X_out, 1)
r = tf.reshape(r, [-1, 1])
D = r - 2*tf.matmul(X_out, tf.transpose(X_out)) + tf.transpose(r) 
loss = tf.reduce_sum(tf.pow(D - dtw_dists,2))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)


with tf.Session() as sess:

	tf.global_variables_initializer().run()

	steps = 10000
	for i in range(steps):
		_, dist_loss = sess.run([optimizer, loss], feed_dict={X_in:X_train, dropout_f:.9})
		Y = X_out.eval(feed_dict={X_in:X_train, dropout_f:.9})
		euclid_dists = squareform(pdist(Y))
		if steps % 100 == 0:
			print 'Total loss: ', np.linalg.norm(euclid_dists-dtw_dists)
			pdb.set_trace()
"""

