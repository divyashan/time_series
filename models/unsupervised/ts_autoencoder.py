# convolutional autoencoder in keras

import os
import pdb
import sys
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import kmedoids


#os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Input, Dense, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, UpSampling2D, Flatten, Reshape, UpSampling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist

sys.path.insert(0, '../')
from time_series.src.clean_datasets import cv_splits_for_dataset
from time_series.parse_dataset.readUcr import UCRDataset
from utils import evaluate_test_embedding, standardize_ts_lengths_1, UCR_DATASETS, MV_DATASETS, create_pairs, create_pair_idxs, get_tp_fp_tn_fn, normalize_rows, plot_filters, rebase_labels, eval_clustering
from utils import UCR_DATASETS, MV_DATASETS
from utils import standardize_ts_lengths
from keras.optimizers import RMSprop, Adam
from keras.callbacks import CSVLogger, Callback, ModelCheckpoint
from keras.initializers import Ones
from keras.constraints import unit_norm
import pandas as pd
import matplotlib.pyplot as plt
from time_series.tsne_python.tsne import debug_plots, debug_plot
from sklearn.metrics.cluster import normalized_mutual_info_score

from time_series.parse_dataset.readUcr import UCRDataset
from time_series.parse_dataset.readEEG import loadEEG
from time_series.src.clean_datasets import cv_splits_for_dataset


class clustering(Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
          debug_plots()


def dae_stackedconv_model( img_shape, kernel_size, embedding_size, conv_chans, pool_size):

  conv_chans = conv_chans
  n_convs = 1
  
  conv_channels = [conv_chans,img_shape[-1]]
  x0 = Input( img_shape, name='dae_input' )
  x = x0

  # TODO: If the input isn't an exact multiple of the pool size, the
  # upsampling doesn't result in the right sizes
  stride_size = STRIDE_SIZE
  new_dim_1 = int(np.ceil(float(img_shape[0])/stride_size))

  new_dim_2 = img_shape[1]
  for i in range(n_convs):
    w1 = Conv1D( conv_channels[i], kernel_size=kernel_size,strides=1, padding='same', name='dae_conv2D_{}'.format(i*2+1) )(x)
    #encoded = GlobalMaxPooling1D()(w1)
    x = MaxPooling1D(pool_size, stride=stride_size, padding='same')(w1)


  x = Flatten()(x)


  encoded = Dense(embedding_size)(x)
  x = Dense(new_dim_1)(encoded)

  x = Reshape( (-1, 1) )(x)
  x = UpSampling1D(size=stride_size)(x)

  x = Conv1D( conv_chans, kernel_size=kernel_size,padding='same', name='dae_deconv2D_{}'.format(0) )(x)
  x_out = Conv1D(conv_channels[-1], kernel_size=kernel_size, padding='same', name='final_deconv')(x)


  model = Model( input = [x0], output=[x_out], name='dae_stackedconv' )

  encoder = Model(input = [x0], output=[encoded])

  filters = Model(input=[x0], output=[w1])
  return model, encoder, filters

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=1)


def eval_clustering(kmeans, labels):
  kmean_sort_ind = np.argsort(labels)
  kmeans = kmeans[kmean_sort_ind]
  labels = labels[kmean_sort_ind]

  unique_labels = np.unique(labels)
  labels = rebase_labels(labels, unique_labels)
  kmeans -= np.min(kmeans)

  co_matrix = np.zeros((len(np.unique(labels)), len(np.unique(kmeans)) ))
  for i,x in enumerate(kmeans):
    co_matrix[int(labels[i])][int(x)] += 1
  tp, fp, tn, fn = get_tp_fp_tn_fn(co_matrix)

  rand_ind = float(tp + tn) / (tp + fp + fn + tn)
  nmi = normalized_mutual_info_score(kmeans, labels)
  print 'NMI Score: ', nmi
  print 'Rand Index: ', rand_ind
  return rand_ind, nmi






"""
dataset = sys.argv[1]
dataset_list = cv_splits_for_dataset(dataset)
n_fold = 0
X_train = dataset_list[n_fold].X_train
y_train = dataset_list[n_fold].y_train
X_test = dataset_list[n_fold].X_test
y_test = dataset_list[n_fold].y_test

if dataset == 'trajectories':
  X_train = [g.T for g in X_train]
  X_test = [g.T for g in X_test]


y_train = np.array(y_train)
y_test = np.array(y_test)
"""


embedding_size_opts = [2, 3, 4, 5, 24, 32, 48]
kernel_size_opts = [5,8,10, 32, 48]
n_filters_opts = [1, 2, 5]
pool_size_opts = [1, 5, 10, 20]


STRIDE_SIZE = 4
def run_dae_stackedconv_model(dataset, embedding_size_pctg, kernel_size_pctg, n_filters, pool_size_pctg):
  if dataset in UCR_DATASETS:
    UCR_DATA_DIR = os.path.expanduser('~/Documents/MEng/time_series/ucr_data/')
    ucr_dataset = UCRDataset(UCR_DATA_DIR + dataset)
    X_train = ucr_dataset.Xtrain
    y_train = ucr_dataset.Ytrain


    X_val = ucr_dataset.Xtest[:2]
    y_val = ucr_dataset.Ytest[:2]
    X_test = ucr_dataset.Xtest[2:]
    y_test = ucr_dataset.Ytest[2:]
    N = X_train.shape[0]
    Ntest = X_test.shape[0]
    D = 1 # Number of varialbes represented in time series
    D_ts = X_train.shape[1]
    X_train = np.expand_dims(X_train, 2)
    X_test = np.expand_dims(X_test, 2)

    n = max([np.max([v.shape[0] for v in X_train]), np.max([v.shape[0] for v in X_test])])
    stride_size = STRIDE_SIZE
    if n % stride_size != 0:
      n += (stride_size - n%stride_size)


    X_train = standardize_ts_lengths_1(X_train, n)
    X_test = standardize_ts_lengths_1(X_test, n)

  else:
    dataset_list = cv_splits_for_dataset(dataset)
    X_train = dataset_list[0].X_train
    y_train = dataset_list[0].y_train
    X_test = dataset_list[0].X_test
    y_test = dataset_list[0].y_test

  
    

    
    n = max([np.max([v.shape[0] for v in X_train]), np.max([v.shape[0] for v in X_test])])
    if n % STRIDE_SIZE != 0:
      n += (STRIDE_SIZE - n%STRIDE_SIZE)

    X_train = standardize_ts_lengths_1(X_train, n)
    X_test = standardize_ts_lengths_1(X_test, n)
    N = X_train.shape[0]
    Ntest = X_test.shape[0]
    D = X_train.shape[1]
    D_ts = X_train.shape[2]

  #X_train = normalize_rows(X_train) - np.mean(normalize_rows(X_train), axis=0)
  #X_test = normalize_rows(X_test) - np.mean(normalize_rows(X_test), axis=0)

  all_X = np.concatenate((X_train, X_test))
  all_y = np.concatenate((y_train, y_test))

  n_classes = len(np.unique(y_train))
  #X_train = np.expand_dims(X_train, 3)
  #X_test = np.expand_dims(X_test, 3)

  N = X_train.shape[0]
  Ntest = X_test.shape[0]
  D = X_train.shape[1]
  img_shape = X_train.shape[1:]
  b_size = 10
  kernel_size = int(kernel_size_pctg*D)
  pool_size = max(int(pool_size_pctg*D), 2 )
  #embedding_size = len(np.unique(all_y))
  embedding_size = max(embedding_size_pctg, 2)
  #embedding_size = int(embedding_size_pctg*D)
  #embedding_size = 4
  n_filters = len(np.unique(all_y))
  #n_filters = 20
  print 'Pool Size: ', pool_size
  print 'Kernel Size: ', kernel_size 
  print 'Embedding Size: ', embedding_size

  model, encoder, filters = dae_stackedconv_model(img_shape, kernel_size, embedding_size, n_filters, pool_size)
  model.summary()
  adam = Adam(lr=0.0001)
  filepath="weights-improvement-{epoch:02d}.hdf5"
  checkpointer = ModelCheckpoint(filepath,verbose=1, save_weights_only=True, period=700)
  model.compile(optimizer=adam, loss='mean_absolute_error')
  train_start = time.clock()
  model.fit(all_X, all_X, epochs=N_EPOCHS, batch_size=b_size, verbose=VERBOSE_VAL, callbacks=[checkpointer])
  train_finish = time.clock()

  weights = filters.layers[-1].get_weights()[0]
  #plot_filters(n_filters, weights)
  test_embedding = encoder.predict(X_test)
  train_embedding = encoder.predict(X_train)

  inf_start = time.clock()
  all_embedding = encoder.predict(all_X)
  inf_finish = time.clock()

  test_reconstruct = model.predict(X_test)
  train_reconstruct = model.predict(X_train)

  test_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(test_embedding).labels_
  train_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(train_embedding).labels_
  all_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(all_embedding).labels_

  #debug_plots(train_embedding, y_train, test_embedding, y_test, img_name='train_vs_test.png')
  #debug_plot(all_embedding, all_y)
  all_X =  np.reshape(all_X, (all_X.shape[0], all_X.shape[1]*all_X.shape[2]))
  orig_X_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(np.squeeze(all_X)).labels_

  print evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
  print 'All X KMeans: '
  all_rand_ind, all_nmi = eval_clustering(all_kmeans, all_y)
  print 'Test KMeans: '
  test_rand_ind = eval_clustering(test_kmeans, y_test)
  print 'Train KMeans: '
  train_rand_ind, train_nmi  = eval_clustering(train_kmeans, y_train)

  print '\n\nOriginal X KMeans: '
  orig_X_ri, orig_X_nmi = eval_clustering(orig_X_kmeans, all_y)

  D = pairwise_distances(all_embedding, metric='euclidean')
  #M, C = kmedoids.kMedoids(D, embedding_size)
  #new_labels = np.zeros(all_y.shape)
  #for i in range(len(C)):
  #  elems = C[i]
  #  for elem in elems:
  #    new_labels[elem] = i



 # print '\nK Medoids: '
  #eval_clustering(new_labels, all_y)
  return all_rand_ind, train_finish - train_start, inf_finish - inf_start

  

def test_model(dataset, pool_size_pctg=.2, embedding_size_pctg=.1, mode="vanilla"):

  kernel_size_pctg = .1
  n_filters = 1
  
  results_dicts = []
  if mode == 'test_embedding_size':
    for embedding_opt in embedding_size_opts:
      result = dict()
      result['embedding_size'] = embedding_size
      result['kernel_size'] = kernel_size
      result['n_filters'] = n_filters
      result['pool_size'] = pool_size

      a, b, c = run_dae_stackedconv_model(dataset, embedding_opt, kernel_size, n_filters, pool_size)
      result['test_rand_ind'] = a
      result['orig_test_rand_ind'] = b
      result['train_rand_ind'] = c
      results_dicts.append(result)

  elif mode == 'test_kernel_size':
    for kernel_opt in kernel_size_opts:
      result = dict()
      result['embedding_size'] = embedding_size
      result['kernel_size'] = kernel_size
      result['n_filters'] = n_filters
      result['pool_size'] = pool_size

      a, b, c = run_dae_stackedconv_model(dataset, embedding_size, kernel_opt, n_filters, pool_size)
      result['test_rand_ind'] = a
      result['orig_test_rand_ind'] = b
      result['train_rand_ind'] = c
      results_dicts.append(result)

  elif mode == 'test_filter_size':
    for n_filters in n_filters_opts:
      result = dict()
      result['embedding_size'] = embedding_size
      result['kernel_size'] = kernel_size
      result['n_filters'] = n_filters
      result['pool_size'] = pool_size

      a, b, c = run_dae_stackedconv_model(dataset, embedding_size, kernel_size, n_filters, pool_size)
      result['test_rand_ind'] = a
      result['orig_test_rand_ind'] = b
      result['train_rand_ind'] = c
      results_dicts.append(result)

  elif mode == 'test_pool_size':
    for pool_size in pool_size_opts:
      result = dict()
      result['embedding_size'] = embedding_size
      result['kernel_size'] = kernel_size
      result['n_filters'] = n_filters
      result['pool_size'] = pool_size

      a, b, c = run_dae_stackedconv_model(dataset, embedding_size, kernel_size, n_filters, pool_size)
      result['test_rand_ind'] = a
      result['orig_test_rand_ind'] = b
      result['train_rand_ind'] = c
      results_dicts.append(result)

  else:
    RI, train_time, inf_time = run_dae_stackedconv_model(dataset, embedding_size_pctg, kernel_size_pctg, n_filters, pool_size_pctg)
    return RI, train_time, inf_time

"""
df = pd.DataFrame(results_dicts)
df.to_csv('layer_size_comparison.csv', index=False)
pdb.set_trace()
"""

VERBOSE_VAL = 0
N_EPOCHS = 300

if __name__ == '__main__':

  dataset = sys.argv[1]
  mode = sys.argv[2]
  pool_size_pctg = .1
  embedding_size_pctg = .1
  RI, train_time, inf_time = test_model(dataset, pool_size_pctg, embedding_size_pctg, mode)





