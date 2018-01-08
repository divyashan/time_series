
'''This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
http://arxiv.org/pdf/1504.00941v2.pdf
Optimizer is replaced with RMSprop which yields more stable and steady
improvement.
Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Lambda, Conv2D, MaxPooling2D, Flatten
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop, Adam
import keras.backend as K


import sys
sys.path.insert(0, '../../')
from time_series.src.clean_datasets import cv_splits_for_dataset
from time_series.parse_dataset.readUcr import UCRDataset
from utils import evaluate_test_embedding, standardize_ts_lengths_1, UCR_DATASETS, MV_DATASETS, create_pairs, create_pair_idxs
import numpy as np
import pdb




batch_size = 256
num_classes = 10
epochs = 200
hidden_units = 100

clip_norm = 1.0
SAME_LABEL = 1
NEG_LABEL = 0

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Evaluate IRNN...')
"""
def get_tower_model(img_shape):
  model = Sequential()
  model.add(SimpleRNN(hidden_units,
                      kernel_initializer=initializers.RandomUniform(minval=-.1,maxval=.1),
                      recurrent_initializer=initializers.Identity(gain=1.0),
                      activation='relu',
                      batch_input_shape=(None, img_shape[0], img_shape[1])))
  model.add(Dense(num_classes))
  return model

def get_tower_cnn_model(img_shape):
  ks = 5
  x0 = Input( img_shape, name='Input')
  
  n_channels = [ 16 ]
  x = x0
  for i in range(len(n_channels)):
    x = Conv2D( n_channels[i], kernel_size=ks, strides=(2,2), activation='relu', padding='same')(x)
    x = MaxPooling2D( (5,1))(x) 

  x = Flatten()(x)
  y = Dense( 20, name='dense_encoding' )(x)

  model = Model( inputs = x0, outputs = y )
  return model

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# from keras siamese tutorial
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# from keras siamese tutorial
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

def siamese_model(img_shape, tower_model):
  input_A = Input(img_shape)
  input_B = Input(img_shape)

  tower_model.summary()
  x_A = tower_model(input_A)
  x_B = tower_model(input_B)

  distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([x_A, x_B])

  model = Model([input_A, input_B], distance, name='siamese')
  return model


def preprocess_x_data(X, window):
  # X is of the form n x d x s where there are n examples, d features, and s measurements of each feature
  # Returns new_X, a n x window*d x s array
  new_X = np.zeros((X.shape[0], X.shape[1],X.shape[2]*2))

  n_examples = X.shape[0]
  n_measurements = X.shape[1]
  n_features = X.shape[2]
  for i in range(n_examples):
    example = X[i]
    for j in range(n_measurements):
      next = np.ndarray.flatten(np.array([example[j,:],example[min(j+1,X.shape[2]-1),:]]))
      new_X[i,j,:] = next

  return new_X



if __name__ == '__main__':
  """
  X = np.array([[[1,2],[1,3]],
       [[1,4],[2,1]],
       [[1,6],[-1,10]],
       [[6,3],[11,2]],
       [[4,-1],[2,3]]])

  input1 = np.array([[[1,2]],[[1,4]],[[1,6]],[[6,3]],[[4,-1]]])
  input2 = np.array([[[1,3]],[[2,1]],[[-1,10]],[[11,2]],[[2,3]]])

  new_X = preprocess_x_data(X, 2)

  Y = np.array([1,0,1,1,0])

  input1_test = np.array([[[1,2]],[[1,4]],[[2,4]],[[4,-1]],[[10,8]]])
  input2_test = np.array([[[1,2]],[[3,4]],[[3,2]],[[1,-1]],[[-3,-1]]])

  y_test = np.array([1,1,0,1,0])

  # Preprocess all time series so that each two time steps become one feature
  """


  def gen_batch(X, tr_pair_idxs, tr_y, batch_size):

    while True:
      choices = [i for i in range(tr_pair_idxs.shape[0])]
      p = np.random.choice(choices,batch_size, replace=False)
      tr_pairs = tr_pair_idxs[p]
      input1 = X[tr_pairs[:,0]]
      input2 = X[tr_pairs[:,1]]

      y_out = tr_y[p]


      yield [input1, input2], y_out

  
  dataset = sys.argv[1]
  dataset_list = cv_splits_for_dataset(dataset)
  n_fold = 0
  X_train = dataset_list[n_fold].X_train
  y_train = dataset_list[n_fold].y_train
  X_test = dataset_list[n_fold].X_test
  y_test = dataset_list[n_fold].y_test

  n = max([np.max([v.shape[0] for v in X_train]), np.max([v.shape[0] for v in X_test])])
  X_train = standardize_ts_lengths_1(X_train, n)
  X_test = standardize_ts_lengths_1(X_test, n)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  X_train = np.expand_dims(X_train, 3)
  X_test = np.expand_dims(X_test, 3)

  N = X_train.shape[0]
  Ntest = X_test.shape[0]
  D = X_train.shape[1]
  D_ts = X_train.shape[2]
  img_shape = X_train.shape[1:]
  learning_rate = 1e-6
  labels = np.unique(y_train)
  digit_indices = [np.where(y_train == i)[0] for i in labels]
  tr_pair_idxs, tr_y = create_pair_idxs(X_train, digit_indices, labels)

  adam = Adam(lr=0.000001)

  print(img_shape)
  t_model = get_tower_cnn_model(img_shape)
  s_model = siamese_model(img_shape, t_model)
  s_model.compile( optimizer=adam, loss=contrastive_loss)
  #s_model.fit([tr_pairs[:,0,:,:], tr_pairs[:,1,:,:]], tr_y, epochs=100, batch_size=50, validation_split=.05)
  n_batches_per_epoch = tr_pair_idxs.shape[0]/batch_size/200
  s_model.fit_generator(gen_batch(X_train, tr_pair_idxs, tr_y, batch_size),steps_per_epoch=n_batches_per_epoch, nb_epoch=1)

  train_embedding = t_model.predict(X_train)
  test_embedding = t_model.predict(X_test)
  print(evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test))

