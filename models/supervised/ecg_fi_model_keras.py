from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Add, Concatenate, concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.initializers import glorot_normal
from keras.regularizers import l2
import keras.backend as K
import sys

sys.path.insert(0, '../')
import numpy as np
import pdb
import os

POOL_PCTG = .05
STRIDE_WIDTH = 10
FILTER_SIZE = 50
PADDED_LENGTH = 260
num_filt_1 = 1     #Number of filters in first conv layer
num_fc_1 = 2       #Number of neurons in fully connected layer
num_fc_0 = 2
max_iterations = 4000
batch_size = 128
dropout = 1      #Dropout rate in the fully connected layer
learning_rate = 2e-4
POOL_SIZE = POOL_PCTG * PADDED_LENGTH
n_classes = 2

def build_fi_model( img_shape):
	initializer = glorot_normal()
	x0 = Input( img_shape, name='Input')

	raw_x = x0
	flattened_raw_x = Flatten()(raw_x)
	fc0 = Dense( num_fc_0, activation='relu', name='fully_informed_nodes', kernel_initializer=initializer,
				kernel_regularizer=l2(.001) )(flattened_raw_x)
	x = Conv1D( num_filt_1, kernel_size=FILTER_SIZE, strides=1, activation='relu', padding='same', kernel_initializer=initializer,
				kernel_regularizer=l2(.001))(raw_x)
	x = MaxPooling1D(pool_size=int(POOL_SIZE), strides=STRIDE_WIDTH, padding='same')(x) 

	x = Flatten()(x)
	first_layer = Concatenate()([x, fc0])
	fc1 = Dense( num_fc_1, activation='relu', name='dense_encoding', kernel_initializer=initializer,
				 kernel_regularizer=l2(.001) )(first_layer)
	y = Dense(1, name='softmax', activation='sigmoid')(fc1)
	embedding_model = Model(inputs = x0, outputs = fc1)
	model = Model( inputs = x0, outputs = y )
	return model, embedding_model
