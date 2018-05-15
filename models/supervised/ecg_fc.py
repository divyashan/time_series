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

num_fc_1 = 5       #Number of neurons in fully connected layer
num_fc_0 = 2

def build_fc_model( img_shape):
	initializer = glorot_normal()
	x0 = Input( img_shape, name='Input')

	raw_x = x0
	flattened_raw_x = Flatten()(raw_x)
	fc0 = Dense( num_fc_0, activation='relu', name='fully_informed_nodes', kernel_initializer=initializer,
				kernel_regularizer=l2(.001) )(flattened_raw_x)

	fc1 = Dense( num_fc_1, activation='relu', name='dense_encoding', kernel_initializer=initializer,
				 kernel_regularizer=l2(.001) )(fc0)
	y = Dense(1, name='softmax', activation='sigmoid')(fc1)
	embedding_model = Model(inputs = x0, outputs = fc1)
	model = Model( inputs = x0, outputs = y )
	return model, embedding_model
