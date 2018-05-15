from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Add
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import math
import numpy as np
import sys
import pdb

sys.path.insert(0, '../')
from time_series.parse_dataset.readECG import loadECG

POOL_PCTG = .05
POOL_SIZE = 10
STRIDE_WIDTH = 10
PRINT = True
FILTER_SIZE = 10
PADDED_LENGTH = 260

"""Hyperparameters"""
num_filt_1 = 2    #Number of filters in first conv layer
num_fc_1 = 2       #Number of neurons in fully connected layer
num_fc_0 = 2
max_iterations = 4000
batch_size = 128
dropout = 1      #Dropout rate in the fully connected layer
learning_rate = 2e-4

def siamese_tower( img_shape, name_prefix ):
	x0 = Input( img_shape, name='Input')

	n_channels = [ num_filt_1 ]
	x = x0
	for i in range(len(n_channels)):
		x = Conv1D( n_channels[i], kernel_size=FILTER_SIZE, strides=1, activation='relu', padding='same')(x)
		x = MaxPooling1D(POOL_SIZE)(x) 

	x = Flatten()(x)
	y = Dense( num_fc_1, name='dense_encoding' )(x)

	model = Model( inputs = x0, outputs = y )
	return model

def siamese_model( img_shape, tower_model, n_inputs ):
	inputs = [Input( img_shape )for i in range(n_inputs)]

	tower_model.summary()
	outputs = [tower_model(i) for i in inputs]

	added_rep = Add()(outputs)

	output = Dense(1, activation='sigmoid')(added_rep)
	model = Model(inputs, output, name='siamese')
	return model

tower = siamese_tower((256, 1), 'tower')
jj = siamese_model((256, 1), tower, 1000 )
jj.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, y_train, X_test, y_test = loadECG(mode="set")
X_train = np.expand_dims(X_train, 3)
X_test = np.expand_dims(X_test, 3)

pdb.set_trace()
for i in range(400):
	jj.fit(x=[x for x in X_train], y=[[1],[1],[0],[0]], epochs=10)
	pdb.set_trace()
