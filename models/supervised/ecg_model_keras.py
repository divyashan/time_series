from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Add
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.initializers import glorot_normal
from keras.regularizers import l2
import keras.backend as K
import math
import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import h5py

sys.path.insert(0, '../')
from time_series.parse_dataset.readECG import loadECG
from time_series.models.ecg_utils import get_all_adjacent_beats
from time_series.models.supervised.ecg_fi_model_keras import build_fi_model 
from time_series.models.supervised.ecg_fc import build_fc_model
from time_series.models.gpu_utils import restrict_GPU_keras

restrict_GPU_keras("0")
POOL_PCTG = .05
STRIDE_WIDTH = 1
FILTER_SIZE = 100
PADDED_LENGTH = 260

"""Hyperparameters"""
num_filt_1 = 5     #Number of filters in first conv layer
num_fc_1 = 2       #Number of neurons in fully connected layer
num_fc_0 = 2
max_iterations = 4000
batch_size = 128
dropout = 1      #Dropout rate in the fully connected layer
learning_rate = 2e-4
POOL_SIZE = POOL_PCTG * PADDED_LENGTH
n_classes = 2


def build_model( img_shape):
	initializer = glorot_normal()
	x0 = Input( img_shape, name='Input')

	x = x0
	x = Conv1D( num_filt_1, kernel_size=FILTER_SIZE, strides=1, activation='relu', padding='same', kernel_initializer=initializer,
				kernel_regularizer=l2(.001))(x)
	x = MaxPooling1D(pool_size=int(POOL_SIZE), strides=STRIDE_WIDTH, padding='same')(x) 

	x = Flatten()(x)
	fc1 = Dense( num_fc_1, name='dense_encoding', kernel_initializer=initializer,
				 kernel_regularizer=l2(.001) )(x)
	y = Dense(1, name='softmax', activation='sigmoid')(fc1)

	embedding_model = Model(inputs = x0, outputs = fc1)
	model = Model( inputs = x0, outputs = y )
	return model, embedding_model

def evaluate_patients(model, patient_ids, threshold=.5):
	scores = []
	for pid in patient_ids: 
		aa = get_all_adjacent_beats(pid, 1000)

		output = model.predict(aa)
		pred_y = [1 if v > threshold else 0 for v in output]
		pid_score = np.mean(pred_y)
		print(pid_score)
		scores.append(pid_score)
	return scores


test_normal_ids =  np.array([1071, 1072, 1073, 1075, 1076,  108, 1086, 1088, 1089,  109, 1090, 1091, 1092, 1093, 1094, 1095,   11,  110, 1104, 1105, 1106, 1107, 1109,  111, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119,  112, 1120, 1121, 1123, 1124, 1126, 1127, 1128, 1129,  113, 1130, 1131, 1133, 1134, 1135, 1138, 1139,  114, 1140, 1141, 1142, 1143, 1146, 1147, 1148, 1149,  115,  116,  117, 1171, 1176, 1179, 118, 1182, 1188,  119,   12,  120,  121,  123, 1232,  124,  125, 126,  127,  128,  129,   13,  130,  131,  132,  133, 1332,  134, 135,  136,  137,  138,  139,  140, 1417,  144, 1463,   15, 1547, 1554,   16, 1616, 1644, 1687,   17, 1723, 1725, 1727, 1731, 1733, 1746,   18, 1804,   19, 1902, 1936, 1992,   20, 2004,   21, 2152, 22, 2234, 2267, 2290,   23, 2390,   24,   25, 2507, 2555, 2556, 2592,   26,   27, 2705, 2779,   28, 2812, 2831, 2832, 2833, 2841, 2844, 2865,   29, 2965, 2966, 2987,    3,   30, 3010, 3014, 3016, 3018, 3040, 3054, 3063, 3077,   31, 3181,   32, 3212, 3217, 3224, 3239, 3249, 3255, 3266, 3268,   33,   34, 3445, 3495, 3522, 3561, 3566, 3575,   36, 3621, 3671, 3679,  368, 3680,  369,   37,  370, 3704,  371, 3717,  372, 3722, 3723, 3727,  373,  374, 3748,  375, 3751,  376,  377, 3779, 3787,  379,   38, 3801,  381,  382, 3861, 39, 3938, 3950, 3957,    4,   40, 4002, 4010, 4018, 4073, 4096, 4132, 4169, 4171, 4174, 4193,   42, 4236, 4240, 4245, 4248, 4257, 4263,   43,  437,  439, 4391,   44, 4407,  441, 4419,  442, 4442, 4450, 4468,   45, 4517, 4523, 4585,   46,   47, 4741, 4747, 4755, 4756, 4766, 4840, 4841, 4869, 4882,   49, 4917,  494,  495, 4954, 497, 4975, 4976,  498, 4980, 4982,  499, 4992,    5,   50,  500, 5004,  502,  503, 5065, 5072, 5083, 5088, 5098,   51, 5106, 5120, 5123, 5144, 5157, 5171])
test_death_ids = np.array([5616, 5687, 5908, 5964, 6160, 6311, 760, 832, 854, 961])
 


m, embedding_m = build_fc_model((256, 1))
m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
m.summary()

if len(sys.argv)  == 1:
	X_train, y_train, X_test, y_test = loadECG()
else:
	hf = h5py.File('data.h5', 'r')
	X_train = np.array(hf.get('X_train'))
	y_train = np.array(hf.get('y_train'))
	X_test = np.array(hf.get('X_test'))
	y_test = np.array(hf.get('y_test')) 

X_train = np.swapaxes(X_train, 1, 2)
X_test = np.swapaxes(X_test, 1, 2)

new_order = np.arange(len(X_train))
np.random.shuffle(new_order)
X_train = X_train[new_order]
y_train = y_train[new_order]

new_order = np.arange(len(X_test))
np.random.shuffle(new_order)
X_test = X_test[new_order]
y_test = y_test[new_order]
X_val = X_test[:30000]
y_val = y_test[:30000]

#X_test = X_test[np.where(y_test == 1)[0]]
#y_test = y_test[np.where(y_test == 1)[0]]
print("loaded data")
for i in range(400):
	m.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=10, verbose=True)
	test_embedding = embedding_m.predict(X_test)
	train_embedding = embedding_m.predict(X_train)
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train_embedding)
	_, indices = nbrs.kneighbors(test_embedding)
	pdb.set_trace()
	#difference = y_train[indices] - y_test
	
	#pctg_correct = len([x for x in difference if x == 0])/float(len(difference)) 

print("lol")
