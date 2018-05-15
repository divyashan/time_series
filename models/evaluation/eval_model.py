import numpy as np
import pdb
from keras.models import load_model
import sys
import h5py
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '../')

m = load_model("fc_3")
embedding_m = load_model("fc_embedding_3")

hf = h5py.File('train_test', 'r')
train = np.expand_dims(np.array(hf.get('train')), 2)
test = np.expand_dims(np.array(hf.get('test')), 2)

train_embedding = embedding_m.predict(train[:,2:,:])
test_embedding = embedding_m.predict(test[:,2:,:])

y_pred = m.predict(test[:,2:,:])
y_test = test[:,1,:].flatten()
test_patients = list(set(test[:,0,:].flatten()))
pdb.set_trace()
patient_scores = {}
for i, pid in enumerate(test_patients):
	pid_pred = y_pred[np.where(test[:,:,0] == pid)[0]]
	label = y_test[np.where(test[:,:,0] == pid)[0]][0]
	patient_scores[pid] = (label, pid_pred)
	print("Pid #: ", i)
pdb.set_trace()

