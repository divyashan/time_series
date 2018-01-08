import os
import numpy as np 
from joblib import Memory

from .. import paths

_memory = Memory('.', verbose=0)

def train_X():
	f_path = os.path.join(paths.EEG, 'SMNI_CMI_TRAIN/all_patients_ts.txt')
	X_train = np.load(f_path)
	return X_train

def train_labels():
	f_path = os.path.join(paths.EEG, 'SMNI_CMI_TRAIN/all_patients_labels.txt')
	Y_train = np.squeeze(np.load(f_path))
	return Y_train 

def test_X():
	f_path = os.path.join(paths.EEG, 'SMNI_CMI_TEST/all_patients_ts.txt')
	X_test = np.load(f_path)
	return X_test

def test_labels():
	f_path = os.path.join(paths.EEG, 'SMNI_CMI_TEST/all_patients_labels.txt')
	Y_test = np.squeeze(np.squeeze(np.load(f_path)))
	return Y_test

def train_data():
    return train_X(), train_labels()


def test_data():
    return test_X(), test_labels()