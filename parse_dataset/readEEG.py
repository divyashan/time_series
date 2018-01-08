import numpy as np 

import pdb


def loadEEG():
	X_train = np.load('./datasets/eeg/SMNI_CMI_TRAIN/all_patients_ts.txt')
	Y_train = np.squeeze(np.load('./datasets/eeg/SMNI_CMI_TRAIN/all_patients_labels.txt'))
	X_test = np.load('./datasets/eeg/SMNI_CMI_TEST/all_patients_ts.txt')
	Y_test = np.squeeze(np.load('./datasets/eeg/SMNI_CMI_TEST/all_patients_labels.txt'))
	return X_train, Y_train, X_test, Y_test
