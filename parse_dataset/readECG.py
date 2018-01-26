import os
import numpy as np
import pdb
from sklearn.model_selection import train_test_split

def loadECG():

	f_path = u'/afs/csail.mit.edu/u/d/divyas/Documents/jiffy_experiments/patient_data_subset.csv'
	xx = np.loadtxt(f_path, delimiter=",")
	pdb.set_trace()

	X_data = xx[:,2:]
	y_data = xx[:,1]

	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.2)
	return X_train, X_test, y_train, y_test
