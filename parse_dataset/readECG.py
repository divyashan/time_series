import os
import numpy as np
import pdb
from sklearn.model_selection import train_test_split


DATASET_DIR = "/Volumes/My Book/merlin_final"

"""
def loadECG():

	train_files = [1, 2, 3, 4]
	test_files = [1, 2, 3, 4]
	train_death = 
	for i in train_files:

	train_death = np.loadtxt("./datasets/patient_data_death_0.csv", delimiter=",")
	test_death = np.loadtxt("./datasets/patient_data_death_1.csv", delimiter=",")
	train_survive = np.loadtxt("./datasets/patient_data_survive_0.csv", delimiter=",")
	test_survive = np.loadtxt("./datasets/patient_data_survive_1.csv", delimiter=",")

	train = np.concatenate((train_death, train_survive))
	test = np.concatenate((test_death, test_survive))

	X_train = train[:,2:300000]
	y_train = train[:,1]

	X_test = test[:,2:300000]
	y_test = test[:,1]


	return np.expand_dims(X_train, 1), y_train, np.expand_dims(X_test, 1), y_test
"""
def loadECG():
	train_patients = [80.0, 102.0, 1.0,3.0]
	test_patients = [694.0, 704.0, 4.0, 18.0]


	train_list = []
	test_list = []
	for pid in train_patients:
		f_path = "./datasets/adjacent_beats/patient_" + str(pid) + ".csv"
		pid_mat = np.loadtxt(f_path)
		train_list.append(pid_mat)
	
	for pid in test_patients:
		f_path = "./datasets/adjacent_beats/patient_" + str(pid) + ".csv"
		pid_mat = np.loadtxt(f_path)
		test_list.append(pid_mat)

	train = np.concatenate(train_list)
	test = np.concatenate(test_list)
	pdb.set_trace()
	X_train = train[:,2:]
	y_train = train[:,1]

	X_test = test[:,2:]
	y_test = test[:,1]

	print "Finished loading data"
	return np.expand_dims(X_train, 1), y_train, np.expand_dims(X_test, 1), y_test
