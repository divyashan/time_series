import os
import numpy as np
import pdb
from sklearn.model_selection import train_test_split


DATASET_DIR = "../jiffy_experiments/adjacent_beats/"

def get_dataset_path(pid):
	return DATASET_DIR + "patient_" + str(pid) + ".csv"
"""
def loadECG():

	train_files = [80.0, 3.0, 102.0, 762.0 ]
	test_files = [694.0, 9.0]]
	test_list = []
	train_list = []
	for i in train_files:

		train_death = np.loadtxt( DATASET_DIR + "death_" +  str(i) + ".csv", delimiter=",")	
		train_survive = np.loadtxt(DATASET_DIR + "survive_" + str(i) + ".csv", delimiter=",")
		train_list.append(train_death)
		train_list.append(train_survive)


	for i in test_files:

		test_death = np.loadtxt( DATASET_DIR + "death_" +  str(i) + ".csv", delimiter=",")	
		test_survive = np.loadtxt(DATASET_DIR + "survive_" + str(i) + ".csv", delimiter=",")
		test_list.append(test_death)
		test_list.append(test_survive)
	train = np.concatenate(train_list)
	test = np.concatenate(test_list)

	X_train = train[:,2:]
	y_train = train[:,1]

	X_test = test[:,2:]
	y_test = test[:,1]


	return np.expand_dims(X_train, 1), y_train, np.expand_dims(X_test, 1), y_test
"""
def loadECG():
	death_patients = [80.0, 1050.0, 102.0] 
	normal_patients = [1.0, 1030.0, 1070.0]


	death_list = []
	normal_list = []
	for pid in death_patients:
		f_path = "./datasets/adjacent_beats/death/patient_" + str(pid) + ".csv"
		pid_mat = np.loadtxt(f_path)
		death_list.append(pid_mat)
	
	for pid in normal_patients:
		f_path = "./datasets/adjacent_beats/normal/patient_" + str(pid) + ".csv"
		pid_mat = np.loadtxt(f_path)
		normal_list.append(pid_mat)
	n_death = 1100
	n_normal = 1000
	death_list = np.concatenate(death_list)
	normal_list = np.concatenate(normal_list)

	death_idxs = np.random.choice(len(death_list), n_death*2)	
	normal_idxs = np.random.choice(len(normal_list), n_normal*2)
	train = np.concatenate([death_list[death_idxs[:n_death]], normal_list[normal_idxs[:n_normal]]])
	test = np.concatenate([death_list[death_idxs[n_death:]], normal_list[normal_idxs[n_normal:]]])
	
	X_train = train[:,2:]
	y_train = train[:,1]

	X_test = test[:,2:]
	y_test = test[:,1]

	print "Finished loading data"
	return np.expand_dims(X_train, 1), y_train, np.expand_dims(X_test, 1), y_test

