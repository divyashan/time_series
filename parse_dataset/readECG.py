import os
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
import h5py
import pandas as pd


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

def get_f_path(pid, folder):
	f_path_1 = "./datasets/adjacent_beats/" + folder + "/patient_" + str(pid) + ".0.csv"
	f_path_2 =  "./datasets/adjacent_beats/" + folder + "/patient_" + str(int(pid)) + ".0.csv"
	f_path = None
	if os.path.isfile(f_path_1):
		f_path = f_path_1
	elif os.path.isfile(f_path_2):
		f_path = f_path_2
	return f_path

def loadECG(mode=None):
	train_normal_patients = [1,   10,  100, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,  101, 1010, 1011, 1012, 1014, 1015, 1016, 1017, 969,  97, 970, 971, 972, 973, 974, 975, 976, 978, 979,  98, 980, 981, 982, 983, 984, 985, 986, 988, 989,  99, 990, 992, 993, 994, 995, 996, 998, 999]
	train_death_patients = [10171, 10239, 1031, 10395, 10422, 1050, 10502, 1087, 1125, 1136, 1175, 1189, 1197, 1241, 1273, 1413, 1515, 1630, 1646, 1698, 1710, 1720, 1726, 1741, 1763, 1816, 1921, 1931, 1938, 2036, 2236, 2252, 2317, 2343, 2388, 2438, 2484, 2617, 2628, 2635, 2718, 2785, 2899, 3046, 3053, 3216, 3232, 3245, 3277, 3319, 3348, 3350, 3525, 3541, 3548, 3570, 3596, 3617, 3630, 3792, 3797, 3865, 3875, 3991, 4000, 4128, 4153, 4210, 4249, 4353]
	test_normal_patients = [1018, 1019,  102, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,  103,  104, 1047, 1048, 1049,  105, 1051]
	test_death_patients = [4381, 4383, 4467, 4516, 4659, 4740, 4763, 48, 4844, 4879, 5133, 5180, 5187, 5194, 5506]


	if mode=="set":
		train_death_patients = [10171.0, 10239.0]
		train_normal_patients = [80.0, 3.0]
		test_death_patients= [5133.0]
		test_normal_patients= [999.0]
	else:
		# train_normal_patients = train_normal_patients[:5]
		train_death_patients = train_death_patients[:50]
		test_normal_patients = test_normal_patients[:15]
		# test_death_patients = test_death_patients[:5]

	train_death_list = []
	test_death_list = []
	train_normal_list = []
	test_normal_list = []

	pdb.set_trace()

	print len(train_death_patients), len(test_death_patients), len(train_normal_patients), len(test_normal_patients)
	n_death = 500
	n_normal = 500
	n_death_per_patient = n_death/len(train_death_patients)
	n_normal_per_patient = n_normal/len(train_normal_patients)

	for pid in train_death_patients:
		f_path = get_f_path(pid, "death")
		if not f_path:
			print "Couldn't find: ", pid
			continue
		pid_mat = pd.read_csv(f_path, delim_whitespace=True).values
		train_death_list.append(pid_mat[:1000])

	for pid in test_death_patients:
		f_path = get_f_path(pid, "death")
		if not f_path:
			print "Couldn't find: ", pid
			continue
		pid_mat = pd.read_csv(f_path, delim_whitespace=True).values
		test_death_list.append(pid_mat[:1000])
	
	for pid in train_normal_patients:
		f_path = get_f_path(pid, "normal")
		if not f_path:
			print "Couldn't find: ", pid
			continue
		pid_mat = pd.read_csv(f_path, delim_whitespace=True).values
		train_normal_list.append(pid_mat[:1000])

	for pid in test_normal_patients:
		f_path = get_f_path(pid, "normal")
		if not f_path:
			print "Couldn't find: ", pid
			continue
		pid_mat = pd.read_csv(f_path, delim_whitespace=True).values
		test_normal_list.append(pid_mat[:1000])

	f = h5py.File("mytestfile.hdf5", "w")

	
	if mode == "set":
		train_death_list = np.swapaxes(np.array(train_death_list), 0, 1)
		test_death_list = np.swapaxes(np.array(test_death_list), 0, 1)
		train_normal_list = np.swapaxes(np.array(train_normal_list), 0, 1)
		test_normal_list = np.swapaxes(np.array(test_normal_list), 0, 1)
		
		train = np.concatenate([train_death_list, train_normal_list], axis=1)
		test = np.concatenate([test_death_list, test_normal_list], axis=1)

		X_train, y_train = train[:,:,2:], train[:,:,1]
		pdb.set_trace()
		y_train = np.concatenate([np.ones((len(train_death_patients), 1)), np.zeros((len(train_normal_patients), 1))])
		X_test, y_test = test[:,:,2:], test[:,:,1]
		y_test = np.concatenate([np.ones((len(test_death_patients), 1)), np.zeros((len(test_normal_patients, 1)))])


		pdb.set_trace()
		return X_train, y_train, X_test, y_test

	else:
		train_death_list = np.concatenate(train_death_list)
		test_death_list = np.concatenate(test_death_list)
		train_normal_list = np.concatenate(train_normal_list)
		test_normal_list = np.concatenate(test_normal_list)

		train = np.concatenate([train_death_list, train_normal_list], axis=0)
		test = np.concatenate([test_death_list, test_normal_list], axis=0)

		X_train, y_train = train[:,2:], train[:,1]
		X_test, y_test = test[:,2:], test[:,1]

		return np.expand_dims(X_train, 1), y_train, np.expand_dims(X_test, 1), y_test

