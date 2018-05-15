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
	train_normal_patients = [127, 698, 4976, 1049, 5636, 788, 51, 784, 993, 128, 995, 5655, 5106, 374, 980, 876, 106, 1091, 4257, 4419, 999, 751, 65, 892, 3722, 825, 494, 5482, 1129, 1014, 1006, 1053, 5523, 1134, 4240, 1110, 39, 4096, 3680, 1133, 5755, 1130, 656, 370, 1114, 1127, 136, 3212, 2832, 1171, 133, 937, 1117, 500, 939, 1051, 932, 2844, 1113, 1060, 699, 1008, 4517, 1088, 1142, 1062, 61, 5563, 785, 652, 917, 1024, 789, 829, 1027, 4263, 1147, 1009, 903, 820]
	train_death_patients = [4210, 3525, 1630, 1921, 2628, 3216, 2317, 961, 3245, 5133, 1189, 1720, 3617, 1816, 5687, 3046, 3319, 5506, 4659, 5180, 5964, 1031, 1087, 4763, 1726, 3991, 4844, 3232, 1698, 3277, 2236, 4879, 2343, 1646, 4128, 4249, 1241, 4383, 4353, 3797, 3875, 3570, 3630, 2718, 1197, 1763, 2388, 2252, 3348, 3541, 3053, 2635, 4153, 1125, 1136, 832, 10395, 5908, 10422, 1175, 6311, 5187, 5194, 1710, 2484, 2617, 2438, 2785, 2036, 10171, 4740, 1931, 4000, 1273, 48, 10502, 1938, 1515, 1741, 4467]
	test_normal_patients = [1, 10, 100, 1001, 1002, 1003, 1004, 1005, 1007, 101, 1010, 1011, 1012, 1015, 1016, 1017, 1018, 1019, 102, 1020, 1021, 1022, 1023, 1025, 1026, 1028, 1029, 103, 104, 1047, 1048, 105, 1052, 1054, 1055, 1056, 1057, 1058, 1059, 1061, 1063, 1064, 1065, 1066, 1067, 1069, 107, 1070, 1071, 1072, 1073, 1075, 1076, 108, 1086, 1089, 109, 1090, 1092, 1093, 1094, 1095, 11, 110, 1104, 1105, 1106, 1107, 1109, 111, 1111, 1112, 1115, 1116, 1118, 1119, 112, 1120, 1121, 1123, 1124, 1126, 1128, 113, 1131, 1135, 1138, 1139, 114, 1140, 1141, 1143, 1146, 1148, 1149, 115, 116, 117, 1176, 1179, 118, 1182, 1188, 119, 12, 120, 121, 123, 1232, 124, 125, 126, 129, 13, 130, 131, 132, 1332, 134, 135, 137, 138, 139, 140, 1417, 144, 1463, 15, 1547, 1554, 16, 1616, 1644, 1687, 17, 1723, 1725, 1727, 1731, 1733, 1746, 18, 1804, 19, 1902, 1936, 1992, 20, 2004, 21, 2152, 22, 2234, 2267, 2290, 23, 2390, 24, 25, 2507, 2555, 2556, 2592, 26, 27, 2705, 2779, 28, 2812, 2831, 2833, 2841, 2865, 29, 2965, 2966, 2987, 3, 30, 3010, 3014, 3016, 3018, 3040, 3054, 3063, 3077, 31, 3181, 32, 3217, 3224, 3239, 3249, 3255, 3266, 3268, 33, 34, 3445, 3495, 3522, 3561, 3566, 3575, 36, 3621, 3671, 3679, 368, 369, 37, 3704, 371, 3717, 372, 3723, 3727, 373, 3748, 375, 3751, 376, 377, 3779, 3787, 379, 38, 3801, 381, 382, 3861, 3938, 3950, 3957, 4, 40, 4002, 4010, 4018, 4073, 4132, 4169, 4171, 4174, 4193, 42, 4236, 4245, 4248, 43, 437, 439, 4391, 44, 4407, 441, 442, 4442, 4450, 4468, 45, 4523, 4585, 46, 47, 4741, 4747, 4755, 4756, 4766, 4840, 4841, 4869, 4882, 49, 4917, 495, 4954, 497, 4975, 498, 4980, 4982, 499, 4992, 5, 50, 5004, 502, 503, 5065, 5072, 5083, 5088, 5098, 5120, 5123, 5144, 5157, 5171, 52, 5235, 528, 529, 53, 54, 5429, 5444, 5480, 5488, 55, 5503, 5513, 5531, 5533, 5545, 5582, 5593, 5601, 5605, 5610, 5629, 564, 5644, 5646, 5649, 567, 5681, 5688, 57, 5703, 5713, 5717, 5719, 5732, 5733, 5745, 5747, 58, 584, 5882, 59, 5903, 5924, 6, 60, 6002, 6086, 6102, 6114, 6120, 613, 614, 615, 618, 62, 6240, 629, 63, 64, 647, 651, 655, 657, 658, 659, 66, 660, 661, 662, 663, 664, 668, 67, 68, 680, 681, 687, 688, 69, 690, 691, 694, 695]
	test_death_patients = [10239, 1050, 1413, 2899, 3350, 3548, 3596, 3792, 3865, 4381, 4516, 5616, 6160, 760, 854]
	
	if mode=="set":
		train_death_patients = [10171.0, 10239.0]
		train_normal_patients = [80.0, 3.0]
		test_death_patients= [5133.0]
		test_normal_patients= [999.0]
	else:
		train_normal_patients = train_normal_patients[:50]
		train_death_patients = train_death_patients[:50]
		test_normal_patients = test_normal_patients[:15]
		test_death_patients = test_death_patients[:15]

	train_death_list = []
	test_death_list = []
	train_normal_list = []
	test_normal_list = []


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


	
	if mode == "set":
		train_death_list = np.swapaxes(np.array(train_death_list), 0, 1)
		test_death_list = np.swapaxes(np.array(test_death_list), 0, 1)
		train_normal_list = np.swapaxes(np.array(train_normal_list), 0, 1)
		test_normal_list = np.swapaxes(np.array(test_normal_list), 0, 1)
		
		train = np.concatenate([train_death_list, train_normal_list], axis=1)
		test = np.concatenate([test_death_list, test_normal_list], axis=1)

		X_train, y_train = train[:,:,2:], train[:,:,1]
		y_train = np.concatenate([np.ones((len(train_death_patients), 1)), np.zeros((len(train_normal_patients), 1))])
		X_test, y_test = test[:,:,2:], test[:,:,1]
		y_test = np.concatenate([np.ones((len(test_death_patients), 1)), np.zeros((len(test_normal_patients), 1))])


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

