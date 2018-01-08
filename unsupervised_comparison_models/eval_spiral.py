import pandas as pd 
import numpy as np 
import pdb
import sys

sys.path.insert(0, '../../')
from utils import MV_DATASETS, standardize_ts_lengths_1
from sklearn.cluster import KMeans
from ts_autoencoder import eval_clustering
#from time_series.src.clean_datasets import cv_splits_for_dataset





SPIRAL_PATH = "../datasets/pca/"


results = []
for dataset in MV_DATASETS:
	print dataset

	"""
	dataset_list = cv_splits_for_dataset(dataset)

	n_fold = 0
	X_train = dataset_list[n_fold].X_train
	y_train = dataset_list[n_fold].y_train
	X_test = dataset_list[n_fold].X_test
	y_test = dataset_list[n_fold].y_test

	n = max([np.max([v.shape[0] for v in X_train]), np.max([v.shape[0] for v in X_test])])

	X_train = standardize_ts_lengths_1(X_train, n)
	X_test = standardize_ts_lengths_1(X_test, n)

	
  	"""
	train_f = np.loadtxt(SPIRAL_PATH + dataset + '.txt', delimiter=',')
	test_f = np.loadtxt(SPIRAL_PATH + dataset + '.txt', delimiter=',')



	X_train = train_f[:, 1:]
	y_train = train_f[:, 0]

	X_test = test_f[:, 1:]
	y_test = test_f[:, 0]

	n_classes = len(np.unique(y_test))

	all_X = np.concatenate((X_train, X_test))
  	all_y = np.concatenate((y_train, y_test))

  	#all_X = np.reshape(all_X, (all_X.shape[0], all_X.shape[1]*all_X.shape[2]))
  	#X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) )
  	#X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1]*X_test.shape[2]) )


	y_pred =  KMeans(n_clusters=n_classes, random_state=0).fit(all_X).labels_

	ri = eval_clustering(y_pred, all_y)
	results.append((dataset, ri))


pdb.set_trace()

