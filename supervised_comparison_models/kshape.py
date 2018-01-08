import sys
import os
import time
import numpy as np
import pdb

sys.path.insert(0, '../')

from time_series.kshape.kshape.core import kshape, zscore
from sklearn.metrics.cluster import normalized_mutual_info_score


from time_series.parse_dataset.readUcr import UCRDataset
from time_series.models.utils import load_dataset

from utils import get_tp_fp_tn_fn, rebase_labels


def test_model(dataset):
	X_train, y_train, X_test, y_test = load_dataset(dataset)

	labels = np.unique(y_train)
	y_train = rebase_labels(y_train, labels)
	y_test = rebase_labels(y_test, labels)

	all_X = np.concatenate((X_train, X_test))
	all_y = np.concatenate((y_train, y_test))

	n_clusters = len(np.unique(y_train))

	start = time.clock()
	clusters = kshape(all_X, n_clusters)
	finish = time.clock()

	labels = np.unique(y_train)
	co_matrix = np.zeros((len(labels), n_clusters))
	for i, cluster in enumerate(clusters):
		idxs = cluster[1]
		for idx in idxs:
			co_matrix[int(all_y[idx])][i] += 1

	assigned_clusters = np.zeros(all_y.shape)
	for i, cluster in enumerate(clusters):
		for idx in cluster[1]:
			assigned_clusters[idx] = i


	tp, fp, tn, fn = get_tp_fp_tn_fn(co_matrix)

	rand_ind = float(tp + tn) / (tp + fp + fn + tn)
	print 'Rand Index: ', rand_ind
	return rand_ind, 0, finish-start

if __name__ == '__main__':
	dataset = sys.argv[1]
	RI, train_time, inf_time = test_model(dataset)