import pdb 
import matplotlib.pyplot as plt
import numpy as np
from utils import produce_pair_labels


dtw_tr = np.ndarray.flatten(np.loadtxt('dtw_train_dist.txt'))
rcnn_tr = np.ndarray.flatten(np.loadtxt('rob_cnn_train_dists.txt'))
main_tr = np.ndarray.flatten(np.loadtxt('main_train_dists.txt'))

tr_labels = produce_pair_labels(np.loadtxt('gun_point_train_labels.txt'))
te_labels = produce_pair_labels(np.loadtxt('gun_point_test_labels.txt'))

dtw_te = np.ndarray.flatten(np.loadtxt('dtw_test_dist.txt'))
rcnn_te = np.ndarray.flatten(np.loadtxt('rob_cnn_test_dists.txt'))
main_te = np.ndarray.flatten(np.loadtxt('main_test_dists.txt'))

def scatter_plot(dists, labels, stagger):
	sort_ind = np.argsort(dists)
	sorted_dists = dists[sort_ind]
	sorted_labels = labels[sort_ind]
	plt.scatter(sorted_dists, [i%stagger for i in range(len(dists))], 20, c=sorted_labels, edgecolor='none')
	plt.show()

pdb.set_trace()
