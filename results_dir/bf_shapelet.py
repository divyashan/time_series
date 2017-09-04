import numpy as np
import sys
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from readUcr import UCRDataset

import pdb

# Brute force shapelet implementation


def s_dist(s, sequence):
	s_length = len(s)
	distances = np.array([fastdtw(s, sequence[i:i+s_length])[0] for i in range(len(sequence)-s_length)])
	return np.min(distances)


def Entropy(labels):
	unique_labels = np.unique(labels)
	n_labels = len(labels)
	e = 0
	for l in unique_labels:
		p_l = len(np.where(labels==l)[0])/float(n_labels)
		if p_l != 0:
			e += p_l*np.log2(p_l)
	return -e


def CalcInfoGain(labels, labels_l, labels_r):
	n_l = len(labels_l)
	n_r = len(labels_r)
	n = len(labels)
	return Entropy(labels) - (float(n_l)/n)*Entropy(labels_l) - (float(n_r)/n)*Entropy(labels_r)

def ContiguousGain(candidate, TS, labels, target_label):
	dtw_dists = np.array([s_dist(candidate, seq) for seq in TS])
	sorted_idx = np.argsort(dtw_dists)
	dtw_dists = dtw_dists[sorted_idx]
	sorted_labels = np.copy(labels[sorted_idx])
	return -np.sum(np.where(sorted_labels==target_label)[0])


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
def GenerateAllShapelets(TS, length):
	#return [TS[i:i+length] for i in range(len(TS)-length)]
	
	shapelets = []
	for seq in TS:
		subseqs = [seq[i:i+length] for i in range(len(seq)-length)]
		shapelets.extend(subseqs)
	return shapelets
	

def plot(seq):
	plt.plot(seq)
	plt.show()


def FindBestSplit(candidate, TS, labels):
	dtw_dists = np.array([s_dist(candidate, seq) for seq in TS])
	sorted_idx = np.argsort(dtw_dists)
	dtw_dists = dtw_dists[sorted_idx]
	sorted_labels = np.copy(labels[sorted_idx])
	best_split = None
	best_ig = float('-inf')
	for i in range(len(sorted_labels)-1):
		gain = CalcInfoGain(sorted_labels, sorted_labels[:i+1], sorted_labels[i+1:])

		if gain > best_ig:
			best_ig = gain
			best_split = i
	return .5*(dtw_dists[best_split] + dtw_dists[best_split+1]), best_ig, best_split

def FindBestShapelet(TS, labels, idxs):
	best_gain = 0
	best_c_gain = float('-inf')
	s = []
	best_sp_idx = 0
	best_sp = 0
	poss_lengths = [6]
	target_label = labels[idxs[0]]
	for i in poss_lengths:
		candidates = GenerateAllShapelets(TS[idxs], i)
		print "Testing ", len(candidates), " shapelets of length: ", i
		for i,candidate in enumerate(candidates):
			if i % 100 == 0:
				print 'Sample ', i
			
			c_gain = ContiguousGain(candidate, TS, labels, target_label)
			if c_gain < best_c_gain:
				continue

			best_c_gain = c_gain
			print best_c_gain
			sp, gain, sp_idx = FindBestSplit(candidate, TS, labels)

			if gain > best_gain:
				best_gain = gain
				print 'BEST_GAIN: ', best_gain
				s.append(candidate)
				best_sp_idx = sp_idx
				best_sp = sp
	return s
	#return s, best_sp, best_sp_idx
 
def DivideDataset(shapelet, sp_idx, TS, labels):
 	dtw_dists = np.array([s_dist(shapelet, seq) for seq in TS])
 	sorted_idx = np.argsort(dtw_dists)
	dtw_dists = dtw_dists[sorted_idx]
	sorted_labels = np.copy(labels[sorted_idx])
	sorted_TS = np.copy(TS[sorted_idx])
	labels_1, TS_1 = sorted_labels[:sp_idx], sorted_TS[:sp_idx]
	labels_2, TS_2 = sorted_labels[sp_idx:], sorted_TS[sp_idx:]
	return TS_1, TS_2, labels_1, labels_2
"""
datasets = [(TS, labels)]
all_shapelets = []
first = True
for i in range(5):
	while len(datasets):
		d, l = datasets.pop(0)
		if first:
			s = np.array([ 0.07079 , -0.071016, -0.20347 , -0.31356 , -0.39424 , -0.44855 ])
			sp_idx = 25
		else:
			s, sp, sp_idx = FindBestShapelet(d, l)
		
		all_shapelets.append(s)
		TS_1, TS_2, labels_1, labels_2 = DivideDataset(s, sp_idx, d, l)
		if len(set(labels_1)) != 1:
			datasets.append((TS_1, labels_1)) 
		if len(set(labels_2)) != 1:
			datasets.append((TS_2, labels_2))
		first = False
		pdb.set_trace()
"""

def find_min_shapelet_path(ts_dist_1, ts_dist_2):
	return min([ts_dist_1[i] + ts_dist_2[i] for i in range(len(ts_dist_1))])



# Given shapelets create distance matrix bt all TS 
# Matrix of each time series' distance to each of the shapelets
#	dist[i][j] distance from TS i to shapelet J 
def create_dist_matrix(TS, shapelets):
	d = np.zeros((TS.shape[0], len(shapelets)))
	for i,ts_sample in enumerate(TS):
		d[i] = np.array([s_dist(shapelet, ts_sample) for shapelet in shapelets])
	dists = np.zeros((TS.shape[0], TS.shape[0]))
	for i in range(len(TS)):
		for j in range(len(TS)):
			if i == j:
				dists[i][j] = 0
				continue
			dists[i][j] = find_min_shapelet_path(d[i], d[j])
	return dists

# Find best shapelet/threshold for a given dataset
dataset_name = "../ucr_data/" + sys.argv[1]
data = UCRDataset(dataset_name)

TS = data.Xtrain
labels = data.Ytrain
test = data.Xtest
test_labels = data.Ytest 
pdb.set_trace()
unique_labels = np.unique(labels)
all_s = []

for l in unique_labels:
	l_idxs = np.where(labels==l)[0][:3]
	s = FindBestShapelet(TS, labels, l_idxs)
	pdb.set_trace()
	all_s.append(s)


all_s = [np.array([ 0.14468 ,  0.018479, -0.10956 , -0.2145  , -0.31115 , -0.3732  ]), np.array([ 0.26469  ,  0.0063638, -0.29055  , -0.43169  , -0.5907   , -0.74768  ])]
"""
def test():
	all_s = [np.array([ 0.081541, -0.036605, -0.1648  , -0.25659 , -0.34733 , -0.411   ]), np.array([ 0.22428 ,  0.016737, -0.20603 , -0.41688 , -0.6253  , -0.80642 ])]
	x = create_dist_matrix(TS, all_s)
	return x
"""

def evaluate_dist_matrix(dist_matrix, labels):
	guesses = [labels[np.argsort(seq)[1]] for seq in dist_matrix]
	print guesses-labels