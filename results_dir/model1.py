from fastdtw import fastdtw
from readUcr import UCRDataset
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import argrelextrema
import pdb 
import os


# Separate dataset into classes
	# Done as a part of class


def choose_class_rep(X_c):
	return np.mean(X_c, axis=0)

def intervals_to_shapelets(c_rep, intervals):
	shapelets = []
	for interval in intervals:
		shapelets.append(c_rep[interval[0]:interval[1]])
	return shapelets

def filter_intervals(intervals):
	n_intervals=len(intervals)
	for i, interval in enumerate(intervals):
		interval_after = intervals[min(i+1,n_intervals-1)]
		interval_before = intervals[max(0, i-1)]
		if interval[1] == interval_after[0]:
			interval[1] = interval_after[1]
		if interval[0] == interval_before[1]:
			interval[0] = interval_before[0]

	return list(set(intervals))


def initialize_shapelets(c_rep, n):
	# Input: Class representative example, number of shapelets desired
	# Return: List of shapelets [s_1, s_2, ..., s_n] 
	# 		  where s_1 represents a shapelet sequence
	local_maxima = argrelextrema(x, np.greater)
	intervals = [(maxima-1, maxima+1) for maxima in local_maxima]
	return intervals, intervals_to_shapelets(c_rep, intervals)


def score_shapelet(shapelet, X_c, X_d):
	same_sample_wise = [fastdtw(shapelet, sample)[0] for sample in X_c]
	diff_sample_wise = [fastdtw(shapelet, sample)[0] for sample in X_d]
	return (np.sum(same_sample_wise) - np.sum(diff_sample_wise))/len(shapelet)

def find_shapelets(UCRDataset):
	classes = UCRDataset.intervals.keys()
	n_classes = len(classes)*4
	interval_map = dict()
	shapelet_map = dict()
	X_d = UCRDataset.get_class[classes[-1]]
	for c in classes:
		X_c = UCRDataset.get_class(c)
		c_rep = choose_class_rep(X_c)
		intervals, shapelets = initialize_shapelets(c_rep, n_classes)
		best_score = float("-inf")
		score = 0
		while score > best_score:
			scores = [score_shapelet(shapelet, X_c, X_d) for shapelet in shapelets]
			best = np.argmax(scores)
			score = sum(scores)
			if score > best_score:
				# Update intervals and shapelets
				intervals[best][0] = max(intervals[best]-1, 0)
				intervals[best][1] = min(intervals[best]+1, len(c_rep))
				intervals = filter_intervals(intervals)
				shapelets = intervals_to_shapelets(c_rep, intervals)
				best_score = score

			else:
				# The total DTW distance has increased
				# (previous intervalss were better)
				break
		X_d = X_c
		intervals_map[c] = intervals
		shapelet_map[c] = shapelets
	return interval_map, shapelet_map

def shapelet_distance(shapelets, splits, sample):
	score = [fastdtw(shapelet, sample[splits[i]:splits[i+1]])[0] for i, shapelet in enumerate(shapelets)]
	return sum(score)


def classify_sample(UCRDataset, shapelet_map, splits):
	guess = []
	labels = UCRDataset.intervals.keys()
	for sample, correct_label in zip(UCRDataset.Xtest, UCRDataset.Ytest):
		scores = [shapelet_distance(shapelet_map[label], splits[label], sample) for label in labels]
		label = labels[np.argmin(scores)]
		guess.append(label)
	print "Percent incorrect: ", np.count_nonzero(UCRDataset.Ytest-np.array(guess))/float(len(UCRDataset.Xtest))


all_datasets = [x[0] for x in os.walk("../ucr_data")][1:]
for subdir in all_datasets:
	print subdir
	a = UCRDataset(subdir)
	split_map, shapelet_map = find_shapelets(a)
	classify_sample(a, shapelet_map, split_map)


