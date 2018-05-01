import matplotlib.pyplot as plt
import numpy as np 
import os 
import pdb

from scipy.stats import mode
from scipy.spatial.distance import cdist
from time_series.models.utils import classify_sample


# Input sequence of embedded representations of adjacent beats


DEATH_DATASET_DIR = "./datasets/adjacent_beats/death/patient_"
NORMAL_DATASET_DIR = "./datasets/adjacent_beats/normal/"

def classify_patient(pid, tr_embedding, tr_y, model, model_type="1"):
	adjacent_beats = get_all_adjacent_beats(pid)
	if adjacent_beats: 
		embedded_signal = embed_signal(adjacent_beats, tr_y, model)
		if model_type == "1":
			return model_1_prediction(tr_embedding, tr_y, embedding_signal)
		return model_2_prediction(tr_embedding, tr_y, embedding_signal)
	else:
		return "Could not locate adjacent beats file for Patient ", pid

def get_all_adjacent_beats(pid, first_n=None):
	death_f_path = "./datasets/adjacent_beats/death/patient_" + str(pid) + ".0.csv"
	normal_f_path = "./datasets/adjacent_beats/normal/patient_" + str(pid) + ".0.csv"
	pid_mat = None
	if os.path.exists(death_f_path): 
		pid_mat = np.expand_dims(np.loadtxt(death_f_path)[:,2:], axis=2)
	elif os.path.exists(normal_f_path): 
		pid_mat = np.expand_dims(np.loadtxt(normal_f_path)[:,2:], axis=2)
	if first_n:
		return pid_mat[:first_n]
	return pid_mat

def embed_signal(adjacent_beats, tr_y, model):
	return model.eval(feed_dict = {x: adjacent_beats, y_: tr_y, keep_prob: 1.0})

# Model 1: Take the majority nearest neighbor classification as the actual classification
def model_1_prediction(tr_embedding, tr_y, embedded_signal, threshold=.5):
	# Return 1 if signal is high risk, 0 if not 
	# Takes majority vote for each embedded pair of adjacent beats
	pairwise_dist = cdist(tr_embedding, embedded_signal)
	pairwise_mins = np.argmin(pairwise_dist, axis=0)
	guesses = tr_y[pairwise_mins]
	if float(np.sum(guesses))/len(guesses.flatten()) > threshold:
		return 1
	return 0

def percentage_death(tr_embedding, tr_y, embedded_signal, threshold=None):
	pairwise_dist = cdist(tr_embedding, embedded_signal)
	training_mins = np.argmin(pairwise_dist, axis=0)
	training_dists = np.min(pairwise_dist, axis=0)
	n_samples = len(embedded_signal)
	if threshold:
		guesses = [tr_y[training_mins[i]] for i in range(n_samples) if training_dists[i] < threshold]
		if len(guesses) == 0:
			print "Embedded signal shape: ", embedded_signal.shape
			print training_dists
			return 0
	else:
		guesses = tr_y[training_mins]
	# TODO: normalize by length of embedded signal (basically biasing towards normal classification)
	#		or normalize by length of all 'confident guesses'?
	return float(np.sum(guesses))/n_samples


# Model 2: Use cluster center of the death adjacent beats and the cluster center of the normal adjacent beats
# 		   Use upper quartile of distances to the death-cluster
def model_2_prediction(embedded_signal, tr_embedding, tr_y):
	# Return 1 if signal is high risk
	# Cluster-center based distance metric for risk
	# Averages distance to cluster center for death across the entire signal

	# The smaller the distance, the higher the risk

	death_idxs = np.where(tr_y == 1)[0]
	normal_idxs = np.where(tr_y == 0)[0]
	death_center = np.mean(tr_embedding[death_idxs], axis=1)
	normal_center = np.mean(tr_embedding[normal_idxs], axis=1)

	return np.mean([np.linalg.norm(pair-death_center) for pair in embedded_signal])

