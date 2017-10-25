import numpy as np
import tensorflow as tf
from itertools import product
import scipy
import pdb
from scipy.misc import comb
import matplotlib.pyplot as plt

NUM_DIFF=10
UCR_DATASETS = ['50Words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'ChlorineConcentration', 
                          'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction', 
                          'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays', 
                          'Earthquakes', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'Fish', 'FordA', 'FordB', 'Gun_Point', 'Ham', 
                          'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 
                          'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 
                          'MiddlePhalanxTW', 'MoteStrain', 'Non-Invasive Fetal ECG Thorax1', 'Non-Invasive Fetal ECG Thorax2', 'OSU Leaf', 
                          'OliveOil Tony Bagnall', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 
                          'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll', 
                          'SmallKitchenAppliances', 'SonyAIBORobot Surface', 'SonyAIBORobot SurfaceII', 'StarLightCurves', 'Strawberry', 
                          'Swedish Leaf', 'Symbols', 'Synthetic Control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'Two Patterns', 
                          'TwoLeadECG', 'UWaveGestureLibraryAll', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga', 
                          'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z']
MV_DATASETS = ['arabic_digits', 'auslan', 'trajectories', 'libras', 'wafer', 'ecg']
NEG_LABEL = 1
SAME_LABEL = 0

def create_triplets(x, digit_indices, labels):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    n_classes = len(labels)
    n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1
    pairs = []
    triplets = []
    for j, d in enumerate(labels):
        eq_pairs = list(product(digit_indices[j], digit_indices[j]))
        eq_pairs = filter(lambda x: x[0] < x[1], eq_pairs)
        for pair in eq_pairs:
            doublet = [x[pair[0]], x[pair[1]]]
            for i, d_c in enumerate(labels):
                if d == d_c:
                    continue 
                idx = i % len(labels)
                new_triplets = []
                n_samples = min(NUM_DIFF,len(digit_indices[idx]))
                for diff_idx in digit_indices[idx][:n_samples]:
                    new_triplet = doublet + [x[diff_idx]]
                    new_triplets.append(new_triplet)
                triplets.extend(new_triplets)

    return np.array(triplets)

def create_triplet_idxs(x, digit_indices, labels):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    n_classes = len(labels)
    n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1
    pairs = []
    triplets = []
    for j, d in enumerate(labels):
        eq_pairs = list(product(digit_indices[j], digit_indices[j]))
        eq_pairs = filter(lambda x: x[0] < x[1], eq_pairs)
        for pair in eq_pairs:
            doublet = [pair[0], pair[1]]
            for i in range(len(labels)):
                if i == j:
                    continue
                idx = i % len(labels)
                new_triplets = []
                n_samples = min(NUM_DIFF,len(digit_indices[idx]))
                for diff_idx in digit_indices[idx][:n_samples]:
                    new_triplet = doublet + [diff_idx]
                    new_triplets.append(new_triplet)
                triplets.extend(new_triplets)

    return np.array(triplets)

def create_pairs(x, digit_indices, labels):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    # todo: turn this into indices
    n_classes = len(labels)
    n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1
    pairs = []
    tags = []
    for j, d in enumerate(labels):
        eq_pairs = list(product(digit_indices[j], digit_indices[j]))
        eq_pairs = filter(lambda x: x[0] < x[1], eq_pairs)
        for pair in eq_pairs:

            pairs.append([x[pair[0]], x[pair[1]]])
            tags.append(SAME_LABEL)
        for i in range(j+1, len(labels)):
            diff_pairs = list(product(digit_indices[j], digit_indices[i]))
            diff_pairs = diff_pairs
            for pair in diff_pairs:
                pairs.append([x[pair[0]], x[pair[1]]])
                tags.append(NEG_LABEL)
    return np.array(pairs), np.array(tags)

def create_pair_idxs(x, digit_indices, labels):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    # todo: turn this into indices
    n_classes = len(labels)
    n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1
    pairs = []
    tags = []
    for j, d in enumerate(labels):
        eq_pairs = list(product(digit_indices[j], digit_indices[j]))
        eq_pairs = filter(lambda x: x[0] < x[1], eq_pairs)
        pairs.extend(eq_pairs)
        tags.extend([SAME_LABEL for x in eq_pairs])
        for i in range(j+1, len(labels)):
            diff_pairs = list(product(digit_indices[j], digit_indices[i]))
            pairs.extend(diff_pairs)
            tags.extend([NEG_LABEL for x in diff_pairs])
    return np.array(pairs), np.array(tags)



def next_batch(s,e,inputs):
    end_ind = min(e, len(inputs))
    input1 = inputs[s:end_ind,0]
    input2 = inputs[s:end_ind,1]
    input3 = inputs[s:end_ind,2]
    return input1,input2,input3



def next_batch_from_idx(s,e,inputs,X):
    end_ind = min(e, len(inputs))
    input1 = X[inputs[s:end_ind,0]]
    input2 = X[inputs[s:end_ind,1]]
    input3 = X[inputs[s:end_ind,2]]
    return input1,input2,input3
"""
def plot_filters(plot_row, n_filters, W):
  f, axarr = plt.subplots(1, n_filters)
  for f_num in range(n_filters):
    plot_f = np.squeeze(W[:,:,:,f_num])
    for n in range(plot_row):
      axarr[f_num].plot(plot_f)
      plt.setp([axarr[f_num].get_xticklabels()], visible=False)
      if not f_num == 0:
          plt.setp([axarr[f_num].get_yticklabels()], visible=False)
  f.subplots_adjust(hspace=0)  #No horizontal space between subplots
  f.subplots_adjust(wspace=0)
  plt.show()
"""

def plot_filters(n_filters, W):
  f, axarr = plt.subplots(1, n_filters)
  for f_num in range(n_filters):
    plot_f = np.squeeze(W[:,:,f_num])
    if n_filters > 1:
      axarr[f_num].plot(plot_f)
      plt.setp([axarr[f_num].get_xticklabels()], visible=False)
    else:
      axarr.plot(plot_f)
      plt.setp([axarr.get_xticklabels()], visible=False)
    if not f_num == 0:
        plt.setp([axarr[f_num].get_yticklabels()], visible=False)
  f.subplots_adjust(hspace=0)  #No horizontal space between subplots
  f.subplots_adjust(wspace=0)
  plt.show()

def normalize_rows(X):
    norms = np.sqrt((X*X).sum(axis=1))
    return X / norms[:, np.newaxis]

def dist(x, y):
    dist = tf.reduce_sum(tf.square(tf.subtract(x, y)), 1)
    return dist

def evaluate_test_embedding(train_embedding, tr_y, test_embedding, test_y):
    n_correct = 0.0
    for sample, correct_label in zip(test_embedding, test_y):
        label = classify_sample(sample, train_embedding, tr_y)
        if label == correct_label:
            n_correct += 1.0
    return n_correct/len(test_y)

def evaluate_train_embedding(train_embedding, tr_y):
    n_correct = 0.0
    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(train_embedding))
    for i in range(D.shape[0]):
        row = D[i]
        if tr_y[i] == tr_y[np.argsort(row)[1]]:
            n_correct += 1
    return n_correct/len(tr_y)

def classify_sample(output, train_embedding, tr_y):
    dists = [np.linalg.norm(output-row) for row in train_embedding]
    return tr_y[np.argmin(dists)]


def standardize_ts_lengths(X, n):
    new_X = np.zeros((len(X), X[0].shape[1], n))
    for i, v in enumerate(X):
        new_X[i] = np.append(v.T, np.zeros((v.shape[1], n-v.shape[0])), axis = 1)
    return new_X

def standardize_ts_lengths_1(X, n):
    new_X = np.zeros((len(X), n, X[0].shape[1]))
    for i, v in enumerate(X):
        new_X[i] = np.append(v, np.zeros((n-v.shape[0], v.shape[1])), axis = 0)
    return new_X

def produce_pair_labels(labels):
    match_list = []
    for i in range(len(labels)):
        for j in range(len(labels)):    
            if labels[i] == labels[j]:
                match_list.append(0)
            else:
                match_list.append(1)
    return np.array(match_list)

def rebase_labels(y, labels):
    new_y = [np.argwhere(labels == y_val)[0][0] for y_val in y]
    return np.array(new_y)



# There is a comb function for Python which does 'n choose k'                                                                                            
# only you can't apply it to an array right away                                                                                                         
# So here we vectorize it...                                                                                                                             
def myComb(a,b):
  return comb(a,b,exact=True)

vComb = np.vectorize(myComb)

def get_tp_fp_tn_fn(cooccurrence_matrix):
  tp_plus_fp = vComb(cooccurrence_matrix.sum(0, dtype=int),2).sum()
  tp_plus_fn = vComb(cooccurrence_matrix.sum(1, dtype=int),2).sum()
  tp = vComb(cooccurrence_matrix.astype(int), 2).sum()
  fp = tp_plus_fp - tp
  fn = tp_plus_fn - tp
  tn = comb(cooccurrence_matrix.sum(), 2) - tp - fp - fn

  return [tp, fp, tn, fn]

def eval_clustering(pred_labels, labels):
  labels = np.array([int(x) for x in labels])
  pred_labels = np.array(pred_labels)
  kmean_sort_ind = np.argsort(labels)
  pdb.set_trace()
  pred_labels = pred_labels.labels_[kmean_sort_ind]

  labels = labels[kmean_sort_ind]

  unique_labels = np.unique(labels)
  labels = rebase_labels(labels, unique_labels)
  pred_labels -= np.min(pred_labels)

  co_matrix = np.zeros((len(np.unique(labels)), len(np.unique(pred_labels)) ))
  for i,x in enumerate(pred_labels):
    co_matrix[int(labels[i])][x] += 1
  tp, fp, tn, fn = get_tp_fp_tn_fn(co_matrix)

  nmi = normalized_mutual_info_score(pred_labels, labels)
  print 'Rand Index: ', rand_ind
  return rand_ind
