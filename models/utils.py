import numpy as np
import tensorflow as tf
from itertools import product
import scipy

NUM_DIFF=30
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
            for i in range(j+1,len(labels)):
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
            for i in range(j+1,len(labels)):
                idx = i % len(labels)
                new_triplets = []
                n_samples = min(NUM_DIFF,len(digit_indices[idx]))
                for diff_idx in digit_indices[idx][:n_samples]:
                    new_triplet = doublet + [diff_idx]
                    new_triplets.append(new_triplet)
                triplets.extend(new_triplets)

    return np.array(triplets)



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

def produce_pair_labels(labels):
    match_list = []
    for i in range(len(labels)):
        for j in range(len(labels)):    
            if labels[i] == labels[j]:
                match_list.append(0)
            else:
                match_list.append(1)
    return np.array(match_list)
