import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
from readUcr import UCRDataset


def plot_time_series(vals, indices=None):
    """
    Plots time series.
    """
    if indices==None:
        # Evenly sampled
        plt.plot(vals)
    else:
        # Unevenly sampled
        plt.plot(vals, indices)
    plt.show()


def plot_class(X_c):
    """ 
    X_c     Matrix of class-specific time series 

    Plots all time series from a given class.       
    """
    Helper function 
    for row in X_c:
        plt.plot(row, 'b-')
    plt.show()

def normalize_rows(X):
    """ 
    Normalizes matrix X along its last dimension 
    """
    return X / np.linalg.norm(X, axis=-1)[:, np.newaxis]

def plot_all_time_series(name):
    data = UCRDataset("../ucr_data/" + name)
    norm_data = normalize_rows(data.Xtrain)
    cmap = plt.cm.get_cmap('hsv', len(data.intervals.keys()))
    idx = data.intervals.keys()
    colors = "bgrcmykw"
    for row, label in zip(norm_data, data.Ytrain):
        plt.plot(row, c=colors[idx.index(label)])
    plt.title(name)
    plt.show()

def compare_testing_training(input_f):
    """ 
    input_f     input_f is a three-column tsv with data formatted 
                as ['dataset_name', 'training_acc', 'testing_acc']

    Plots the training versus testing accuracy across a set of datasets
    """
    results = [x.split('\t') for x in open(input_f, 'r').readlines()]
    training = [float(x[1]) for x in results]
    testing = [float(x[2][:-1]) for x in results]
    datasets = [x[0] for x in results]
    
    plt.plot(training, 'bo', ms=10, label='Training')
    plt.plot(testing, 'ro', ms=10, label='Testing')
    plt.xlabel('Datasets')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.ylim([0, 100])
    
    plt.xticks(range(0,len(datasets)+1), datasets, rotation='vertical')
    plt.subplots_adjust(bottom=0.3)
    plt.title("Training v. Testing")
    plt.show()

def get_dataset_names(file):
    """ 
    file        one path to a results file. each file is tsv where the first column 
                is the dataset name and the second is the accuracy 

    Returns list of datasets described by a results file 
    """

    f = open(file, 'r')
    datasets = [l.split('\t')[0] for l in f.readlines()]
    return datasets

def get_common_tested_datasets(files):
    """
    files       list of paths to files containing results. each file is tsv where the first column 
                is the dataset name and the second is the accuracy 

    Returns list of datasets commonly tested across all results files 
    """
    dataset_lists = [set(get_dataset_names(file)) for file in files]
    common_datasets = dataset_lists[0]
    for i in range(1,len(files)):
        common_datasets = common_datasets.intersection(dataset_lists[i])
    return list(common_datasets)




def compare_results(files, names):
    """ 
    files       list of paths to files containing results. each file must be tsv where the first column 
                is the dataset name and the second is the accuracy. 
    names       corresponding method names for each file 
    """
    fig, ax = plt.subplots()
    bar_width = .2
    opacity = .8

    datasets = get_common_tested_datasets(files)
    n_groups = len(datasets)
    index = np.arange(n_groups)
    model_vals_arr = []
    colors = ['b', 'g', 'y', 'r']
    idx = 0

    for file,name in zip(files,names):

        results_1 = [x.split('\t') for x in open(file, 'r').readlines()]
        #if name == 'Triplet_MLP':
        #    results_1 = [x.split(' ') for x in open(file, 'r').readlines()]
        model1 = {result[0]: result[1:] for result in results_1}
        means = {}
        stds = {}
        for key in model1:
            model1[key] = np.array([float(val[:-1]) for val in model1[key]])
            means[key] = np.mean(model1[key])
            stds[key] = np.std(model1[key])

        model_vals = [means[k] for k in datasets]

        model_err = [stds[k] for k in datasets]
        plt.bar(index+idx*bar_width, model_vals, bar_width,
                     alpha=opacity,
                     color=colors[idx],
                     label=name, yerr=model_err, error_kw=dict(ecolor='black', lw=3, capzie=7, capthick=2))
        idx += 1




    #plt.plot(model1_vals, label=r_1_name)
    #plt.plot(model2_vals, label=r_2_name)
    plt.xlabel('Datasets')
    plt.ylabel('Accuracy')
    plt.title(' v. '.join(names))
    plt.legend(loc='upper right')

    plt.xticks(index+bar_width, datasets, rotation='vertical')
    plt.tight_layout()

    plt.show()


def get_class_avg(D):
    """
    D       Time series dataset; first column contains labels

    Returns average time series member of each class 
    """
    new_ind = np.argsort(D[:,0])
    sorted_D = D[new_ind,:]
    labels = sorted_D[:,0]
    pdb.set_trace()
    class_reps = []
    curr_class = sorted_D[0,0]
    curr = 0
    while curr != len(D):
        next = np.where(labels>sorted_D[curr,0])[0]
        if len(next) == 0:  
            next = np.append(next, len(D))

        class_rep = np.sum(sorted_D[curr:next[0],:], axis=0)/float(next[0]-curr)
        class_reps.append(class_rep)
        curr = next[0]

    
    return class_reps

def time_vars(D):
    return np.var(D, axis=0)
#compare_testing_training(sys.argv[1])

def plot_pairwise_distances(train_embedding, y_train, test_embedding, y_test, mode="euclidean"):
    """ 
    train_embedding   N_trxD matrix of the D-dimensional embeddings of N_tr training points 
    y_train           Nx1 matrix of class labels for N training points 
    test_embedding    N_texD matrix of the D-dimensional embeddings of N_te testing points 
    y_test            N_tex1 matrix of class labels of N_te training points 
    mode              If "euclidean", calculates euclidean distance. otherwise uses DTW distance. 

    Produces bar plot visualizing the distributions of distances between same-class object pairs
    and different-class object pairs. Object pairs are generated from all-pairs matchings between 
    testing and training objects.
    """
    n_samples = 50
    if mode == "euclidean":
      print "Calculating euclidean distances"
      e_dist, e_labels = compute_distances_to_points(train_embedding[0:-1], y_train[0:n_samples], test_embedding, y_test)
    else:
      print "Calculating DTW distances"
      e_dist, e_labels = compute_dtw_distances_to_points(train_embedding[0:-1], y_train[0:n_samples], test_embedding, y_test)
    sort_idx = np.argsort(e_dist)
    sorted_e_dist = e_dist[sort_idx]
    sorted_e_labels = e_labels[sort_idx]
    results = []
    start_idx = np.where(sorted_e_dist > 0)[0][0]
    for idx in range(min(len(sorted_e_dist), 600)):
      result = dict()
      result['x'] = idx
      result['y'] = sorted_e_dist[idx]
      result['label'] = sorted_e_labels[idx]
      results.append(result)
    results_df = pd.DataFrame(results)

    palette_dir = {0: 'blue', 1: 'orange'}
    ax = seaborn.barplot(x="x", y="y", hue="label", data=results_df)

    plt.show()




def plot_distances(train_embedding, y_train, mode="euclidean"):
  """ 
  train_embedding   N_trxD matrix of the D-dimensional embeddings of N_tr training points 
  y_train           Nx1 matrix of class labels for N training points 
  mode              If "euclidean, calculates euclidean distance. otherwise uses DTW distance

  Produces bar plot visualizing the distributions of distance between same-class object pairs
  and different-class object pairs. Object pairs are generated from all-pairs matching
  within training set. Visualizes the 200000 smallest distances.
  """
  marker_size = 10
  if mode == 'euclidean':
    e_dist, e_labels = compute_pairwise_distances(train_embedding, y_train)
    print len(e_dist)
    sort_idx = np.argsort(e_dist)
    sorted_e_dist = e_dist[sort_idx]
    sorted_e_labels = e_labels[sort_idx]
    plt.scatter(np.arange(sorted_e_labels), sorted_e_dist[:-200000], marker_size, sorted_e_labels[:-200000])
  else:
    dtw_dist, dtw_labels = compute_dtw_pairwise_distances(train_embedding, y_train)
    print "Calculated all dtw distances"
    sort_idx = np.argsort(dtw_dist)
    sorted_dtw_dist = dtw_dist[sort_idx]
    sorted_dtw_labels = dtw_labels[sort_idx]
    plt.scatter(sorted_e_labels, sorted_dtw_dist[:-200000], marker_size, sorted_dtw_labels[:-200000])
  plt.show()



