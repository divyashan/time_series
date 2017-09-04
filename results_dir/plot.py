import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
from readUcr import UCRDataset


def plot_time_series(vals, indices=None):
    if indices==None:
        # Evenly sampled
        plt.plot(vals)
    else:
        # Unevenly sampled
        plt.plot(vals, indices)
    plt.show()


def plot_class(X_c):
    for row in X_c:
        plt.plot(row, 'b-')
    plt.show()

def normalize_rows(X):
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
    f = open(file, 'r')
    datasets = [l.split('\t')[0] for l in f.readlines()]
    return datasets

def get_common_tested_datasets(files):
    dataset_lists = [set(get_dataset_names(file)) for file in files]
    common_datasets = dataset_lists[0]
    for i in range(1,len(files)):
        common_datasets = common_datasets.intersection(dataset_lists[i])
    return list(common_datasets)




def compare_results(files, names):

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

if (len(sys.argv)-1) % 2 != 0:
    print 'u goose this is an odd number of arguments'
else:
    n_args = len(sys.argv)-1
    compare_results(sys.argv[1:1+n_args/2], sys.argv[1+n_args/2:])
#dataset = "../ucr_data/" +  sys.argv[1] 
#pdb.set_trace()
#plot_all_time_series(sys.argv[1])
