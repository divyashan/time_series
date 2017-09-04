import numpy as np
import os
from numpy import r_

import pdb
UCR_DATASETS_DIR = "../"

# ================================================================
# Public
# ================================================================

def getAllUCRDatasets():
    for dataDir in getAllUCRDatasetDirs():
        yield UCRDataset(dataDir)


class UCRDataset(object):

    def __init__(self, datasetDir):
        self.Xtrain, self.Ytrain = readUCRTrainData(datasetDir)
        p = np.random.permutation(len(self.Xtrain))
        self.Xtrain = self.Xtrain[p]
        self.Ytrain = self.Ytrain[p]

        val_idx = int(len(self.Xtrain)*.9)
        self.Xval = self.Xtrain[val_idx:]
        self.Yval = self.Ytrain[val_idx:]
        self.Xtrain = self.Xtrain[:val_idx]
        self.Ytrain = self.Ytrain[:val_idx]

        self.Xtest, self.Ytest = readUCRTestData(datasetDir)
        self.name = nameFromDir(datasetDir)
        self.sorted_Xtrain, self.sorted_Ytrain = self.sort()
        self.intervals = self.get_class_intervals()

    def sort(self):
        # Returns new D matrix 
        # and class intervals
        sorted_ind = np.argsort(self.Ytrain)
        sorted_Xtrain = self.Xtrain[sorted_ind,:]
        sorted_Ytrain = self.Ytrain[sorted_ind]
        return sorted_Xtrain, sorted_Ytrain

    def get_class_intervals(self):
        # Returns dictionary of index intervals for each class 
        # Maps label --> (start_ind, end_ind)
        interval_map = dict()
        labels = np.unique(self.sorted_Ytrain)
        indices = [np.searchsorted(self.sorted_Ytrain, l) for l in labels]
        indices.append(len(self.sorted_Ytrain))
        for i,l in enumerate(labels):
            interval_map[l] = (indices[i], indices[i+1])
        return interval_map

    def get_class(self, label):
        # Return all training for query label
        i, j = self.intervals[label][0], self.intervals[label][1]
        return self.sorted_Xtrain[i:j,:]


# ================================================================
# Private
# ================================================================


def readDataFile(path):
    if '50wordsx' in path:
        pdb.set_trace()

    D = np.genfromtxt(path, delimiter=",")
    labels = D[:,0]
    X = D[:,1:]
    return (X, labels)


def nameFromDir(datasetDir):
    return os.path.basename(datasetDir)


def readUCRDataInDir(datasetDir, train):
    datasetName = nameFromDir(datasetDir)
    if train:
        fileName = datasetName + "_TRAIN.txt"
    else:
        fileName = datasetName + "_TEST.txt"
    filePath = os.path.join(datasetDir,fileName)
    return readDataFile(filePath)


def readUCRTrainData(datasetDir):
    return readUCRDataInDir(datasetDir, train=True)


def readUCRTestData(datasetDir):
    return readUCRDataInDir(datasetDir, train=False)


# combines train and test data
def readAllUCRData(ucrDatasetDir):
    X_train, Y_train = readUCRTrainData(ucrDatasetDir)
    X_test, Y_test = readUCRTestData(ucrDatasetDir)
    X = r_[X_train, X_test]
    Y = r_[Y_train, Y_test]
    return (X,Y)


def getAllUCRDatasetDirs():
    datasetsPath = os.path.expanduser(UCR_DATASETS_DIR)
    files = os.listdir(datasetsPath)
    for i in range(len(files)):
        files[i] = os.path.join(datasetsPath, files[i])
    dirs = filter(os.path.isdir, files)
    return dirs

if __name__ == '__main__':
    import sequence as seq

    # print out a table of basic stats for each dataset to verify
    # that everything is working
    nameLen = 22
    print("%s\tTrain\tTest\tLength\tClasses" % (" " * nameLen))
    for i, datasetDir in enumerate(getAllUCRDatasetDirs()):
        Xtrain, _ = readUCRTrainData(datasetDir)
        Xtest, Ytest = readUCRTestData(datasetDir)
        print('%22s:\t%d\t%d\t%d\t%d' % (nameFromDir(datasetDir),
            Xtrain.shape[0], Xtest.shape[0], Xtrain.shape[1],
            len(seq.uniqueElements(Ytest))))
