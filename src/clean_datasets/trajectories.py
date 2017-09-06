#!/usr/bin/env python

import os
import numpy as np
import scipy.io as sio
from joblib import Memory

from .. import paths
from ..utils import files

_memory = Memory('.', verbose=1)


@_memory.cache
def all_data():
    path = os.path.join(paths.TRAJECTORIES, 'mixoutALL_shifted.mat')
    # mat = sio.loadmat(path, struct_as_record=False)
    mat = sio.loadmat(path)

    print "------------------------ mat keys and other basic info:"
    print "whosmat output: ", sio.whosmat(path)
    print sorted(mat.keys())
    print mat['__header__']
    print mat['__version__']
    print mat['__globals__']
    print "consts dtypes:"
    consts = mat['consts']
    print consts.dtype
    # print consts.keys()
    # print consts['key'][0, 0]
    # print "charlabels: ", consts['charlabels'][0, 0].shape
    print "mixout dtypes:"
    data = mat['mixout']
    print data.dtype
    print data[0, 0].shape
    print '------------------------'

    labels = consts['charlabels'][0, 0].ravel()
    # print "labels.shape", labels.shape
    print "initial labels: ", labels[0:1000:50]

    ts_list = [data[0, i] for i in range(2858)]
    print "initial ts shapes: ", [ts.shape for ts in ts_list[:20]]

    return ts_list, labels

    # print mat.keys()
    # print mat['consts']

    # return

    # data = mat['mixout']
    # # print type(data)
    # # print mat['consts'].shape
    # print data.shape
    # # print data[0].shape


def all_X():
    return all_data()[0]


def all_labels():
    return all_data()[1]


# ================================================================ main

def main():
    y = all_labels()
    X = all_X()

    print "inital ts shapes: ", [ts.shape for ts in X[:20]]
    print "inital labels: ", y[:20]


if __name__ == '__main__':
    main()

