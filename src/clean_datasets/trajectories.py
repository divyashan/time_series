#!/usr/bin/env python

import os
import numpy as np
import scipy.io as sio
from joblib import Memory

from .. import paths
from ..utils import files

_memory = Memory('.', verbose=0)


@_memory.cache
def all_data():
    path = os.path.join(paths.TRAJECTORIES, 'mixoutALL_shifted.mat')
    # mat = sio.loadmat(path, struct_as_record=False)
    mat = sio.loadmat(path)

    consts = mat['consts']
    data = mat['mixout']

    # print "------------------------ mat keys and other basic info:"
    # print "whosmat output: ", sio.whosmat(path)
    # print sorted(mat.keys())
    # print mat['__header__']
    # print mat['__version__']
    # print mat['__globals__']
    # print "consts dtypes:"
    # print consts.dtype
    # # print consts.keys()
    # # print consts['key'][0, 0]
    # # print "charlabels: ", consts['charlabels'][0, 0].shape
    # print "mixout dtypes:"

    # print data.dtype
    # print data[0, 0].shape
    # print '------------------------'

    labels = consts['charlabels'][0, 0].ravel()
    # print "labels.shape", labels.shape
    # print "initial labels: ", labels[0:1000:50]

    ts_list = [np.ascontiguousarray(data[0, i].T) for i in range(2858)]
    # print "initial ts shapes: ", [ts.shape for ts in ts_list[:20]]

    return ts_list, labels

    # print mat.keys()
    # print mat['consts']

    # return

    # data = mat['mixout']
    # # print type(data)
    # # print mat['consts'].shape
    # print data.shape
    # # print data[0].shape


# ================================================================ main

def main():
    X, y = all_data()

    print "inital ts shapes: ", [ts.shape for ts in X[:20]]
    print "inital labels: ", y[:20]


if __name__ == '__main__':
    main()
