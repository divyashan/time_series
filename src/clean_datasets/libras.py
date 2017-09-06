#!/usr/bin/env python

import os
import numpy as np
from joblib import Memory

from .. import paths

_memory = Memory('.', verbose=0)

DATA_FILE = os.path.join(paths.LIBRAS, 'movement_libras.data.txt')


def all_data():
    mat = np.genfromtxt(DATA_FILE, delimiter=',')
    X = mat[:, :-1]
    y = mat[:, -1] - 1  # sub 1 so class 0 is the min
    X = X.reshape(360, 45, 2)  # sanity check dims

    # abscissas = X[:, :, 0]
    # ordinates = X[:, :, 1]

    ts_list = [np.ascontiguousarray(X[i]) for i in range(len(X))]

    # make sure we munged that properly; we're trying to take alternating
    # points from each row of the original X and turn them into a 2d ts; i.e.,
    # they interleaved the two variables to get each row to be 1d
    assert ts_list[0][0, 0] == mat[0, 0]
    assert ts_list[0][0, 1] == mat[0, 1]
    assert ts_list[0][1, 0] == mat[0, 2]
    assert ts_list[37][0, 1] == mat[37, 1]

    return ts_list, y


# ================================================================ main

def main():

    X, y = all_data()

    print "inital ts shapes: ", [ts.shape for ts in X[0:200:20]]
    print "inital labels: ", y[0:200:20]


if __name__ == '__main__':
    main()
