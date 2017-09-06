#!/usr/bin/env python

import os
import numpy as np
from joblib import Memory

from .. import paths

_memory = Memory('.', verbose=0)

TRAIN_FILE = os.path.join(paths.PEN_DIGITS, 'pendigits.tra')
TEST_FILE = os.path.join(paths.PEN_DIGITS, 'pendigits.tes.txt')


@_memory.cache
def _read_data_file(path, true_nrows=-1):
    mat = np.genfromtxt(path, delimiter=',')
    X = mat[:, :-1]
    y = mat[:, -1]

    # # maybe first half and second half are x and y ?
    # X = X.reshape(true_nrows * 2, 8)
    # x_rows = X[::2]
    # y_rows = X[1::2]
    # ts_list = [np.vstack((xvals, yvals)).T for xvals, yvals in zip(x_rows, y_rows)]
    # ts_list = [np.ascontiguousarray(ts) for ts in ts_list]

    # each row has (x, y) value interleaved
    X = X.reshape(true_nrows, 8, 2)  # sanity check dims
    ts_list = [np.ascontiguousarray(X[i]) for i in range(len(X))]

    # make sure we munged that properly; we're trying to take alternating
    # points from each row of the original X and turn them into a 2d ts; ie,
    # they interleaved the two variables to get each row to be 1d
    assert ts_list[0][0, 0] == mat[0, 0]
    assert ts_list[0][0, 1] == mat[0, 1]
    assert ts_list[0][1, 0] == mat[0, 2]
    assert ts_list[37][0, 1] == mat[37, 1]

    return ts_list, y.astype(np.int32)


def train_data():
    return _read_data_file(TRAIN_FILE, true_nrows=7494)


def test_data():
    return _read_data_file(TEST_FILE, true_nrows=3498)


# ================================================================ main

def main():
    X, y = train_data()
    print "inital train ts shapes: ", [ts.shape for ts in X[0:200:20]]
    print "inital train labels: ", y[0:200:20]

    X, y = test_data()
    print "inital test ts shapes: ", [ts.shape for ts in X[0:200:20]]
    print "inital test labels: ", y[0:200:20]

    # ya, these basically look like digits, accounting for the fact
    # that it's only 8 points and the raw values are scaled in a somewhat
    # inconsistent way (making the plot tall and skinny helps)
    import matplotlib.pyplot as plt
    which_examples = np.array([0, 1, 2, 3]) + 10
    for which_example in which_examples:
        plt.figure(figsize=(2, 4))
        ts = X[which_example]
        plt.scatter(ts[:, 0], ts[:, 1])
        plt.title(str(y[which_example]))
    plt.show()


if __name__ == '__main__':
    main()
