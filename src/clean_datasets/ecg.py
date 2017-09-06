#!/usr/bin/env python

import os
import numpy as np
from joblib import Memory

from .. import paths
from ..utils import files

_memory = Memory('.', verbose=0)

NORMAL_DIR = os.path.join(paths.ECG, 'normal')
ABNORMAL_DIR = os.path.join(paths.ECG, 'abnormal')


def _all_data_in_dir(dirpath):
    data_files = files.list_files(dirpath, abs_paths=True)
    # print "data files: ", "\n".join(data_files)
    prefixes = [f.split('.')[-2] for f in data_files]
    prefixes = list(set(prefixes))
    # print "prefixes:\n", "\n".join(sorted(prefixes))

    ts_list = []
    for prefix in prefixes:
        data0 = np.genfromtxt(prefix + '.0')
        data1 = np.genfromtxt(prefix + '.1')

        # each is two cols; first col is useless index (literally just
        # 1, 2, 3, ...); combine useful data from each lead into one Mx2 mat
        data1[:, 0] = data0[:, 1]
        ts_list.append(data1)

    # print "initial ts shapes: ", [ts.shape for ts in ts_list[:20]]

    return ts_list


def all_data():
    ts_list_normal = _all_data_in_dir(NORMAL_DIR)
    ts_list_abnormal = _all_data_in_dir(ABNORMAL_DIR)

    ts_list = ts_list_normal + ts_list_abnormal

    y = np.zeros(len(ts_list), dtype=np.int32)
    y[len(ts_list_normal):] = 1

    return ts_list, y

    # y_normal = np.zeros(len(ts_list_normal), dtype=np.int32)
    # y_abnormal = np.zeros(len(ts_list_abnormal), dtype=np.int32)


# ================================================================ main

def main():
    X, y = all_data()

    print "inital ts shapes: ", [ts.shape for ts in X[0:200:20]]
    print "inital labels: ", y[0:200:20]

    print "number of normal heartbeats: ", np.where(y > 0)[0][0]
    print "number of abnormal heartbeats: ", np.sum(y)


if __name__ == '__main__':
    main()
