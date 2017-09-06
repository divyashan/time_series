#!/usr/bin/env python

import os
import numpy as np
from joblib import Memory

from .. import paths
from ..utils import files

_memory = Memory('.', verbose=0)

NORMAL_DIR = os.path.join(paths.WAFER, 'normal')
ABNORMAL_DIR = os.path.join(paths.WAFER, 'abnormal')


@_memory.cache
def _all_data_in_dir(dirpath):
    data_files = files.list_files(dirpath, abs_paths=True)
    # print "data files: ", "\n".join(data_files)
    prefixes = [f.split('.')[-2] for f in data_files]
    prefixes = list(set(prefixes))

    # print "prefixes:\n", "\n".join(sorted(prefixes))
    # return

    ts_list = []
    exts = ['.6', '.7', '.8', '.11', '.12', '.15']
    for prefix in prefixes:
        data_mats = [np.genfromtxt(prefix + ext) for ext in exts]
        N = len(data_mats[0])
        ts = np.empty((N, len(exts)), dtype=np.float32)
        for i, mat in enumerate(data_mats):
            ts[:, i] = mat[:, 1]  # col 0 is index, which is completely useless

        ts_list.append(ts)

    return ts_list


def all_data():
    ts_list_normal = _all_data_in_dir(NORMAL_DIR)
    ts_list_abnormal = _all_data_in_dir(ABNORMAL_DIR)

    ts_list = ts_list_normal + ts_list_abnormal

    y = np.zeros(len(ts_list), dtype=np.int32)
    y[len(ts_list_normal):] = 1

    return ts_list, y


# ================================================================ main

def main():
    X, y = all_data()

    print "number of time series, number of labels: ", len(X), len(y)
    print "some ts shapes: ", [ts.shape for ts in X[::100]]
    print "some labels: ", y[::100]

    print "number of normal wafers: ", np.where(y > 0)[0][0]
    print "number of abnormal wafers: ", np.sum(y)


if __name__ == '__main__':
    main()
