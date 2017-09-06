#!/usr/bin/env python

import os
import numpy as np
from joblib import Memory

from .. import paths
# from ..utils import files

_memory = Memory('.', verbose=0)


# @_memory.cache
def _all_data_with_name(name, expected_count=None):
    path = os.path.join(paths.ROBOT_FAILURE, name + '.data.txt')
    # print "path: ", path
    with open(path, 'r') as f:
        text = f.read()

    blocks = text.split('\n\n\n')
    labels = [b[:b.find('\n')] for b in blocks]
    examples_strs = [b[b.find('\n'):] for b in blocks]
    # print blocks[:3]
    # print labels[:20]

    # extract labels and convert them to numbers
    uniq_labels = sorted(list(set(labels)))
    lbl2yval = {lbl: i for i, lbl in enumerate(uniq_labels)}
    y = np.array([lbl2yval[lbl] for lbl in labels], dtype=np.int32)

    # extract time series
    if expected_count:
        assert len(examples_strs) == expected_count
    ts_list = [np.fromstring(s, sep=' ').reshape(-1, 6) for s in examples_strs]

    for ts in ts_list:
        assert ts.shape == (15, 6)  # check dims

    return ts_list, y


class lp1:
    @staticmethod
    def all_data():
        return _all_data_with_name(name='lp1', expected_count=88)


class lp2:
    @staticmethod
    def all_data():
        return _all_data_with_name(name='lp2', expected_count=47)


class lp3:
    @staticmethod
    def all_data():
        return _all_data_with_name(name='lp3', expected_count=47)


class lp4:
    @staticmethod
    def all_data():
        return _all_data_with_name(name='lp4', expected_count=117)


class lp5:
    @staticmethod
    def all_data():
        return _all_data_with_name(name='lp5', expected_count=164)


# ================================================================ main

def main():
    for clz in [lp1, lp2, lp3, lp4, lp5]:
        X, y = clz.all_data()
        print "inital ts shapes: ", [ts.shape for ts in X[0:100:10]]
        print "inital labels: ", y[0:100:10]


if __name__ == '__main__':
    main()
