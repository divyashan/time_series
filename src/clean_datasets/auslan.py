#!/usr/bin/env python

import numpy as np
from joblib import Memory

from .. import paths
from ..utils import files

_memory = Memory('.', verbose=0)


@_memory.cache
def _all_data_files():
    subsubdirs = []
    for subdir in files.list_subdirs(paths.AUSLAN, abs_paths=True):
        subsubdirs += files.list_subdirs(subdir, abs_paths=True)
    # print "subsubdirs:", "\n".join(subsubdirs)

    all_data_files = []
    for d in subsubdirs:
        all_data_files += files.list_files(d, abs_paths=True, endswith='.tsd')

    return all_data_files


@_memory.cache
def all_X():
    return [np.genfromtxt(path, delimiter='\t') for path in _all_data_files()]


@_memory.cache
def _all_raw_labels():
    return [path.split('/')[-1].split('-')[0].strip('_') for path in _all_data_files()]


@_memory.cache
def all_labels():
    all_labels = _all_raw_labels()

    uniq_labels = np.unique(all_labels)
    print "Auslan uniq labels:\n", ", ".join(uniq_labels)

    lbl2yval = {lbl: i for i, lbl in enumerate(uniq_labels)}
    all_labels = [lbl2yval[lbl] for lbl in all_labels]
    return np.array(all_labels, dtype=np.int32)


def all_data():
    return all_X(), all_labels()


# ================================================================ main

def main():
    X, y = all_data()

    print "inital ts shapes: ", [ts.shape for ts in X[:20]]
    print "inital labels: ", y[:20]


if __name__ == '__main__':
    main()
