#!/usr/bin env python

import collections
import numpy as np
from sklearn.model_selection import StratifiedKFold


from . import arabic_digits, auslan, ecg, libras, pen_digits
from . import robot_failure, trajectories, vowels, wafer


# UCI archive datasets
ARABIC_DIGITS = 'arabic_digits'
AUSLAN = 'auslan'
TRAJECTORIES = 'trajectories'
VOWELS = 'vowels'
LIBRAS = 'libras'
PEN_DIGITS = 'pen_digits'
ROBOT_FAILURE_LP1 = 'robot_failure_lp1'
ROBOT_FAILURE_LP2 = 'robot_failure_lp2'
ROBOT_FAILURE_LP3 = 'robot_failure_lp3'
ROBOT_FAILURE_LP4 = 'robot_failure_lp4'
ROBOT_FAILURE_LP5 = 'robot_failure_lp5'
# datasets from http://www.cs.cmu.edu/~bobski/data/data.html
ECG = 'ecg'
WAFER = 'wafer'

ROBOT_FAILURE_DATASETS = [ROBOT_FAILURE_LP1, ROBOT_FAILURE_LP2,
                          ROBOT_FAILURE_LP3, ROBOT_FAILURE_LP4,
                          ROBOT_FAILURE_LP5]

_DSET_TO_MODULE = {
    ARABIC_DIGITS: arabic_digits,
    AUSLAN: auslan,
    TRAJECTORIES: trajectories,
    VOWELS: vowels,
    LIBRAS: libras,
    PEN_DIGITS: pen_digits,
    ROBOT_FAILURE_LP1: robot_failure.lp1,
    ROBOT_FAILURE_LP2: robot_failure.lp2,
    ROBOT_FAILURE_LP3: robot_failure.lp3,
    ROBOT_FAILURE_LP4: robot_failure.lp4,
    ROBOT_FAILURE_LP5: robot_failure.lp5,
    ECG: ecg,
    WAFER: wafer,
    }

ALL_DATASETS = sorted(_DSET_TO_MODULE.keys())
NOT_TINY_DATASETS = [ARABIC_DIGITS, AUSLAN, TRAJECTORIES, LIBRAS, ECG, WAFER]

_SEED = 123  # random seed for cross validation

CVSplit = collections.namedtuple(
    'CVSplit', 'X_train y_train X_test y_test'.split())


def cv_splits(X, y, n_folds=5):
    np.random.seed(_SEED)
    skf = StratifiedKFold(n_splits=n_folds)

    # to be certain it ignores X, since it won't expect a list, let alone
    # a list of variable-length 2D arrays; the docs say it ignores X, but
    # this way we can be certain
    X_ignore = np.zeros(len(X))

    splits = []
    for train_idxs, test_idxs in skf.split(X_ignore, y):
        # print "train idxs: ", train_idxs
        # print "test idxs: ", test_idxs
        splits.append(CVSplit(X_train=[X[idx] for idx in train_idxs],
                      y_train=y[train_idxs],
                      X_test=[X[idx] for idx in test_idxs],
                      y_test=y[test_idxs]))

    return splits


def cv_splits_for_dataset(dset_name, n_folds=5):
    if dset_name not in ALL_DATASETS:
        raise ValueError("Invalid dataset name '{}'; valid names"
                         " include:\n{}".format(ALL_DATASETS))

    mod = _DSET_TO_MODULE[dset_name]

    try:
        X_train, y_train = mod.train_data()
        X_test, y_test = mod.test_data()
        return [CVSplit(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)]
    except AttributeError:
        X, y = mod.all_data()
        return cv_splits(X, y, n_folds=n_folds)
