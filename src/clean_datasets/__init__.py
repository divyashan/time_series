#!/usr/bin env python

import collections
import numpy as np
from sklearn.model_selection import StratifiedKFold


from . import arabic_digits, auslan, ecg, libras, pen_digits
from . import robot_failure, trajectories, vowels, wafer, eeg

import pdb

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
EEG = 'eeg'
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
    EEG: eeg
    }

ALL_DATASETS = sorted(_DSET_TO_MODULE.keys())
TINY_DATASETS = [VOWELS, PEN_DIGITS, ROBOT_FAILURE_LP1, ROBOT_FAILURE_LP2,
                 ROBOT_FAILURE_LP3, ROBOT_FAILURE_LP4, ROBOT_FAILURE_LP5]
NOT_TINY_DATASETS = [ECG, LIBRAS, AUSLAN, WAFER, TRAJECTORIES, ARABIC_DIGITS]

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
        #print "train idxs: ", train_idxs
        #print "test idxs: ", test_idxs
        splits.append(CVSplit(X_train=[X[idx] for idx in train_idxs],
                      y_train=y[train_idxs],
                      X_test=[X[idx] for idx in test_idxs],
                      y_test=y[test_idxs]))

    return splits


def _check_dset_valid(dset_name):
    if dset_name not in ALL_DATASETS:
        raise ValueError("Invalid dataset name '{}'; valid names"
                         " include:\n{}".format(ALL_DATASETS))

def _pad_to_same_length(ts_list):
    max_len = np.max([len(ts) for ts in ts_list])
    ret = []
    for ts in ts_list:
        length = len(ts)
        pad_length = max_len - length
        if pad_length <= 0:
            new_ts = ts
        else:
            new_ts = np.zeros((max_len, ts.shape[1]))
            new_ts[:len(ts), :] = ts

        ret.append(new_ts)
    return ret


def _resample_to_same_length(ts_list):
    max_len = int(np.max([len(ts) for ts in ts_list]))
    idxs = np.arange(max_len)
    ret = []
    for ts in ts_list:
        take_idxs = (idxs * float(len(ts)) / max_len).astype(np.int32)
        ret.append(ts[take_idxs])
    return ret


def cv_splits_for_dataset(dset_name, n_folds=5, length_adjust=None):
    _check_dset_valid(dset_name)

    mod = _DSET_TO_MODULE[dset_name]

    try:
        X_train, y_train = mod.train_data()
        X_test, y_test = mod.test_data()
        print "Training/Testing split exists"
        if length_adjust == 'pad':
            X_train = _pad_to_same_length(X_train)
            X_test = _pad_to_same_length(X_test)
        elif length_adjust == 'upsample':
            X_train = _resample_to_same_length(X_train)
            X_test = _resample_to_same_length(X_test)

        return [CVSplit(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)]

    except AttributeError:
        X, y = mod.all_data()
        if length_adjust == 'pad':
            X = _pad_to_same_length(X)
        elif length_adjust == 'upsample':
            X = _resample_to_same_length(X)

        return cv_splits(X, y, n_folds=n_folds)


def dset_as_mat(dset_name, length_adjust='upsample'):
    _check_dset_valid(dset_name)
    mod = _DSET_TO_MODULE[dset_name]

    try:
        X_train, y_train = mod.train_data()
        X_test, y_test = mod.test_data()

        X = X_train + X_test
        y = np.hstack((y_train, y_test))

    except AttributeError:
        X, y = mod.all_data()

    if length_adjust == 'pad':
        X = _pad_to_same_length(X)
    elif length_adjust == 'upsample':
        X = _resample_to_same_length(X)

    # print "ts lengths in X: ", [len(ts) for ts in X]
    flat_X = [ts.ravel() for ts in X]
    return np.vstack(flat_X), y


# zero-pads time series to be all be the length of the longest one in the
# dataset, then flattens to 1D by concatenating data from each variable;
# finally, prepends label to each 1D time series; end result is a dataset
# that, when dumped to a csv file, is in the same format as the UCR archive
# datasets
#
# Also note that we concatenate the training and test sets into one matrix,
# because that's what we need right now (to assess clustering)
def dset_to_ucr_mat(dset_name):
    _check_dset_valid(dset_name)
    mod = _DSET_TO_MODULE[dset_name]
    try:
        X_train, y_train = mod.train_data()
        X_test, y_test = mod.test_data()

        X = X_train + X_test
        y = np.hstack((y_train, y_test))

    except AttributeError:
        X, y = mod.all_data()

    max_len = np.max([len(ts) for ts in X])
    X_new = []
    for ts in X:
        length = len(ts)
        pad_length = max_len - length
        if pad_length <= 0:
            new_ts = ts
        else:
            new_ts = np.zeros((max_len, ts.shape[1]))
            new_ts[:len(ts), :] = ts

        flattened = np.asfortranarray(new_ts).T.ravel()
        assert np.array_equal(flattened[:length], ts[:length, 0])

        # print "ts old shape, intermediate shape, new shape: ", \
        #     ts.shape, new_ts.shape, flattened.shape

        X_new.append(flattened)

    X = np.vstack(X_new)

    shape = np.array(X.shape)
    shape[1] += 1
    out = np.empty(shape)
    out[:, 0] = y
    out[:, 1:] = X

    return out
