#!/usr/bin/env python

from __future__ import division
# from __future__ import print_function

# import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import time
from joblib import Memory

from src import clean_datasets as ds
from src import dtw
# from src.utils import files

_memory = Memory('.', verbose=1)

SAVE_DIR = 'src/final-results'
DTW_I_SAVE_PATH = os.path.join(SAVE_DIR, 'dtwi-accuracies.csv')


def _upsample(ts, new_len):
    if len(ts.shape) == 1:  # each col must be one variable
        ts = ts.reshape(-1, 1)

    orig_len = ts.shape[0]
    assert new_len >= orig_len

    out = np.empty((new_len, ts.shape[1]), dtype=ts.dtype)
    ratio = orig_len / float(new_len)

    out_idxs = np.arange(new_len)
    in_idxs = (out_idxs * ratio).astype(np.int32)
    out[out_idxs, :] = ts[in_idxs, :]

    return out


def dtwi_1nn_classify(X_list, y_train, q_list, warp_frac=.1):
    # make time series all the same length by resampling; also, make
    # them doubles because our c impl currently only supports ints and doubles
    lengths = [len(ts) for ts in X_list]
    lengths += [len(ts) for ts in q_list]
    max_len = np.max(lengths)
    X_list = [_upsample(ts, max_len).astype(np.float64) for ts in X_list]
    q_list = [_upsample(ts, max_len).astype(np.float64) for ts in q_list]

    yhat = np.zeros(len(q_list), dtype=np.int32)
    r = max(1, int(warp_frac * max_len))

    for i, q in enumerate(q_list):
        d_best = dtw.dtw_i(X_list[0], q, r)
        idx_best = 0
        for j, X in enumerate(X_list[1:]):
            d = dtw.dtw_i(X, q, r, d_best=d_best)
            if d < d_best:
                d_best = d
                idx_best = j + 1
        yhat[i] = y_train[idx_best]

    return yhat


def dtwi_1nn_accuracy(dset_name, splits):
    num_correct = 0.
    total_examples = 0.

    print "------------------------ {}".format(dset_name)
    print "size of data in each split: "
    print "len(X_train)", len(splits[0].X_train)
    print "len(y_train)", len(splits[0].y_train)
    print "len(X_test)", len(splits[0].X_test)
    print "len(y_test)", len(splits[0].y_test)

    for i, split in enumerate(splits):
        t0 = time.clock()
        print "running split {}...".format(i)
        print "X_train[0] shape", split.X_train[0].shape
        print "X_test[0] shape", split.X_test[0].shape
        yhat = dtwi_1nn_classify(
            X_list=split.X_train, y_train=split.y_train, q_list=split.X_test)
        correct = yhat == split.y_test
        num_correct += np.sum(correct)
        total_examples += len(split.y_test)

        t = (time.clock() - t0)
        print "...took {}s".format(t)

    return num_correct / float(total_examples)


@_memory.cache
def dtwi_run_one_experiment(dset):
    splits = ds.cv_splits_for_dataset(dset)
    acc = dtwi_1nn_accuracy(dset, splits)
    print("{}: {} splits\tacc = {}".format(dset, len(splits), acc))
    return (dset, acc)


def dtwi_run_experiments():
    accuracies = []

    # for dset in [ds.VOWELS, ds.AUSLAN]:
    # for dset in [ds.VOWELS, ds.LIBRAS, ds.ECG]:
    # for dset in [ds.TRAJECTORIES]:
    # for dset in ds.NOT_TINY_DATASETS:
    for dset in ds.TINY_DATASETS:
        accuracies.append(dtwi_run_one_experiment(dset))

    df = pd.DataFrame.from_records(accuracies, columns=[
        'dataset', 'DTW-I'])
    print df
    df.to_csv(DTW_I_SAVE_PATH)


# ================================================================ main

def sanity_check_dsets():
    for dset in ds.ALL_DATASETS:
        splits = ds.cv_splits_for_dataset(dset)
        print("{}: {} splits".format(dset, len(splits)))

        for split in splits:
            print "split attr types: ", type(split.X_train), type(split.X_test), \
                type(split.y_train), type(split.y_test)
            assert split.X_train is not None
            assert split.X_test is not None
            assert split.y_train is not None
            assert split.y_test is not None

            assert len(np.unique(split.y_train)) > 1
            assert len(np.unique(split.y_test)) > 1

            print "len(split.X_train)", len(split.X_train)
            print "len(split.y_train)", len(split.y_train)
            print "len(split.X_test)", len(split.X_test)
            print "len(split.y_test)", len(split.y_test)
            assert len(split.y_train) == len(split.X_train)
            assert len(split.y_test) == len(split.X_test)


def main():
    # sanity_check_dsets()
    # return
    dtwi_run_experiments()


if __name__ == '__main__':
    main()
