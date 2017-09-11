#!/usr/bin/env python

import os
import numpy as np
from joblib import Memory

from .. import paths

_memory = Memory('.', verbose=0)

TRAIN_FILE = os.path.join(paths.VOWELS, 'ae.train.txt')
TEST_FILE = os.path.join(paths.VOWELS, 'ae.test.txt')


# ------------------------------------------------ labels

def train_labels():
    instances_per_cls = 30
    nclasses = 9
    y = np.zeros(instances_per_cls * nclasses, dtype=np.int32)
    for i in range(1, nclasses):
        start_idx = instances_per_cls * i
        end_idx = start_idx + instances_per_cls
        y[start_idx:end_idx] = i

    return y


def test_labels():
    class_sizes = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    y = [np.zeros(length) + i for i, length in enumerate(class_sizes)]
    y = np.hstack(y)

    # sanity check
    # _, counts = np.unique(y, return_counts=True)
    # assert np.array_equal(counts, class_sizes)

    return y.astype(np.int32)


# ------------------------------------------------ input

@_memory.cache
def _read_example_strs(path):
    with open(path, 'r') as f:
        examples_strs = f.read().split('\n\n')
    return examples_strs


@_memory.cache
def _read_examples(path, ndims=12, expected_count=None):
    examples_strs = _read_example_strs(path)
    if expected_count:
        assert len(examples_strs) == expected_count
    return [np.fromstring(s, sep=' ').reshape(-1, ndims) for s in examples_strs]


def train_X():
    return _read_examples(TRAIN_FILE)


def test_X():
    return _read_examples(TEST_FILE)


def train_data():
    return train_X(), train_labels()


def test_data():
    return test_X(), test_labels()


# ================================================================ main

def main():

    X = train_X()
    print "initial train X shapes: ", [mat.shape for mat in X[:10]]
    X = test_X()
    print "initial test X shapes: ", [mat.shape for mat in X[:10]]

    print "inital train labels: ", train_labels()[0:200:20]
    print "inital test labels: ", test_labels()[0:200:20]

    y = test_labels()
    print "X_test len, y_test shape", len(X), y.shape


if __name__ == '__main__':
    main()
