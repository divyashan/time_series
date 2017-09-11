#!/usr/bin/env python

import os
import numpy as np
from joblib import Memory

from .. import paths

_memory = Memory('.', verbose=0)

TRAIN_FILE = os.path.join(paths.ARABIC_DIGITS, 'Train_Arabic_Digit.txt')
TEST_FILE = os.path.join(paths.ARABIC_DIGITS, 'Test_Arabic_Digit.txt')


# ------------------------------------------------ labels

def _create_labels(instances_per_cls):
    y = np.zeros(instances_per_cls * 10, dtype=np.int32)
    for i in range(1, 10):
        start_idx = instances_per_cls * i
        end_idx = start_idx + instances_per_cls
        y[start_idx:end_idx] = i

    return y


def train_labels():
    return _create_labels(660)


def test_labels():
    return _create_labels(220)


# ------------------------------------------------ input
# data is M x 13 time series, where M is variable


@_memory.cache
def _read_example_strs(path):
    with open(path, 'r') as f:
        examples_strs = f.read().split('\n\n')
    return examples_strs


@_memory.cache
def _read_examples(path, expected_count=None):
    examples_strs = _read_example_strs(path)
    if expected_count:
        assert len(examples_strs) == expected_count
    return [np.fromstring(s, sep=' ').reshape(-1, 13) for s in examples_strs]


def train_X():
    return _read_examples(TRAIN_FILE, expected_count=6600)


def test_X():
    return _read_examples(TEST_FILE, expected_count=2200)


def train_data():
    return train_X(), train_labels()


def test_data():
    return test_X(), test_labels()


# ================================================================ main

def main():
    X = train_X()
    print [mat.shape for mat in X[:10]]
    X = test_X()
    print [mat.shape for mat in X[:10]]


if __name__ == '__main__':
    main()
