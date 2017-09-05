#!/usr/bin/env python

import os

DATASETS_DIR = os.path.expanduser("../datasets/")


def to_path(*args):
    return os.path.join(DATASETS_DIR, *args)


ARABIG_DIGITS = to_path('arabic-digits')
