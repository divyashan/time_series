#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import src.clean_datasets as dsets
from src.utils import files

# TODO this path (and rest of this file) are quick-and-dirty / hideous
USHAPELET_SAVE_DIR = os.path.expanduser('~/codez/UShapelet/our_datasets')


def main():
    files.ensure_dir_exists(USHAPELET_SAVE_DIR)

    for dset in dsets.NOT_TINY_DATASETS:
        mat = dsets.dset_to_ucr_mat(dset)
        print("received mat with shape {}".format(mat.shape))
        path = os.path.join(USHAPELET_SAVE_DIR, dset + '.txt')
        print("saving to path: {}".format(path))
        np.savetxt(path, mat, delimiter=',')


if __name__ == '__main__':
    main()
