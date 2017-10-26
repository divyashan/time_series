#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import src.clean_datasets as dsets
from src.utils import files

# TODO this path (and rest of this file) are quick-and-dirty / hideous
SAVE_DIR = os.path.expanduser('~/codez/SPIRAL/data')


def main():
    files.ensure_dir_exists(SAVE_DIR)
    for dset in dsets.NOT_TINY_DATASETS:
        mat = dsets.dset_to_ucr_mat(dset)
        print("received mat with shape {}".format(mat.shape))

        # we're just going to run clustering on whole dataset, but their code
        # expects separate training and test sets; just create a test set of
        # size 2 to appease it (size 1 might be treated as a vector and
        # break things)
        mat_train = mat[:-2]
        mat_test = mat[-2:]

        save_dir = os.path.join(SAVE_DIR, dset)
        path_train = os.path.join(save_dir, dset + '_TRAIN')
        path_test = os.path.join(save_dir, dset + '_TEST')
        print("saving to paths: {}, {}".format(path_train, path_test))

        files.ensure_dir_exists(save_dir)
        np.savetxt(path_train, mat_train, delimiter=',')
        np.savetxt(path_test, mat_test, delimiter=',')


if __name__ == '__main__':
    main()
