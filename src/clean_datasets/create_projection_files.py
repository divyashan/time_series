#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import src.clean_datasets as ds
from src.utils import files


SAVE_DIR = os.path.expanduser('./datasets/')


# X_train = (X - means) / std
# X_eval = X[-50000:] - means / std
# X_eval = X_train # for pca means / stds
# svd = TruncatedSVD(n_components=65).fit(X_train)

# def _pca(X, ndims=64):
#     return svd.transform((X - means) / std)[:, 1:(ndims+1)] # top component is weird, at least for rand walks


def project(X, y, method):
    ndims = len(np.unique(y))
    if method == 'pca':
        svd = TruncatedSVD(n_components=ndims).fit(X)
        return svd.transform(X)
    elif method == 'rand':
        mat = np.random.randn(X.shape[1], ndims)
        return np.dot(X, mat)

    return None


def main(method):
    files.ensure_dir_exists(SAVE_DIR)
    for dset in ds.NOT_TINY_DATASETS:
        X, y = ds.dset_as_mat(dset)

        X = project(X, y, method=method)

        print("received mat with shape {}".format(X.shape))
        save_dir = os.path.join(SAVE_DIR, method)
        print("new shape: {}".format(X.shape))

        files.ensure_dir_exists(save_dir)
        path = os.path.join(save_dir, dset + '.txt')
        print("saving to path: {}".format(path))

        # save mat in UCR format
        ret = np.zeros((X.shape[0], X.shape[1] + 1))
        ret[:, 0] = y
        ret[:, 1:] = X
        np.savetxt(path, ret, delimiter=',')


if __name__ == '__main__':
    # main(method='rand')
    main(method='pca')
