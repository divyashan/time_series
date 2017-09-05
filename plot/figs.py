#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from files import ensure_dir_exists

"""Figures we create here:
    -Justifying design decisions / illustrating principles:
        -{Conv, Siamese, Triplet} loss vs (1nn acc on UCR datasets)
        -(Number of neurons in fc layers) vs (1nn acc on UCR datasets)
        -(Amount of maxpooling) vs (1nn acc on UCR datasets)
            -actually, how about
    -Histograms of distances for some 2-class dataset (gun-point)
        -one hist color for same class, another for other class
        -for DTW vs our method (and ideally others)
    -Univariate Time Series
        -{Ours, decade, etc} vs acc on UCR datasets
    -Multivariate Time Series
        -{Ours, 1d dist measures} vs acc on {EEG, ...}
    // ^ actually, accuracy results should probably just be a table
    //      -and so should conv vs siamese vs triplet, probably
"""


CAMERA_READY_FONT = 'DejaVu Sans'

SAVE_DIR = os.path.expanduser('figs/')
ensure_dir_exists(SAVE_DIR)

ACC_PATH = 'placeholder-data/placeholder-acc-results.csv'
NET_SIZE_PATH = 'placeholder-data/placeholder-netsize-results.csv'
POOL_SIZE_PATH = 'placeholder-data/placeholder-poolsize-results.csv'

DATASET_COL = 'dataset'
ALGOROTHM_COL = 'algorithm'
ACC_COL = 'accuracy'
NET_SIZE_COL = 'size'
POOL_SIZE_COL = 'size_pct'

ACC_COLS = [DATASET_COL, ALGOROTHM_COL, ACC_COL]
NET_SIZE_COLS = [DATASET_COL, NET_SIZE_COL, ACC_COL]
POOL_SIZE_COLS = [DATASET_COL, POOL_SIZE_COL, ACC_COL]


def save_fig(name):
    plt.savefig(os.path.join(SAVE_DIR, name + '.pdf'), bbox_inches='tight')


def save_fig_png(name):
    plt.savefig(os.path.join(SAVE_DIR, name + '.png'),
                dpi=300, bbox_inches='tight')


def set_palette(ncolors=8):  # use this to change color palette in all plots
    pal = sb.color_palette("Set1", n_colors=8)
    sb.set_palette(pal)
    return pal


def _param_effect_fig(data_path, ycol, title, xlabel, ylabel,
                      placeholder=True, ax=None, kind='tsplot'):
    if ax is None:
        sb.set_context('talk', rc={"figure.figsize": (7, 4)})
        fig, ax = plt.subplots()

    df = pd.read_csv(data_path)

    if kind == 'tsplot':  # distro of accuracies for each size across dsets
        sb.tsplot(df, ax=ax, time=ycol, unit=DATASET_COL, value=ACC_COL)

    elif kind == 'hist':  # times each size is the best
        best_sizes = []
        all_dsets = df[DATASET_COL].unique()
        for dset in all_dsets:
            # print "dset: ", dset
            sub_df = df[df[DATASET_COL] == dset]
            idx = sub_df[ACC_COL].idxmax()
            best_sz = sub_df[ycol][idx]
            # best_counts[best_sz] = best_counts.get(best_sz, 0) + 1
            best_sizes.append(best_sz)

            # print "dset, best_sz, idx, acc: "
            print dset, best_sz, idx, sub_df[ACC_COL][idx]

        # bins = [15, 31, 63, 127, 255, 511, 1024]
        best_sizes = np.log2(best_sizes)
        bins = np.array([3, 4, 5, 6, 7, 8, 9, 10])
        sb.distplot(best_sizes, ax=ax, kde=False, bins=bins, hist_kws={
            'normed': True})

        # ax.set_title('Distribution of Best FC Layer Sizes', fontsize=18)
        # ax.set_xlabel('Log2(Neurons in each fully connected layer)', fontsize=16)
        # ax.set_ylabel('Probability of yielding highest accuracy', fontsize=16)
        # ax.set_xscale('log')
    else:
        raise ValueError("Unrecognized figure kind '{}'".format(kind))

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    # ax.set_xscale('log')

    plt.tight_layout()
    plt.show()


# def net_size_fig(placeholder=True, ax=None, kind='hist'):
def net_size_fig(kind='tsplot', **kwargs):
    if kind == 'tsplot':
        _param_effect_fig(
            data_path=NET_SIZE_PATH,
            ycol=NET_SIZE_COL,
            title='Fully Connected Layer size vs Accuracy',
            xlabel='Neurons in each fully connected layer',
            ylabel='Accuracy across all UCR datasets',
            kind=kind,
            **kwargs)
    elif kind == 'hist':
        _param_effect_fig(
            data_path=NET_SIZE_PATH,
            ycol=NET_SIZE_COL,
            title='Distribution of Best FC Layer Sizes',
            xlabel='Log2(Neurons in each fully connected layer)',
            ylabel='Probability of yielding highest accuracy',
            kind=kind,
            **kwargs)


def pool_size_fig(kind='tsplot', **kwargs):
    if kind == 'tsplot':
        _param_effect_fig(
            data_path=POOL_SIZE_PATH,
            ycol=POOL_SIZE_COL,
            title='Max Pool Size vs Accuracy',
            xlabel='Max pool size (fraction of mean time series length)',
            ylabel='Accuracy across all UCR datasets',
            kind=kind,
            **kwargs)
    elif kind == 'hist':
        _param_effect_fig(
            data_path=POOL_SIZE_PATH,
            ycol=POOL_SIZE_COL,
            title='Distribution of Best Max Pool Sizes',
            xlabel='Max pool size (fraction of mean time series length)',
            ylabel='Probability of yielding highest accuracy',
            kind=kind,
            **kwargs)


# ================================================================ main

def main():
    net_size_fig(kind='tsplot')
    pool_size_fig(kind='tsplot')
    net_size_fig(kind='hist')
    pool_size_fig(kind='hist')


if __name__ == '__main__':
    main()
