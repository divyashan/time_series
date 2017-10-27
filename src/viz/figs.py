#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
# import titlecase
import numpy as np
import pandas as pd
import seaborn as sb

from src.utils.files import ensure_dir_exists

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
# NET_SIZE_PATH_SUPERVISED = 'placeholder-data/placeholder-netsize-results.csv'
# POOL_SIZE_PATH_SUPERVISED = 'placeholder-data/placeholder-poolsize-results.csv'
NET_SIZE_PATH_SUPERVISED = 'src/viz/results/netsize.csv'
POOL_SIZE_PATH_SUPERVISED = 'src/viz/results/poolsize.csv'
NET_SIZE_PATH_UNSUPERVISED = 'src/viz/results/netsize_unsupervised.csv'
POOL_SIZE_PATH_UNSUPERVISED = 'src/viz/results/poolsize_unsupervised.csv'

DATASET_COL = 'dataset'
ALGOROTHM_COL = 'algorithm'
ACC_COL = 'accuracy'
NET_SIZE_COL = 'size'
POOL_SIZE_COL = 'poolsize'

# ACC_COLS = [DATASET_COL, ALGOROTHM_COL, ACC_COL]
# NET_SIZE_COLS = [DATASET_COL, NET_SIZE_COL, ACC_COL]
# POOL_SIZE_COLS = [DATASET_COL, POOL_SIZE_COL, ACC_COL]


def save_fig(name):
    plt.savefig(os.path.join(SAVE_DIR, name + '.pdf'), bbox_inches='tight')


def save_fig_png(name):
    plt.savefig(os.path.join(SAVE_DIR, name + '.png'),
                dpi=300, bbox_inches='tight')


def set_palette(ncolors=8):  # use this to change color palette in all plots
    pal = sb.color_palette("Set1", n_colors=8)
    sb.set_palette(pal)
    return pal


def _param_effect_fig(data_path, xcol, title, xlabel, ylabel,
                      placeholder=True, ax=None, kind='tsplot'):
    if ax is None:
        sb.set_context('talk', rc={"figure.figsize": (7, 4)})
        fig, ax = plt.subplots()

    df = pd.read_csv(data_path)

    if kind == 'tsplot':  # distro of accuracies for each size across dsets
        sb.tsplot(df, ax=ax, time=xcol, unit=DATASET_COL, value=ACC_COL)
        plt.tight_layout()

    elif kind == 'hist':  # times each size is the best
        best_sizes = []
        all_dsets = df[DATASET_COL].unique()
        for dset in all_dsets:
            print "dset: ", dset
            print "df dset col: ", df[DATASET_COL]
            sub_df = df[df[DATASET_COL] == dset]
            idx = sub_df[ACC_COL].idxmax()
            best_sz = sub_df[xcol][idx]
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
        plt.tight_layout()

    elif kind == 'line':
        dsets = df[DATASET_COL].unique()
        # dset_names = [name.capitalize() for name in dsets]
        for dset in dsets:
            sub_df = df[df[DATASET_COL] == dset]
            xvals = sub_df[xcol]
            yvals = sub_df[ACC_COL]
            name = dset.replace('_', ' ').replace('-', ' ').capitalize()
            ax.plot(xvals, yvals, label=name)

        leg_lines, leg_labels = ax.get_legend_handles_labels()
        plt.figlegend(leg_lines, leg_labels, loc='lower center',
                      ncol=2, labelspacing=0)

        # plt.tight_layout(w_pad=.02)
        plt.tight_layout()
        plt.subplots_adjust(bottom=.25)
    else:
        raise ValueError("Unrecognized figure kind '{}'".format(kind))

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    # ax.set_xscale('log')

    plt.show()


def param_effects_fig(placeholder=True, supervised=True):
    sb.set_context('talk')
    # sb.set_context('poster')
    # sb.set_context('notebook')
    # fig, axes = plt.subplots(2, figsize=(6, 8))
    # fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    if supervised:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.72))  # 10% taller

    KEEP_HOW_MANY = 10  # plotting too many makes fig hideous

    if supervised:
        df_fc = pd.read_csv(NET_SIZE_PATH_SUPERVISED)
        df_pool = pd.read_csv(POOL_SIZE_PATH_SUPERVISED)
    else:
        df_fc = pd.read_csv(NET_SIZE_PATH_UNSUPERVISED)
        df_pool = pd.read_csv(POOL_SIZE_PATH_UNSUPERVISED)

    # print "df fc:"
    # print df_fc
    # print "df_pool"
    # print df_pool
    # return

    # make sure both use the same datasets, because otherwise the
    # legend will break / be wrong
    dsets_fc = sorted(df_fc[DATASET_COL].unique())
    dsets_pool = sorted(df_pool[DATASET_COL].unique())
    assert np.array_equal(dsets_fc, dsets_pool)
    dsets = dsets_fc

    dset_names_lens = np.array([len(name) for name in dsets])
    sort_idxs = np.argsort(dset_names_lens)
    dsets = [dsets[i] for i in sort_idxs[:KEEP_HOW_MANY]]

    # print "param_effects_fig: using datasets: ", dsets
    # return

    # ------------------------ top plot: fc layer size

    fc_params = (df_fc, NET_SIZE_COL, axes[0])
    pool_params = (df_pool, POOL_SIZE_COL, axes[1])
    for (df, xcol, ax) in (fc_params, pool_params):
        for dset in dsets:
            sub_df = df[df[DATASET_COL] == dset]
            sub_df = sub_df.sort_values(xcol)
            xvals, yvals = sub_df[xcol], sub_df[ACC_COL]
            yvals /= yvals.max()
            # name = dset.replace('_', ' ').replace('-', ' ').capitalize()
            name = dset.replace('_', ' ').replace('-', ' ')
            ax.plot(xvals, yvals, label=name)

    leg_lines, leg_labels = ax.get_legend_handles_labels()
    plt.figlegend(leg_lines, leg_labels, loc='lower center',
                  ncol=5, labelspacing=0)

    ax = axes[0]
    ax.set_title("Effect of Fully Connected Layer Size", y=1.03)
    if supervised:
        ax.set_xlabel("Neurons in Each Fully Connected Layer")
    else:
        ax.set_xlabel("Neurons in Each Fully Connected Layer\n"
                      "(Fraction of # of classes)")
    ax.set_ylabel("Normalized Accuracy")
    ax = axes[1]
    ax.set_title("Effect of Max Pool Size", y=1.03)
    ax.set_xlabel("Fraction of Mean Time Series Length")
    # ax.set_xlabel("Max Pool Size\n(Fraction of Mean Time Series Length)")
    ax.set_ylabel("Normalized Accuracy")

    # plt.tight_layout(w_pad=.02)
    # plt.tight_layout(h_pad=2.0)
    plt.tight_layout(h_pad=1.8)
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=.32)  # this one with horz but 2 legend cols
    # plt.subplots_adjust(bottom=.23)  # this one for vertical subplots
    if supervised:
        plt.subplots_adjust(bottom=.25)
    else:
        plt.subplots_adjust(bottom=.27)

    # plt.show()
    figname = 'param_effects'
    if not supervised:
        figname += '_unsupervised'
    save_fig_png(figname)


# def net_size_fig(placeholder=True, ax=None, kind='hist'):
def net_size_fig(kind='tsplot', **kwargs):
    if kind == 'tsplot':
        _param_effect_fig(
            data_path=NET_SIZE_PATH_SUPERVISED,
            xcol=NET_SIZE_COL,
            title='Fully Connected Layer size vs Accuracy',
            xlabel='Neurons in each fully connected layer',
            ylabel='Accuracy across all UCR datasets',
            kind=kind, **kwargs)
    elif kind == 'hist':
        _param_effect_fig(
            data_path=NET_SIZE_PATH_SUPERVISED,
            xcol=NET_SIZE_COL,
            title='Distribution of Best FC Layer Sizes',
            xlabel='Log2(Neurons in each fully connected layer)',
            ylabel='Probability of yielding highest accuracy',
            kind=kind, **kwargs)
    elif kind == 'line':
        _param_effect_fig(
            data_path=NET_SIZE_PATH_SUPERVISED,
            xcol=NET_SIZE_COL,
            title='Fully Connected Layer Size vs Accuracy',
            xlabel='Neurons in each fully connected layer',
            ylabel='Accuracy',
            kind=kind, **kwargs)


def pool_size_fig(kind='tsplot', **kwargs):
    if kind == 'tsplot':
        _param_effect_fig(
            data_path=POOL_SIZE_PATH_SUPERVISED,
            xcol=POOL_SIZE_COL,
            title='Max Pool Size vs Accuracy',
            xlabel='Max pool size (fraction of mean time series length)',
            ylabel='Accuracy across all UCR datasets',
            kind=kind, **kwargs)
    elif kind == 'hist':
        _param_effect_fig(
            data_path=POOL_SIZE_PATH_SUPERVISED,
            xcol=POOL_SIZE_COL,
            title='Distribution of Best Max Pool Sizes',
            xlabel='Max pool size (fraction of mean time series length)',
            ylabel='Probability of yielding highest accuracy',
            kind=kind, **kwargs)
    elif kind == 'line':
        _param_effect_fig(
            data_path=POOL_SIZE_PATH_SUPERVISED,
            xcol=POOL_SIZE_COL,
            title='Max Pooling Size vs Accuracy',
            xlabel='Max pool size (fraction of mean time series length)',
            ylabel='Accuracy',
            kind=kind, **kwargs)


# ================================================================ main

def main():
    # these are the old, unused version
    # net_size_fig(kind='line')
    # pool_size_fig(kind='line')
    # pool_size_fig(kind='tsplot')
    # net_size_fig(kind='tsplot')
    # pool_size_fig(kind='tsplot')
    # net_size_fig(kind='hist')
    # pool_size_fig(kind='hist')

    # param_effects_fig()
    param_effects_fig(supervised=False)


if __name__ == '__main__':
    main()
