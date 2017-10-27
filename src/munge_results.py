#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy import stats


PLACEHOLDER_RESULTS_DIR = 'src/placeholder_results'
ERR_RATES_PATH = os.path.join(PLACEHOLDER_RESULTS_DIR, 'placeholder_err_rates.csv')
ACCURACIES_PATH = os.path.join(PLACEHOLDER_RESULTS_DIR, 'placeholder_accuracies.csv')
WEASEL_PATH = os.path.join(PLACEHOLDER_RESULTS_DIR, 'weasel-results-spreadsheet.csv')
DECADE_PATH = os.path.join(PLACEHOLDER_RESULTS_DIR, 'decade-accuracies.csv')

RESULTS_DIR = 'src/final-results'
UCR_PATH = os.path.join(RESULTS_DIR, 'ucr-accuracies.csv')
# UCI_PATH = os.path.join(RESULTS_DIR, 'uci-err-rates.csv')
UCI_PATH = os.path.join(RESULTS_DIR, 'uci-accuracies.csv')
UCI_PATH_UNSUPERVISED = os.path.join(RESULTS_DIR, 'unsupervised-results-cleaned.csv')
# UCI_ALL_PATH = os.path.join(RESULTS_DIR, 'uci-accuracies-all.csv')


def _compute_ranks(df, lower_better=True):
    """assumes each row of X is a dataset and each col is an algorithm"""
    # return df.rank(axis=1, numeric_only=True, ascending=lower_better)
    return df.rank(axis=1, numeric_only=True, ascending=lower_better, method='min')


def cd_diagram(df, lower_better=True):
    import Orange as orn  # requires py3.4 or greater environment

    ranks = _compute_ranks(df, lower_better=lower_better)
    names = [s.strip() for s in ranks.columns]
    mean_ranks = ranks.mean(axis=0)
    ndatasets = df.shape[0]

    print("--- raw ranks:")
    print(ranks)

    print("--- mean ranks:")
    print("\n".join(["{}: {}".format(name, rank)
                     for (name, rank) in zip(names, mean_ranks)]))

    # alpha must be one of {'0.1', '0.05', '0.01'}
    cd = orn.evaluation.compute_CD(mean_ranks, ndatasets, alpha='0.1')
    # cd = orn.evaluation.compute_CD(mean_ranks, ndatasets, alpha='0.05')
    orn.evaluation.graph_ranks(mean_ranks, names, cd=cd, reverse=True)
    print("\nNemenyi critical difference: ", cd)


def pairwise_significance(df, lower_better=True, alpha=.05):
    names = [s.strip() for s in df.columns]
    df.columns = names
    print("col names:", names)
    ours = df['Proposed']
    names.remove('Dataset')
    names.remove('Proposed')

    # print("required p val with bonferroni: {}".format(alpha / len(names)))
    # print("required p val with bonferroni: {}".format(alpha))

    # corrected_alphas = np.array([.1, .05, .01]) / float(len(names))
    alphas = np.array([.1, .05, .01])
    corrected_alphas = alphas / float(len(names))

    print("Method\t\tp\tSignif\tUncorrected Signif")
    for name in names:
        _, p = stats.mannwhitneyu(ours, df[name], alternative='two-sided')
        marker = '^' * np.sum(p < alphas)
        marker_corrected = '*' * np.sum(p < corrected_alphas)
        if len(name) < 8:
            name += '\t'
        print("{}\t{:.5f}\t{}\t{}".format(
            name, p, marker_corrected, marker))


# ================================================================ main

def main():
    # # cd_diagram(pd.read_csv(ACCURACIES_PATH), lower_better=False)
    # # cd_diagram(pd.read_csv(ERR_RATES_PATH), lower_better=True)
    # # cd_diagram(pd.read_csv(WEASEL_PATH), lower_better=False)
    # # cd_diagram(pd.read_csv(DECADE_PATH), lower_better=False)
    # # cd_diagram(pd.read_csv(UCR_PATH), lower_better=False)
    # cd_diagram(pd.read_csv(UCI_PATH), lower_better=False)
    cd_diagram(pd.read_csv(UCI_PATH_UNSUPERVISED), lower_better=False)
    # # cd_diagram(pd.read_csv(UCI_ALL_PATH), lower_better=False)
    # plt.show()
    # return

    # pairwise_significance(pd.read_csv(UCI_PATH), lower_better=False)

    # df = pd.read_csv(ERR_RATES_PATH)
    # ranks = _compute_ranks(df)
    # mean_ranks = ranks.mean(axis=0)
    # print "df, ranks, mean ranks:"
    # print df
    # print mean_ranks


if __name__ == '__main__':
    main()
