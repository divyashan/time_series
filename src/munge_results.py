#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
import os
# import numpy as np
import pandas as pd


PLACEHOLDER_RESULTS_DIR = 'src/placeholder_results'
ERR_RATES_PATH = os.path.join(PLACEHOLDER_RESULTS_DIR, 'placeholder_err_rates.csv')
ACCURACIES_PATH = os.path.join(PLACEHOLDER_RESULTS_DIR, 'placeholder_accuracies.csv')
WEASEL_PATH = os.path.join(PLACEHOLDER_RESULTS_DIR, 'weasel-results-spreadsheet.csv')
DECADE_PATH = os.path.join(PLACEHOLDER_RESULTS_DIR, 'decade-accuracies.csv')

RESULTS_DIR = 'src/final-results'
UCR_PATH = os.path.join(RESULTS_DIR, 'ucr-accuracies.csv')
# UCI_PATH = os.path.join(RESULTS_DIR, 'uci-err-rates.csv')
UCI_PATH = os.path.join(RESULTS_DIR, 'uci-accuracies.csv')


def _compute_ranks(df, lower_better=True):
    """assumes each row of X is a dataset and each col is an algorithm"""
    return df.rank(axis=1, numeric_only=True, ascending=lower_better)


def cd_diagram(df, lower_better=True):
    import Orange as orn  # requires py3.4 or greater environment

    ranks = _compute_ranks(df, lower_better=lower_better)
    names = [s.strip().capitalize() for s in ranks.columns]
    mean_ranks = ranks.mean(axis=0)
    ndatasets = df.shape[0]

    print("--- mean ranks:")
    print("\n".join(["{}: {}".format(name, rank)
                     for (name, rank) in zip(names, mean_ranks)]))

    # alpha must be one of {'0.1', '0.05', '0.01'}
    cd = orn.evaluation.compute_CD(mean_ranks, ndatasets, alpha='0.05')
    orn.evaluation.graph_ranks(mean_ranks, names, cd=cd, reverse=True)
    print("\nNemenyi critical difference: ", cd)


# ================================================================ main

def main():
    # cd_diagram(pd.read_csv(ACCURACIES_PATH), lower_better=False)
    # cd_diagram(pd.read_csv(ERR_RATES_PATH), lower_better=True)
    # cd_diagram(pd.read_csv(WEASEL_PATH), lower_better=False)
    # cd_diagram(pd.read_csv(DECADE_PATH), lower_better=False)
    # cd_diagram(pd.read_csv(UCR_PATH), lower_better=False)
    cd_diagram(pd.read_csv(UCI_PATH), lower_better=False)
    # cd_diagram(pd.read_csv(UCI_PATH), lower_better=True)
    plt.show()

    # df = pd.read_csv(ERR_RATES_PATH)
    # ranks = _compute_ranks(df)
    # mean_ranks = ranks.mean(axis=0)
    # print "df, ranks, mean ranks:"
    # print df
    # print mean_ranks


if __name__ == '__main__':
    main()
