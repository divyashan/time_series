#!/usr/bin/env python

import os

DATASETS_DIR = "datasets/"  # assumes running from proj root


def to_path(*args):
    return os.path.join(DATASETS_DIR, *args)


# UCI archive datasets
ARABIC_DIGITS = to_path('arabic-digits')
AUSLAN = to_path('auslan-high-quality')
TRAJECTORIES = to_path('char-trajectories')
VOWELS = to_path('JapaneseVowels')
LIBRAS = to_path('Libras')
PEN_DIGITS = to_path('PenDigits')
ROBOT_FAILURE = to_path('robot-failure')

# datasets from http://www.cs.cmu.edu/~bobski/data/data.html
ECG = to_path('ecg')
WAFER = to_path('wafer')


# ================================================================ main

def main():
    """simple smoketest to make sure all directories exist"""
    from .utils import files
    all_dirs = [ARABIC_DIGITS, AUSLAN, TRAJECTORIES, VOWELS, LIBRAS,
                PEN_DIGITS, ROBOT_FAILURE, ECG, WAFER]

    print "========================== paths.py: all dataset dirs and contents"
    for d in all_dirs:
        print "{}:\n\t{}".format(d, "\n\t".join(files.list_visible_files(d)))


if __name__ == '__main__':
    main()
