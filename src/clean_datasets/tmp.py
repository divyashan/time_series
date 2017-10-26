#!/usr/bin/env python

from src import clean_datasets as ds


def main():

    ds.cv_splits_for_dataset(ds.ECG)
    ds.cv_splits_for_dataset(ds.ECG, length_adjust='pad')
    ds.cv_splits_for_dataset(ds.ECG, length_adjust='upsample')


if __name__ == '__main__':
    main()
