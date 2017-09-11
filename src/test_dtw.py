#!/usr/bin/env python

import numpy as np
import unittest

import dtw


def _allclose(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.allclose(x, y, atol=1e-4)


class TestDTW(unittest.TestCase):

    def test_1d(self):
        a = np.array([5, 2, 2, 3, 5.1])
        b = np.array([5, 2, 3, 3, 4.])

        d = dtw.dtw(a, b, r=0)
        assert _allclose(d, 1 + (1.1 * 1.1))

        d = dtw.dtw(a, b, r=1)
        assert _allclose(d, 1.1 * 1.1)
        # assert np.allclose(d, 1.1 * 1.1, atol=1e-3)

        a = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0])
        b = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0])
        assert _allclose(11, dtw.dtw(a, b, r=0))
        assert _allclose(8, dtw.dtw(a, b, r=1))
        assert _allclose(4, dtw.dtw(a, b, r=2))

    def test_dist_to_same_vect_is_zero(self):
        X = np.random.randn(1000, 40)
        for r in [0, 1, 5, 20]:
            for x in X:
                assert dtw.dtw(x, x, r, use_c_impl=True) == 0
            for x in X[:50]:
                assert dtw.dtw(x, x, r, use_c_impl=False) == 0

    def test_dtw_d(self):
        a = np.array([5, 2, 2, 3, 5.1])
        b = np.array([5, 2, 3, 3, 4.])
        c = np.arange(len(a))
        d = np.zeros(len(a))

        x = np.vstack((a, c)).T
        y = np.vstack((b, c)).T
        assert _allclose(1 + (1.1 * 1.1), dtw.dtw_d(x, y, r=0))
        assert _allclose(1 + (1.1 * 1.1), dtw.dtw_d(x, y, r=1))

        x = np.vstack((a, d)).T
        y = np.vstack((b, d)).T
        assert _allclose(1 + (1.1 * 1.1), dtw.dtw_d(x, y, r=0))
        assert _allclose(1.1 * 1.1, dtw.dtw_d(x, y, r=1))

    def test_dtw_i(self):
        a = np.array([5, 2, 2, 3, 5.1])
        b = np.array([5, 2, 3, 3, 4.])
        c = np.arange(len(a))
        # d = np.zeros(len(a))

        x = np.vstack((a, c)).T
        y = np.vstack((b, c)).T
        assert _allclose(1 + (1.1 * 1.1), dtw.dtw_i(x, y, r=0))
        assert _allclose((1.1 * 1.1), dtw.dtw_i(x, y, r=1))

        a = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0])
        b = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0])
        c = np.arange(len(a))
        d = c[::-1]
        x = np.vstack((c, a, d)).T
        y = np.vstack((c, b, d)).T
        assert _allclose(11, dtw.dtw_i(x, y, r=0))
        assert _allclose(8, dtw.dtw_i(y, x, r=1))
        assert _allclose(4, dtw.dtw_i(x, y, r=2))


if __name__ == '__main__':
    unittest.main()
