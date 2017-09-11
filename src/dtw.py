#!/usr/bin/env python

import numpy as np

import numba

try:
    import dig  # for dtw; github.com/dblalock/dig
    HAS_C_IMPL = True
except ImportError:
    HAS_C_IMPL = False

assert HAS_C_IMPL  # TODO rm


def _distsq(x, y):
    diffs = x - y
    return np.sum(diffs * diffs)


@numba.jit
def _inner_loop_jit(x_i, y, costs_prev, costs, k,
                    first_row_in_window, last_row_in_window):
    for j in xrange(first_row_in_window + 1, last_row_in_window + 1):
        in_new_row = j == last_row_in_window
        d_h = in_new_row and np.inf or costs_prev[k + 1]   # horizontal
        d_d = costs_prev[k]                                # diagonal
        d_v = costs[k - 1]                                 # vertical
        diff = x_i - y[j]
        costs[k] = min(min(d_h, d_d), d_v) + np.sum(diff * diff)
        k += 1


def _dtw_jit(x, y, r, thresh, costs, costs_prev):
    m = len(x)
    final_idx = m - 1
    k = r  # first entry in costs array initially at index r

    # calculate dists in first col
    diff = x[0] - y[0]
    costs_prev[k] = np.sum(diff * diff)
    k += 1
    last_row_in_window = r
    x0 = x[0]
    for j in range(1, last_row_in_window + 1, 1):
        diff = (x0 - y[j])
        costs_prev[k] = costs_prev[k - 1] + np.sum(diff * diff)
        k += 1

    # calculate dists in remaining cols
    for i in xrange(1, m):
        k = max(0, r - i)
        first_row_in_window = max(0, i - r)
        last_row_in_window = min(final_idx, i + r)

        # handle first row in warping window separately,
        # since it has no vertical component
        in_first_row = first_row_in_window == 0
        min_prev_dist = costs_prev[k + 1]  # horizontal distance
        if (not in_first_row) and (costs_prev[k] < min_prev_dist):
            min_prev_dist = costs_prev[k]  # diagonal distance
        diff = x[i] - y[first_row_in_window]
        costs[k] = min_prev_dist + np.sum(diff * diff)
        k += 1

        _inner_loop_jit(
            x[i], y, costs_prev, costs, k, first_row_in_window, last_row_in_window)

        # set current array as prev array by swapping "pointers"
        tmp = costs
        costs = costs_prev
        costs_prev = tmp

        # early abandon if we've been given an early abandoning threshold
        # if thresh is not None:
        min_dist = np.min(costs_prev)
        if min_dist > thresh:
            return min_dist

    return costs_prev[r]


def dtw(x, y, r, dist_func=None, thresh=None, jit=False, use_c_impl=False):
    # x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape
    # x = np.ascontiguousarray(x, dtype=np.float32)
    # y = np.ascontiguousarray(y, dtype=np.float32)
    m = len(x)

    if isinstance(r, float):  # fractional warping constraint to int
        r = int(r * m)
    assert r >= 0
    if dist_func is None:
        dist_func = _distsq

    # edge case of no warping window
    if r == 0:
        return dist_func(x, y)

    # hmm...can't get JIT func to actually be faster
    # if jit and dist_func == _distsq:
    #     costs = np.zeros(2*r + 1, dtype=x.dtype)
    #     costs_prev = np.zeros(2*r + 1, dtype=x.dtype)
    #     if thresh is None or thresh <= 0:
    #         thresh = 1e9
    #     return _dtw_jit(x, y, r, thresh, costs, costs_prev)

    if use_c_impl:
        return dig.dist_dtw(x, y, r)

    # use_c_impl = (len(x.shape) == 1) and dist_func == _distsq
    # use_c_impl = use_c_impl and (x.dtype == y.dtype)
    # use_c_impl = use_c_impl and (x.dtype == np.float64)  # only does dbls...
    # if use_c_impl:
    #     # print "x shape, y shape", x.shape, y.shape
    #     try:
    #         # assert x.dtype == np.float64  # only implemented for dbls...
    #         # assert y.dtype == np.float64  # only implemented for dbls...
    #         return dist_dtw(x, y, r)
    #     except ImportError:
    #         pass

    # allocate tmp storage
    costs = np.zeros(2*r + 1, dtype=x.dtype)
    costs_prev = np.zeros(2*r + 1, dtype=x.dtype)

    final_idx = m - 1
    k = r  # first entry in costs array initially at index r

    # calculate dists in first col
    costs_prev[k] = dist_func(x[0], y[0])
    k += 1
    last_row_in_window = r
    x0 = x[0]
    for j in range(1, last_row_in_window + 1, 1):
        costs_prev[k] = costs_prev[k - 1] + dist_func(x0, y[j])
        k += 1

    # calculate dists in remaining cols
    for i in xrange(1, m):
        k = max(0, r - i)
        first_row_in_window = max(0, i - r)
        last_row_in_window = min(final_idx, i + r)

        # handle first row in warping window separately,
        # since it has no vertical component
        in_first_row = first_row_in_window == 0
        min_prev_dist = costs_prev[k + 1]  # horizontal distance
        if (not in_first_row) and (costs_prev[k] < min_prev_dist):
            min_prev_dist = costs_prev[k]  # diagonal distance
        costs[k] = min_prev_dist + dist_func(x[i], y[first_row_in_window])
        k += 1

        # classic DTW for remaining rows
        for j in xrange(first_row_in_window + 1, last_row_in_window + 1):
            in_new_row = j == last_row_in_window

            # next line is equivalent to this, but avoids index error by
            # short-circuiting the else
            # d_h = np.inf if in_new_row else costs_prev[k + 1]
            d_h = in_new_row and np.inf or costs_prev[k + 1]   # horizontal
            d_d = costs_prev[k]                                # diagonal
            d_v = costs[k - 1]                                 # vertical
            costs[k] = min(min(d_h, d_d), d_v) + dist_func(x[i], y[j])
            k += 1

        # set current array as prev array by swapping "pointers"
        tmp = costs
        costs = costs_prev
        costs_prev = tmp

        # early abandon if we've been given an early abandoning threshold
        if thresh is not None:
            min_dist = np.min(costs_prev)
            if min_dist > thresh:
                return min_dist

    return costs_prev[r]


def dtw_d(x, y, r, **kwargs):
    return dtw(x, y, r, **kwargs)


def dtw_i(x, y, r, d_best=np.inf, **kwargs):
    if HAS_C_IMPL:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    else:
        x, y = np.asarray(x), np.asarray(y)
    if len(x.shape) == 1:
        return dtw(x, y, r, **kwargs)

    total = 0.
    for d in range(x.shape[1]):
        total += dtw(x[:, d], y[:, d], r, use_c_impl=HAS_C_IMPL, **kwargs)
        if total >= d_best:  # try early abandoning
            return total

    return total


# ================================================================ main

def main():
    a = [5, 2, 2, 3, 5.1]
    b = [5, 2, 3, 3, 4]

    # d = dtw(a, b, r=0)
    # assert np.abs(d - (1 + 1.1 * 1.1)) < .001

    d = dtw(a, b, r=1)
    assert np.abs(d - (1.1 * 1.1)) < .001


if __name__ == '__main__':
    main()
