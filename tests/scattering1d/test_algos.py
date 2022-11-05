# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Tests related to algorithms (search, sort, etc)."""
import pytest
import numpy as np
from wavespin.utils.algos import (smallest_interval_over_threshold,
                                  smallest_interval_over_threshold_indices)
from utils import FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 0


#### Algorithms tests ##########################################################
def test_smallest_interval_over_threshold():
    """Tests the algorithm on

      1. |WGN|, validated using greedy search
      2. Shifted constants. Many otherwise valid "support" computation algorithms
         will fail this, as all array values are equal and there's no decay.
      3. handled exceptions:
          - `threshold < x.min()`
          - `x.min() < 0`
          - complex `x`
    """
    # noise ##################################################################
    for N in (11, 101, 1001, 10001, 100001):
        # test various extrema on short sequences
        n_trials = 1000 if N in (11, 101) else 10
        for i in range(n_trials):
            np.random.seed(i)
            x = np.abs(np.random.randn(N))
            thresholds = make_thresholds(x)

            for j, threshold in enumerate(thresholds):
                interval = smallest_interval_over_threshold(x, threshold)
                greedy_validate_interval(x, threshold, interval)

    # shifted constant #######################################################
    # here we must restrict `shift` as otherwise it changes the meaning of
    # "support", e.g. `[1, 1, 0]` has support 2 but `[1, 0, 1]` has 3
    N = 11
    for const_len in range(1, N + 1):
        x = np.zeros(N)
        x[:const_len] = 1
        threshold = const_len - .01
        for shift in range(N - const_len + 1):
            xr = np.roll(x, shift)
            # test interval
            interval = smallest_interval_over_threshold(xr, threshold)
            assert interval == const_len, (interval, const_len, shift)
            # test indexing, "center" around `shift` will include any interval
            start, end = smallest_interval_over_threshold_indices(
                xr, threshold, c=shift, interval=interval)
            assert start == shift, (start, shift, interval)
            assert end == shift + interval, (end, shift, interval)

    # exception handling #####################################################
    x = np.array([1. + 1j])
    # complex dtype
    with pytest.raises(TypeError):
        _ = smallest_interval_over_threshold(x, 0.)
    # threshold < x.min()
    with pytest.raises(ValueError) as e:
        _ = smallest_interval_over_threshold(x.real, -1.)
        assert "cannot be lower" in e.value.args[0], e.value.args[0]
    # x.min() < 0
    with pytest.raises(ValueError) as e:
        _ = smallest_interval_over_threshold(-x.real, 0.)
        assert "cannot contain negatives" in e.value.args[0], e.value.args[0]


def test_smallest_interval_over_threshold_around_c():
    """Tests the functionality of the `start <= c < end` constraint, used
    to ensure bandwidth intervals include `argmax(psi)`.
    """
    # noise ##################################################################
    for N in (11, 101, 1001, 10001, 100001):
        n_trials = 1000 if N in (11, 101) else 10
        for i in range(n_trials):
            np.random.seed(i)
            x = np.abs(np.random.randn(N))
            c = np.argmax(x)
            thresholds = make_thresholds(x)

            for j, threshold in enumerate(thresholds):
                interval = smallest_interval_over_threshold(x, threshold, c)
                greedy_validate_interval(x, threshold, interval, c)

                # now validate indices
                start, end = smallest_interval_over_threshold_indices(
                    x, threshold, c, interval)
                sm = x[start:end].sum()
                assert sm > threshold, (sm, threshold)
                assert start <= c < end, (start, c, end)

    # exception handling #####################################################
    with pytest.raises(AssertionError):
        _ = smallest_interval_over_threshold(x, threshold, c=-5)


#### Validation methods ######################################################
def greedy_validate_interval(x, threshold, interval, c=None):
    """Tests that `x[start:end].sum() > threshold` exists for
    `end = start + interval`, and doesn't exist for `end = start + interval - 1`.
    """
    result, sums, start, end = _found_sum_over_threshold(x, threshold,
                                                         interval, c)
    if not result:
        assert False, ("no interval of size {} sums over threshold {:.2f}; "
                       "max found sum is {:.2f}").format(
                           interval, threshold, np.max(sums))

    result, sums, start, end = _found_sum_over_threshold(x, threshold,
                                                         interval - 1, c)
    if result:
        assert False, ("x[{}:{}].sum() = {:.2f} > {:.2f}\n"
                       "{}-{}={} is smaller than {} and still beats threshold"
                       ).format(start, end, x[start:end].sum(), threshold,
                                end, start, end - start, interval)


def _found_sum_over_threshold(x, threshold, interval, c=None):
    N = len(x)
    csumz = np.zeros(N + 1, dtype=x.dtype)
    csumz[1:] = np.cumsum(x)

    start = 0
    end_start = interval if c is None else c + 1
    sums = []
    for end in range(end_start, N + 1):
        start = end - interval
        if c is not None and start > c:
            # print("RETURNED")
            return False, sums, -1, -1
        sm = csumz[end] - csumz[start]
        sums.append(sm)
        if sm > threshold:
            # print("BROKE", sm, threshold)
            break
    else:
        return False, sums, -1, -1
    return True, sums, start, end


def make_thresholds(x):
    return [
        x.sum()/10,    # moderate interval
        x.sum()/1.1,   # large interval
        x.min()*1.01,  # minimal interval, *1.01 for numeric stability
        x.max()*0.99,  # minimal interval from max side, *.99 for num stab
    ]


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_smallest_interval_over_threshold()
        test_smallest_interval_over_threshold_around_c()
    else:
        pytest.main([__file__, "-s"])
