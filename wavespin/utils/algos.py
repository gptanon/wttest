# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Algorithms: search, sort, etc."""
import numpy as np
from ._compiled._algos import (
    smallest_interval_over_threshold as
    smallest_interval_over_threshold_c,
    smallest_interval_over_threshold_indices as
    smallest_interval_over_threshold_indices_c)


def smallest_interval_over_threshold(x, threshold, c=None):
    """Finds smallest `interval` such that `x[start:end].sum() > threshold`,
    optionally with the interval enclosing `c`, i.e. `start <= c < end`,
    and `end = start + interval`.

    Parameters
    ----------
    x : np.ndarray, 1D
        Non-negative, real-valued array.

    threshold : float
        Threshold.

    c : int
        Index of interest (typically `argmax(x)`), must be `>= 0` and `< len(x)`.

    Returns
    -------
    interval : int
        Size of interval.

    Algorithm
    ---------
    Starting from `sum = x[0]`, increment right pointer until
    `x[left:right].sum()` exceeds `threshold`, then increment left pointer until
    it's below `threshold`. Repeat until end of array is reached. Return the
    smallest qualifying `right - left`.

    If `c` is provided, additionally constrains `left <= c` and `right > c`.
    See commented source code in `wavespin.compiled.algos.pyx`.
    """
    x, threshold = _smallest_interval_input_checks(x, threshold, c)
    if c is None:
        c = -1
    return smallest_interval_over_threshold_c(x, threshold, c)


def smallest_interval_over_threshold_indices(x, threshold, c, interval=None):
    """Finds interval whose sum exceeds `threshold` but also contains `c` -
    that is, tuple[int] such that `x[start:end].sum() > threshold` and
    `start <= c < end`.

    Prioritizes `start, end` that best centers `c`, if there are multiple
    qualifying intervals.

    Is the indexed version of `wavespin.algos.smallest_interval_over_threshold`.

    Parameters
    ----------
    x : np.ndarray, 1D
        Non-negative, real-valued array.

    threshold : float
        Threshold.

    c : int
        Index of interest (typically `argmax(x)`), must be `>= 0` and `< len(x)`.

    interval : int / None
        If already computed, will spare from recomputing.

    Returns
    -------
    start, end : (int, int)
        Indices to directly slice `x`.

    Algorithm
    ---------
    Sweep `left, right` through all possible values, as determined by `c`
    and `interval`. If there's a second qualifying interval, plan to return it
    instead if its `(left + right)/2` (midpoint) is closer to `c`. Repeat for
    third and any other qualified interval.

    See commented source code in `wavespin.compiled.algos.pyx`.
    """
    x, threshold = _smallest_interval_input_checks(x, threshold, c)
    if interval is None:
        interval = smallest_interval_over_threshold_c(x, threshold, c)
    return smallest_interval_over_threshold_indices_c(x, threshold, c, interval)


def _smallest_interval_input_checks(x, threshold, c=None):
    # x is non-empty array
    assert isinstance(x, np.ndarray), type(x)
    assert x.size > 0

    # x is real-valued
    if x.dtype.name.startswith('complex'):
        raise TypeError("`x` must be real-valued, got %s" % x.dtype)

    # x.min(), thresholds
    x_min = x.min()
    if x_min < -1e-16:
        raise ValueError("`x` cannot contain negatives; x.min() == %s" % x_min)
    elif threshold < x_min:
        raise ValueError(("`threshold` cannot be lower than `x.min()`: "
                          "{:.1e} < {:.1e}".format(threshold, x_min)))

    # c is within `len(x)`
    if c is not None:
        assert 0 <= c < len(x), c

    # must be float64 to avoid numeric errors
    if not isinstance(x, np.float64):
        x = x.astype('float64')
    if not isinstance(threshold, np.float64):
        threshold = np.array(threshold, dtype=np.float64)
    return x, threshold
