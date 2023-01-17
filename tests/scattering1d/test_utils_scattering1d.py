# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
import numpy as np
import pytest
from wavespin import Scattering1D
from wavespin.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D
from wavespin.scattering1d.scat_utils import (
    compute_border_indices, compute_padding)
from utils import FORCED_PYTEST


# set True to execute all test functions without pytest
run_without_pytest = 0


def test_compute_padding():
    """
    Test the compute_padding function

    From `tests/scattering1d/test_utils_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """

    pad_left, pad_right = compute_padding(5, 16)
    assert pad_left == 8 and pad_right == 8

    with pytest.raises(ValueError) as ve:
        _, _ = compute_padding(3, 16)
    assert "should be larger" in ve.value.args[0]


def test_border_indices(random_state=42):
    """
    Tests whether the border indices to unpad are well computed

    From `tests/scattering1d/test_utils_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    rng = np.random.RandomState(random_state)
    J_signal = 10  # signal lives in 2**J_signal
    J = 6  # maximal subsampling

    T = 2**J_signal
    log2_T = int(np.log2(T))

    i0 = rng.randint(0, T // 2 + 1, 1)[0]
    i1 = rng.randint(i0 + 1, T, 1)[0]

    x = np.ones(T)
    x[i0:i1] = 0.

    ind_start, ind_end = compute_border_indices(log2_T, J, i0, i1)

    for j in range(J + 1):
        assert j in ind_start.keys()
        assert j in ind_end.keys()
        x_sub = x[::2**j]
        # check that we did take the strict interior
        assert np.max(x_sub[ind_start[j]:ind_end[j]]) == 0.
        # check that we have not forgotten points
        if ind_start[j] > 0:
            assert np.min(x_sub[:ind_start[j]]) > 0.
        if ind_end[j] < x_sub.shape[-1]:
            assert np.min(x_sub[ind_end[j]:]) > 0.


# Check that the default frontend is numpy and that errors are correctly launched.
def test_scattering1d_frontend():
    """
    From `tests/scattering1d/test_utils_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    scattering = Scattering1D(shape=(10,), J=2, Q=1)
    assert isinstance(scattering, ScatteringNumPy1D
                      ), 'could not be correctly imported'

    with pytest.raises(ValueError) as ve:
        scattering = Scattering1D(shape=(10,), J=2, Q=1, frontend='doesnotexist')
    assert "is not valid" in ve.value.args[0]


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_compute_padding()
        test_border_indices()
        test_scattering1d_frontend()
    else:
        pytest.main([__file__, "-s"])
