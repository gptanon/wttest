# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
"""
Testing all functions in filters_bank
"""
import pytest
import math
import numpy as np

from wavespin.scattering1d.filter_bank import (
    adaptive_choice_P, fold_filter_fourier, get_normalizing_factor,
    compute_sigma_psi, compute_xi_max, morlet_1d, gauss_1d,
    calibrate_scattering_filters)
from wavespin.utils.measures import compute_bandwidth
from wavespin import CFG
from utils import FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1


def test_adaptive_choice_P():
    """
    Tests whether adaptive_choice_P provides a bound P which satisfies
    the adequate requirements

    From `tests/scattering1d/test_filters_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    sigma_range = np.logspace(-5, 2, num=10)
    eps_range = np.logspace(-10, -5, num=8)
    for i in range(sigma_range.size):
        for j in range(eps_range.size):
            sigma = sigma_range[i]
            eps = eps_range[j]
            # choose the formula
            P = adaptive_choice_P(sigma, eps=eps)
            # check at the boundaries
            denom = 2 * (sigma**2)
            lim_left = np.exp(-((1 - P)**2) / denom)
            lim_right = np.exp(-(P**2) / denom)
            assert lim_left <= eps
            assert lim_right <= eps


def test_fold_filter_fourier(random_state=42):
    """
    Tests whether the periodization in Fourier corresponds to
    a subsampling in time

    From `tests/scattering1d/test_filters_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    rng = np.random.RandomState(random_state)
    size_signal = [2**j for j in range(5, 10)]
    periods = [2**k for k in range(0, 6)]

    for N in size_signal:
        x = rng.randn(N) + 1j * rng.randn(N)
        x_f = np.fft.fft(x)
        for per in periods:
            x_per_f = fold_filter_fourier(x_f, nperiods=per,
                                               aggregation='mean')
            x_per = np.fft.ifft(x_per_f)
            assert np.max(np.abs(x_per - x[::per])) < 1e-7


def test_get_normalizing_factor(random_state=42):
    """
    Tests whether the computation of the normalizing factor does the correct
    job (i.e. actually normalizes the signal in l1 or l2)

    From `tests/scattering1d/test_filters_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    rng = np.random.RandomState(random_state)
    size_signal = [2**j for j in range(5, 13)]
    norm_type = ['l1', 'l2']
    for N in size_signal:
        x = rng.randn(N) + 1j * rng.randn(N)
        x_f = np.fft.fft(x)
        for norm in norm_type:
            kappa = get_normalizing_factor(x_f, norm)
            x_norm = kappa * x
            if norm == 'l1':
                assert np.isclose(np.sum(np.abs(x_norm)) - 1, 0.)
            elif norm == 'l2':
                assert np.isclose(np.sqrt(np.sum(np.abs(x_norm)**2)) - 1., 0.)

    with pytest.raises(ValueError) as ve:
        get_normalizing_factor(np.zeros(4))
    assert "Zero division error is very likely" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        get_normalizing_factor(np.ones(4), normalize='l0')
    assert "`normalize` must be" in ve.value.args[0]


def test_morlet_1d():
    """
    Tests for Morlet wavelets:
    - Make sure that it has exact zero mean
    - Make sure that it has a fast decay in time
    - Check that the maximal frequency is relatively close to xi,
        up to 1% accuracy

    From `tests/scattering1d/test_filters_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    size_signal = [2**13]
    Q_range = np.arange(1, 20, dtype=int)
    P_range = [1, 5]
    for N in size_signal:
        for Q in Q_range:
            xi_max = compute_xi_max(Q)
            xi_range = xi_max / np.power(2, np.arange(7))
            for xi in xi_range:
                for P in P_range:
                    sigma = compute_sigma_psi(xi, Q)
                    # get the morlet for these parameters
                    psi_f = morlet_1d(N, xi, sigma, normalize='l2', P_max=P)
                    # make sure that it has zero mean
                    assert np.isclose(psi_f[0], 0.)
                    # make sure that it has a fast decay in time
                    psi = np.fft.ifft(psi_f)
                    psi_abs = np.abs(psi)
                    assert np.min(psi_abs) / np.max(psi_abs) < 1e-3
                    # Check that the maximal frequency is relatively close to xi,
                    # up to 1 percent
                    k_max = np.argmax(np.abs(psi_f))
                    xi_emp = float(k_max) / float(N)
                    assert np.abs(xi_emp - xi) / xi < 1e-2

    Q = 1
    xi = compute_xi_max(Q)
    sigma = compute_sigma_psi(xi, Q)

    with pytest.raises(ValueError) as ve:
        morlet_1d(size_signal[0], xi, sigma, P_max=5.1)
    assert "should be an int" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        morlet_1d(size_signal[0], xi, sigma, P_max=-5)
    assert "should be non-negative" in ve.value.args[0]


def test_gauss_1d():
    """
    Tests for Gabor low-pass
    - Make sure that it has a fast decay in time
    - Make sure that it is symmetric, up to 1e-7 absolute precision

    From `tests/scattering1d/test_filters_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    N = 2**13
    J = 7
    P_range = [1, 5]
    sigma0 = 0.1
    tol = 1e-7
    for j in range(1, J + 1):
        for P in P_range:
            sigma_low = sigma0 / math.pow(2, j)
            g_f = gauss_1d(N, sigma_low, P_max=P)
            # check the symmetry of g_f
            assert np.max(np.abs(g_f[1:N // 2] - g_f[N // 2 + 1:][::-1])) < tol
            # make sure that it has a fast decay in time
            phi = np.fft.ifft(g_f)
            assert np.min(phi) > - tol
            assert np.min(np.abs(phi)) / np.max(np.abs(phi)) < 1e-4

    Q = 1
    xi = compute_xi_max(Q)
    sigma = compute_sigma_psi(xi, Q)

    with pytest.raises(ValueError) as ve:
        gauss_1d(N, xi, sigma, P_max=5.1)
    assert "should be an int" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        gauss_1d(N, xi, sigma, P_max=-5)
    assert "should be non-negative" in ve.value.args[0]


def test_calibrate_scattering_filters():
    """
    Various tests on the central frequencies xi and spectral width sigma
    computed for the scattering filterbank
    - Checks that all widths are > 0
    - Check that sigma_low is smaller than all sigma2

    From `tests/scattering1d/test_filters_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    J_range = np.arange(2, 11)
    Q_range = np.arange(1, 21, dtype=int)
    for J in J_range:
        for Q in Q_range:
            sigma_low, xi1, sigma1, is_cqt1, xi2, sigma2, is_cqt2 = \
                calibrate_scattering_filters(J, Q, T=2**J)
            # Check that all sigmas are > 0
            assert sigma_low > 0
            for sig in sigma1:
                assert sig > 0
            for sig in sigma2:
                assert sig > 0
            # check that sigma_low is smaller than all sigma2
            for sig in sigma1:
                assert sig >= sigma_low
            for sig in sigma2:
                assert sig >= sigma_low

    with pytest.raises(ValueError) as ve:
        calibrate_scattering_filters(J_range[0], 0.9, T=2**J_range[0])
    assert "should always be >= 1" in ve.value.args[0]


def test_compute_xi_max():
    """
    Tests that 0.25 <= xi_max(Q) <= 0.5, whatever Q

    From `tests/scattering1d/test_filters_scattering1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    Q_range = np.arange(1, 21, dtype=int)
    for Q in Q_range:
        xi_max = compute_xi_max(Q)
        assert xi_max <= 0.5
        assert xi_max >= 0.25


def test_sigma0():
    """Ensure `sigma0` is such that `gauss_1d(sigma=sigma0/T)` can be losslessly
    subsampled by `T` according to `criterion_amplitude`.

    Include cases where `gauss_1d` has insufficient temporal decay.
    """
    # get sigma0 and remove duplicate
    sigma0s = {nm: CFG[nm]['sigma0'] for nm in ('S1D', 'JTFS')}
    if sigma0s['S1D'] == sigma0s['JTFS']:
        del sigma0s['JTFS']

    for name, sigma0 in sigma0s.items():
        for N in (16, 32, 64, 128, 256, 512, 1024, 2048, 65536):
            for log2_T in range(1, int(np.log2(N)) + 1):
                T = 2**log2_T
                phi = gauss_1d(N, sigma=sigma0/T)

                bw = compute_bandwidth(phi)
                # `-1` to exclude peak, `/2` to take half, assuming symmetry
                bw_positive = int(np.ceil((bw - 1) / 2))
                if N == T:
                    # special case: the *total* bandwidth must be 1,
                    # but the method returns right and left (right_only etc)
                    # that sum to total in excess by 1. so N/2/T for one-sided
                    # is still 'correct', but yields "0.5 samples"
                    bound = 1
                else:
                    bound = N/2/T
                assert bw_positive <= bound, (
                    name, sigma0, bw_positive, bound, N, T)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_adaptive_choice_P()
        test_fold_filter_fourier()
        test_get_normalizing_factor()
        test_morlet_1d()
        test_gauss_1d()
        test_calibrate_scattering_filters()
        test_compute_xi_max()
        test_sigma0()
    else:
        pytest.main([__file__, "-s"])
