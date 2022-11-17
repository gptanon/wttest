# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Tests related to discrete measures (support, bandwidth, etc)."""
import warnings
import pytest
import numpy as np
from scipy.fft import fft, ifft

from wavespin.scattering1d.filter_bank import morlet_1d, gauss_1d
from wavespin.utils.measures import (
    compute_spatial_support, compute_spatial_width,
    compute_bandwidth, compute_bw_idxs, compute_max_dyadic_subsampling)
from wavespin.toolkit import fft_upsample
from wavespin import CFG
from utils import FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 0
# will run most tests with this backend
default_frontend = ('numpy', 'torch', 'tensorflow')[0]

prec_adj = np.finfo(np.float64).eps * 10

#### Measures tests ##########################################################
def test_compute_spatial_support():
    """Test support computation on a flat line, and Gaussian."""
    # flat line ##############################################################
    N = 16
    for guarantee_decay in (False, True):
        for supp_spec in range(2, N + 1, 2):
            pt = np.zeros(N)
            pt[:supp_spec//2] = 1
            pt[-supp_spec//2:] = 1
            pf = fft(pt)
            supp_computed = compute_spatial_support(
                pf, guarantee_decay=guarantee_decay)
            assert supp_computed == supp_spec, (supp_computed, supp_spec,
                                                guarantee_decay)

    # gaussian ###############################################################
    # complete decay requires N >= 16*T
    p0f = gauss_1d(256, CFG['S1D']['sigma0'] / 8)
    p1f = gauss_1d(256, CFG['S1D']['sigma0'] / 16)
    s0, s1 = compute_spatial_support(p0f), compute_spatial_support(p1f)
    # discard the center (n=0), which isn't doubled
    s0_adj, s1_adj = s0 - 1, s1 - 1
    # set tolerance as doubled sigma spec may not necessarily yield doubled output
    tol = .025
    th = 2 * s0_adj * (1 - tol)
    assert s1_adj >= th, (s1_adj, th, tol)


def test_compute_spatial_width():
    """Tests that `compute_spatial_width` works as intended."""
    # library defaults
    sigma0 = CFG['S1D']['sigma0']
    criterion_amplitude = 1e-3
    complete_decay_factor = 16  # follows from above

    J_pad = 9
    filter_len = 2**J_pad
    pts_per_scale = 6
    # don't allow underestimating by more than this
    th_undershoot = -1
    # consider `T` above this as close to global averaging
    T_global_avg_earliest = int(.6 * filter_len // 2)
    T_global_avg_latest = int(.8 * filter_len // 2)

    Ts = np.arange(2, 256)
    # test for different input sizes relative to filter sizes
    for N in (filter_len, filter_len // 2, filter_len // 4):
        T_ests = []
        for T in Ts:
            phi_f = gauss_1d(filter_len, sigma=sigma0/T)
            if T > N // 2:
                break
            T_est = compute_spatial_width(
                phi_f, N, sigma0=sigma0, criterion_amplitude=criterion_amplitude,
                pts_per_scale=pts_per_scale)
            T_ests.append(T_est)
        Ts = Ts[:len(T_ests)]
        T_ests = np.array(T_ests)

        for (T, T_est) in zip(Ts, T_ests):
            test_params_str = 'T={}, N={}, T_est={}'.format(T, N, T_est)

            # check global averaging cases
            if N == filter_len:
                if T_est == N:
                    assert T >= T_global_avg_earliest, "{} < {} | {}".format(
                        T, T_global_avg_earliest, test_params_str)
                elif T >= T_global_avg_latest:
                    assert T_est == N, "{} != {} | {}".format(
                        T_est, N, test_params_str)
            elif T == Ts[-1]:
                # last is max
                assert T_est == T_ests.max(), "{} != {} | {}".format(
                    T_est, T_ests.max(), test_params_str)

            # check other cases
            complete_decay = bool(T <= filter_len // complete_decay_factor)
            if complete_decay:
                # must match perfectly
                assert T_est == T, "{} != {} | {}".format(
                    T_est, T, test_params_str)
            else:
                assert T_est - T > th_undershoot, "{} - {} <= {} | {}".format(
                    T_est, T, th_undershoot, test_params_str)

    # Agreement of `fast=True` with `False` for complete decay
    # also try non-default `sigma0`
    N = 256
    for T in range(1, 16):
        p_f = gauss_1d(N, sigma=sigma0 / T)
        w0 = compute_spatial_width(p_f, sigma0=sigma0, fast=True)
        w1 = compute_spatial_width(p_f, sigma0=sigma0, fast=False)
        assert w0 == w1 == T, (w0, w1, T)

        for sigma0_different in (sigma0/2, sigma0*2):
            p_f = gauss_1d(N, sigma=sigma0_different / T)
            # need greater accuracy for lesser sigma
            pts_per_scale = 6 if sigma0_different > sigma0 else 20
            w = compute_spatial_width(p_f, sigma0=sigma0_different, fast=False,
                                      pts_per_scale=pts_per_scale)
            assert w == T, (w, T, sigma0_different)


def test_compute_bandwidth():
    """Tests `compute_bandwidth` for a wide variety of waveforms, including
    and beyond Morlets.

    Also tests `compute_bw_idxs`.
    """
    # energy func
    E = lambda x: np.sum(np.abs(x)**2)

    N = 128
    pf = morlet_1d(N, 16 * 1.4 / N, 16 * .32 / N)
    c = np.argmax(pf)  # peak index
    # ensure sufficient assymetry
    left_right_r = abs(1 - pf[c+1] / pf[c-1])
    assert left_right_r > .01, left_right_r

    # reusable
    e_total = E(pf)
    e_ratio = lambda pf_slc: E(pf_slc) / (e_total - E(pf_slc))

    # CASE 1: peak only, bw=1 ################################################
    # ratio that should yield bw=1: E_max / (E_total - E_max)
    bw1_r = e_ratio(pf.max())
    # account for `criterion_ratio` and `measure='energy'`, and
    # make required ratio a bit smaller per numeric imprecision uncertainty
    adjup, adjdn = 1 + prec_adj, 1 - prec_adj
    bw1_r_re = 1 / np.sqrt(bw1_r)
    bw = compute_bandwidth(pf, criterion_amplitude=bw1_r_re*adjup)
    assert bw == 1, bw
    # making it a bit bigger should now yield 2
    bw = compute_bandwidth(pf, criterion_amplitude=bw1_r_re*adjdn)
    assert bw == 2, bw

    # CASE 2: one-sided bw, left and right of peak ###########################
    bw1_r_righte = e_ratio(pf[c-1:c+1])
    bw1_r_lefte  = e_ratio(pf[c:c+2])
    bw1_r_righte_re = 1 / np.sqrt(bw1_r_righte)
    bw1_r_lefte_re  = 1 / np.sqrt(bw1_r_lefte)

    bw_righte_l, bw_righte_r = compute_bandwidth(
        pf, return_sided=True, criterion_amplitude=bw1_r_righte_re*adjup)
    bw_lefte_l, bw_lefte_r = compute_bandwidth(
        pf, return_sided=True, criterion_amplitude=bw1_r_lefte_re*adjup)
    assert bw_righte_l == 2, bw_righte_l
    assert bw_righte_r == 1, bw_righte_r
    assert bw_lefte_l == 1, bw_lefte_l
    assert bw_lefte_r == 2, bw_lefte_r

    # CASE 3: broad bw / constant ############################################
    N = 16
    for guarantee_decay in (False, True):
        for bw_spec in range(2, N + 1, 2):
            pf = np.zeros(N)
            pf[:bw_spec] = 1
            # unbounded `shift` is tested in `test_compute_bw_idxs`
            for shift in range(N - bw_spec + 1):
                pfr = np.roll(pf, shift)
                # must specify "center" manually, argmax is ambiguous
                c = bw_spec//2 + shift
                bw_computed = compute_bandwidth(
                    pfr, guarantee_decay=guarantee_decay, c=c)
                assert bw_computed == bw_spec, (
                    bw_computed, bw_spec, guarantee_decay, shift)

    # CASE 4: aliased Morlet (excess bw) #####################################
    N = 128
    pf = morlet_1d(N, .45, .2)

    # bw #######################################
    bw_left, bw_right = compute_bandwidth(pf, return_sided=True)
    assert bw_left == N//2 + 1, (bw_left, N//2 + 1)
    assert bw_right == N//2, (bw_right, N//2)

    # bw_idxs ##################################
    idx_left, idx_right = compute_bw_idxs(pf)
    # note `c_computed` in the method, and the fact that each side's bw is maxed
    idx_left_expected = -(N//2 - np.argmax(pf))
    assert idx_left == idx_left_expected, (idx_left, idx_left_expected)

    # idx_right should be such that `pf[idx_left:idx_right + 1]` slices everything
    idx_right_expected = N + idx_left - 1
    assert idx_right == idx_right_expected, (idx_right, idx_right_expected)

    # CASE 5: aliased Morlet (excess bw), anti-analytic ######################
    N = 128
    pf = morlet_1d(N, .45, .2)
    pf[1:] = pf[1:][::-1]

    # bw #######################################
    bw_left, bw_right = compute_bandwidth(pf, return_sided=True)
    assert bw_left == N//2 + 1, (bw_left, N//2 + 1)
    assert bw_right == N//2, (bw_right, N//2)

    # bw_idxs ##################################
    idx_left, idx_right = compute_bw_idxs(pf)
    # note `c_computed` in the method, and the fact that each side's bw is maxed.
    # also "left" is to left of peak, and sum that wraps around has negative
    # left index
    idx_left_expected = -(np.argmax(pf) - N//2)
    # wrap around
    idx_left_expected = -((N - 1) + idx_left_expected)
    assert idx_left == idx_left_expected, (idx_left, idx_left_expected)

    # idx_right should be such that `pf[idx_left:idx_right + 1]` slices everything
    idx_right_expected = N + idx_left - 1
    assert idx_right == idx_right_expected, (idx_right, idx_right_expected)

    # CASE 6: leaky Morlet (low freq) ########################################
    pf = morlet_1d(N, .02, .01)

    # measure first point to drop below a surrogate (representative of bw compute)
    # threshold
    apf = np.abs(pf)
    c = np.argmax(apf)
    ca_default = CFG['S1D']['criterion_amplitude']
    decayed_neg_idx = np.where(apf[::-1][:N//2] / apf.max() < ca_default)[0][0]
    bw_left_expected = c + decayed_neg_idx

    # bw #######################################
    bw_left, bw_right = compute_bandwidth(pf, return_sided=True)
    assert bw_left == bw_left_expected, (bw_left, bw_left_expected)
    # bw_right should be no more than bw_left+1
    assert bw_right <= bw_left + 1, (bw_right, bw_left + 1)

    # bw_idxs ##################################
    idx_left, idx_right = compute_bw_idxs(pf)
    # assert both reflect same total bandwidth; note, indices are inclusive,
    # and bws double-count the peak
    assert abs(idx_left) + idx_right + 1 == bw_left + bw_right - 1, (
        idx_left, idx_right, bw_left, bw_right)

    assert idx_left == -(decayed_neg_idx - 1), (idx_left, -(decayed_neg_idx - 1))

    # CASE 7: strictly analytic Morlet #######################################
    # plant peak at center, which means it's actually at center - 1 due to
    # Nyquist halving
    pf = morlet_1d(N, .5, .2)
    pf[N//2 + 1:] = 0
    pf[N//2] /= 2

    bw_left, bw_right = compute_bandwidth(pf, return_sided=True)
    # we don't expect it at N//2+1 since DC bin is zero and pf no longer spills
    # over negatives, and `c_compute` shouldn't change this.
    # we also don't expect it at N//2 since the peak is at N//2-1.
    # we also don't expect it at N//2-1 per Morlet's sufficiently rapid decay.
    # but we do expect it very close to N//2, so we just say within 6 samples
    # (not verified for all N)
    th_low, th_high = N//2 - 6, N//2 - 1
    assert th_low < bw_left < th_high, (th_low, bw_left, th_high)
    assert bw_right == 2, bw_right

    # CASE 8: need for `guarantee_decay` #####################################
    # L2 takes much more to fail than L1, but still finite; loosen
    # `criterion_amplitude` so the test is faster
    ca_test = .1
    # bw computes as `E(inside)/E(outside) > ratio`, where `ratio = 1/ca`.
    # For `measure='abs'`, it's `sum(inside)/sum(outside)` which for constant
    # is `len(inside)/len(outside)`, so if `len(total) = ca + 1` then the
    # longest `inside` is `10` and our ratio is `10/1`, but this doesn't
    # *exceed* `ca=10`, so the length must be `ca + 2`. For a constant line
    # `E(interval) == sum(interval)`, so this part is the same with
    # measure='energy', but `ratio` is now squared, hence `1/ca**2 + 2`.
    fail_length = int(np.ceil(1 / ca_test**2)) + 2
    N = fail_length
    pf = np.ones(N)

    ckw = dict(pf=pf, criterion_amplitude=ca_test)
    bw_no_guarantee = compute_bandwidth(**ckw, guarantee_decay=False)
    bw_guarantee    = compute_bandwidth(**ckw, guarantee_decay=True)
    assert bw_no_guarantee < N, (bw_no_guarantee, N)
    assert bw_guarantee == N, (bw_guarantee, N)


def test_compute_bw_idxs():
    """Additional test cases, including anti-analytic."""
    # CASE 1: one-sided bw, left and right of peak ###########################
    E = lambda x: np.sum(np.abs(x)**2)

    N = 128
    pf = morlet_1d(N, 16 * 1.4 / N, 16 * .32 / N)
    c = np.argmax(pf)  # peak index
    # ensure sufficient assymetry
    left_right_r = abs(1 - pf[c+1] / pf[c-1])
    assert left_right_r > .01, left_right_r

    # reusable
    e_total = E(pf)
    e_ratio = lambda pf_slc: E(pf_slc) / (e_total - E(pf_slc))
    bw1_r_righte = e_ratio(pf[c-1:c+1])
    bw1_r_lefte  = e_ratio(pf[c:c+2])
    bw1_r_righte_re = 1 / np.sqrt(bw1_r_righte)
    bw1_r_lefte_re  = 1 / np.sqrt(bw1_r_lefte)

    adjup = 1 + prec_adj
    bil, bir = compute_bw_idxs(pf, criterion_amplitude=bw1_r_righte_re*adjup)
    assert bil == (c - 1), (bil, c - 1)
    assert bir == c, (bir, c)

    bil, bir = compute_bw_idxs(pf, criterion_amplitude=bw1_r_lefte_re*adjup)
    assert bil == c, (bil, c)
    assert bir == (c + 1), (bir, c + 1)

    # CASE 2: leaky Morlets, low & high bw, analytic & anti-analytic #########
    def idxs_to_sum(pf, l, r):
        if l < 0:
            # circular wrap-around sum
            sm = 0
            sm += pf[l:].sum()
            sm += pf[:r + 1].sum()
        else:
            sm = pf[l:r + 1].sum()
        return sm

    # make analytic & anti-analytic
    N = 128
    pf0 = morlet_1d(N, .45, .2)
    pf1 = morlet_1d(N, .02, .01)
    pf0a, pf1a   = pf0.copy(), pf1.copy()
    pf0aa, pf1aa = pf0.copy(), pf1.copy()
    pf0aa[1:], pf1aa[1:] = pf0aa[1:][::-1], pf1aa[1:][::-1]

    # analytic
    l0a, r0a = compute_bw_idxs(pf0a)
    l1a, r1a = compute_bw_idxs(pf1a)
    sm0a, sm1a = idxs_to_sum(pf0a, l0a, r0a), idxs_to_sum(pf1a, l1a, r1a)
    # anti-analytic
    l0aa, r0aa = compute_bw_idxs(pf0aa)
    l1aa, r1aa = compute_bw_idxs(pf1aa)
    sm0aa, sm1aa = idxs_to_sum(pf0aa, l0aa, r0aa), idxs_to_sum(pf1aa, l1aa, r1aa)

    # assert equality
    assert np.allclose(sm0a, sm0aa), (sm0a, sm0aa, l0a, r0a, l0aa, r0aa)
    assert np.allclose(sm1a, sm1aa), (sm1a, sm1aa, l1a, r1a, l1aa, r1aa)

    # CASE 3: extreme bw #####################################################
    N = 128
    pf0 = morlet_1d(N, .99, .99)
    pf1 = pf0.copy()
    pf1[1:] = pf1[1:][::-1]

    l0, r0 = compute_bw_idxs(pf0)
    l1, r1 = compute_bw_idxs(pf1)

    sm0, sm1 = idxs_to_sum(pf0, l0, r0), idxs_to_sum(pf1, l1, r1)
    assert np.allclose(sm0, sm1), (sm0, sm1)
    assert np.allclose(sm0, pf0.sum()), (sm0, pf0.sum())

    # CASE 4: flat line ######################################################
    # note, does not test very long flat line, N > (1/criterion_amplitude)**2
    N = 128
    pf = np.ones(N)
    l, r = compute_bw_idxs(pf)
    sm = idxs_to_sum(pf, l, r)
    assert np.allclose(sm, pf.sum()), (sm, pf.sum())

    # CASE 5: impulse ########################################################
    for idx in (0, -1):
        pf = np.zeros(N)
        pf[idx] = 1
        l, r = compute_bw_idxs(pf)
        sm = idxs_to_sum(pf, l, r)
        assert np.allclose(sm, pf.sum()), (sm, pf.sum(), idx)

    # CASE 6: randn ##########################################################
    np.random.seed(0)
    pf = np.random.randn(N)
    l, r = compute_bw_idxs(pf)
    sm = idxs_to_sum(pf, l, r)
    assert np.allclose(sm, pf.sum()), (sm, pf.sum())

    # CASE 7: constant shiftings, intended use ###############################
    N = 37
    for bw_spec in range(2, N + 1, 2):
        pf = np.zeros(N)
        pf[:bw_spec] = 1
        for shift in range(N - bw_spec + 1):
            pfr = np.roll(pf, shift)
            # must specify "center" manually, argmax is ambiguous
            c = bw_spec//2 + shift
            l, r = compute_bw_idxs(pfr, c=c)
            sm = idxs_to_sum(pfr, l, r)
            assert np.allclose(sm, pf.sum()), (sm, pf.sum(), l, r, shift)

    # CASE 8: constant shiftings, all ########################################
    # with properly defined `c` this must work, but it's largely untested
    # territory and against intended use past certain `bw` per ambiguity;
    # see docs. for example it fails if `c` isn't well-centered on `pfr` yet
    # it shouldn't, but we keep this test since a well-centered `pfr` must pass.
    # "well-centered" means total contiguity in `_integral_ratio_bound`.
    N = 16
    for bw_spec in range(1, N + 1, 1):
        pf = np.zeros(N)
        pf[:bw_spec] = 1
        for shift in range(N):
            pfr = np.roll(pf, shift)
            c0 = bw_spec//2 + shift
            c = c0 - N if c0 >= N else c0

            bw_computed0 = compute_bandwidth(pfr, c=c)
            bw_idxs = compute_bw_idxs(pfr, c=c)
            bw_computed1 = bw_idxs[1] - bw_idxs[0] + 1

            info = (bw_computed0, bw_computed1, bw_spec, shift)
            assert bw_computed0 == bw_computed1 == bw_spec, info


def test_compute_max_dyadic_subsampling_and_fft_upsample():
    """Test max subsampling, and upsampling, with exact non-aliasing.
    Specifically, tests that

      - if non-zero subsampling is predicted, that perfect recovery is possible.
      - if zero subsampling is predicted, that the upsampling function fails
      - "nonzero Nyquist == aliasing" is handled with warnings and failed
        upsamplings
      - strictly analytic or anti-analytic waveforms are treated differently
        by both methods
      - methods can correctly detect strictly analytic or anti-analytic waveforms,
        and if they can't, it's only because it cannot be done
    """
    def _test_fn(xf, j, info, xf_analytic, assert_false=False):
        # compute original in time
        xt = ifft(xf)
        # compute downsampled in time, but without going to time since we
        # go back to freq right away for next step
        xfd = xf.reshape(2**j, -1).mean(axis=0)
        # test auto-determination
        if xf_analytic is not None:
            analytic      = xf_analytic
            anti_analytic = not xf_analytic
        else:
            analytic = anti_analytic = None

        try:
            # FFT-upsample
            xfdu = fft_upsample(xfd, factor=2**j, analytic=analytic,
                                anti_analytic=anti_analytic)
            xtu = ifft(xfdu)
            if assert_false:
                assert not np.allclose(xt, xtu)
            else:
                assert np.allclose(xt, xtu)
        except Exception as e:
            print(info)
            try: print("xf.shape={}, xfd.shape={}, xfdu.shape={}".format(
                xf.shape, xfd.shape, xfdu.shape))
            except: pass
            raise e

    # CASE 1: complex, one-sided case, leap over to other side also ##########
    np.random.seed(0)
    N = 32
    xf_full = np.random.randn(N) + 1j*np.random.randn(N)
    # `compute_bw_idxs` does not account for the case where most of bandwidth
    # is on left but peak is on right, and vice versa, so avoid risk
    # (defies "well-centered", see note in `test_compute_bw_idxs`)
    xf_full[1] *= np.abs(xf_full).max() / abs(xf_full[1]) * 1.01

    for bw in range(1, N + 1):
        xf = np.zeros(N, dtype=xf_full.dtype)
        xf[:bw] = xf_full[:bw]

        # restrict `shift` to avoid accounting for negative `bw_idxs` in test code
        # also see "well-centered" note in `test_compute_bw_idxs`.
        # we test both-sided xf in CASE 2.
        for shift in range(N - bw + 1):
            xfr = np.roll(xf, shift)
            bw_idxs = compute_bw_idxs(xfr)

            ckw = dict(pf=xfr, bw_idxs=bw_idxs)
            j    = compute_max_dyadic_subsampling(**ckw, for_scattering=False)
            j_sc = compute_max_dyadic_subsampling(**ckw, for_scattering=True)

            # see "Note: information bandwidth" in compute_max_dyadic_subsampling
            if shift == 0:
                div = N // 2 / (bw - 1) if bw > 2 else N / bw
            else:
                bi0, bi1 = bw_idxs
                if bi1 >= bi0 >= N//2:
                    bi0, bi1 = N - bi0, N - bi1
                bmax = max(bi0, bi1)
                div = N // 2 / bmax if bmax > 1 else N / (bmax + 1)
            div_sc = div
            j_expected    = int(np.log2(div))
            j_sc_expected = int(np.log2(div_sc))

            # assert j and recovery
            info = "bw={}, bw_idxs={}, j={}, shift={}".format(
                bw, bw_idxs, j, shift)
            assert j    == j_expected,    (j,    j_expected,    info)
            assert j_sc == j_sc_expected, (j_sc, j_sc_expected, info)

            # determine whether to pass in information about analyticity
            # or allow `fft_upsample` to determine it. we pass in when
            # determination is impossible.
            len_xfd = N//2**j
            if (bw == 1 and abs(bw_idxs[0]) in (N//2//2**j, N - N//2//2**j)):
                # subsampled is only Nyquist but original isn't, must specify
                xf_analytic = True if bw_idxs[0] <= N//2 else False
            elif len_xfd <= 2:
                # same story but special case
                xf_analytic = True
            else:
                # can determine
                xf_analytic = None

            if j > 0:
                _test_fn(xfr, j, info, xf_analytic=xf_analytic)
            else:
                with warnings.catch_warnings(record=True) as ws:
                    _test_fn(xfr, 1, info, xf_analytic=xf_analytic,
                             assert_false=True)
                # 1: subsampled includes Nyquist; 2: original crosses Nyquist
                if (N//4 in np.arange(*bw_idxs) and
                        bw_idxs[0] < N//2 and bw_idxs[1] > N//2):
                    assert any("Non-zero Nyquist" in str(w.message)
                               for w in ws), info

    # CASE 2: complex, two-sided case ########################################
    np.random.seed(0)
    N = 32
    xf_full = np.random.randn(N) + 1j*np.random.randn(N)
    for bw in range(1, N + 1):
        bl = int(np.ceil(bw / 2))
        br = bw - bl
        assert bl + br == bw, (bl, br, bw)

        xf = np.zeros(N, dtype=xf_full.dtype)
        xf[:bl] = xf_full[:bl]
        if br > 0:
            xf[-br:] = xf_full[-br:]

        bw_idxs = compute_bw_idxs(xf)
        # special case at bw==2: it's anti-analytic by coincidence, but
        # fft_upsample doesn't handle it. else leave None to test auto detection
        analytic = False if bw == 2 else None
        ckw = dict(pf=xf, bw_idxs=bw_idxs, analytic=analytic)
        j    = compute_max_dyadic_subsampling(**ckw, for_scattering=False)
        j_sc = compute_max_dyadic_subsampling(**ckw, for_scattering=True)

        info = "bw={}, bw_idxs={}, j={}".format(bw, bw_idxs, j)

        # no Nyquist at bw==1
        div = (N / (bw + 1) if bw > 1 else
               N)
        div_sc = N // 2 / max(bl - 1, br) if bw > 1 else N
        j_expected    = int(np.log2(div))
        j_sc_expected = int(np.log2(div_sc))
        # assert j and recovery
        assert j    == j_expected,    (j,    j_expected,    info)
        assert j_sc == j_sc_expected, (j_sc, j_sc_expected, info)

        xf_analytic = None
        if j > 0:
            _test_fn(xf, j, info, xf_analytic=xf_analytic)
        else:
            with warnings.catch_warnings(record=True) as ws:
                _test_fn(xf, 1, info, xf_analytic=xf_analytic, assert_false=True)
            assert any("Non-zero Nyquist" in str(w.message)
                       for w in ws)


def test_short():
    """N==1, N==2."""
    x = fft(np.ones(1))
    o0 = compute_spatial_width(x, fast=True)
    o1 = compute_spatial_width(x, fast=False)
    assert o0 == o1 == 1, (o0, o1)

    # for N==2 and `fast=True`, only test that it doesn't error, we don't
    # expect accurate results here.
    x = fft(np.ones(2))
    _ = compute_spatial_width(x, fast=True)
    o1 = compute_spatial_width(x, fast=False)
    assert o1 == 2, o1

    x = fft(np.array([1., 0.]))
    _ = compute_spatial_width(x, fast=True)
    o1 = compute_spatial_width(x, fast=False)
    assert o1 == 1, o1


def test_exceptions():
    """Only important ones."""
    with pytest.raises(Exception) as e:
        compute_bandwidth(np.zeros(16))
    assert "all zeros" in e.value.args[0]


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_compute_spatial_support()
        test_compute_spatial_width()
        test_compute_bandwidth()
        test_compute_bw_idxs()
        test_compute_max_dyadic_subsampling_and_fft_upsample()
        test_short()
        test_exceptions()
    else:
        pytest.main([__file__, "-s"])
