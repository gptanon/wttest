# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Miscellaneous tests to increase coverage of meaningful lines, others being
omitted by `# no-cov` in `.coveragerc`.
"""
import pytest
import numpy as np
import torch
from scipy.fft import fft

import wavespin
from wavespin import visuals as v
from wavespin import toolkit as tkt
from wavespin import TimeFrequencyScattering1D, Scattering1D
from wavespin.scattering1d.filter_bank import gauss_1d
from utils import FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1

def test_configs():
    # alter `CFG` and ensure the file's dict wasn't altered
    CFG = wavespin.configs.get_defaults(library=False)
    CFG['x'] = 5
    assert 'x' not in wavespin.configs._USER_DEFAULTS

    # simply run
    wavespin.configs.restore_defaults()


def test_toolkit():
    # `meta2coeff_jtfs` ######################################################
    jtfs = TimeFrequencyScattering1D(256)
    x = np.random.randn(jtfs.N)
    Scx = jtfs(x)
    jmeta = jtfs.meta()

    meta_goal = {'xi': [None, None, .25], 'j': [1, 2, None]}
    tkt.meta2coeff_jtfs(Scx, jmeta, meta_goal, pair=None)

    # `l1`, `rel_l1`, `rel_ae` ###############################################
    x = np.random.randn(64)
    _ = tkt.l1(x)
    _ = tkt.rel_l1(x, x - 1)
    _ = tkt.rel_ae(x, x - 1, ref_both=False)

    # `bag_o_waves` ##########################################################
    _ = tkt.bag_o_waves(128)


def test_visuals():
    # `imshow` ###############################################################
    v.imshow(np.random.randn(4, 4), borders=0)

    # `scat` #################################################################
    v.scat(None, np.random.randn(4), abs=1, vlines=1, hlines=(1, {}), title="x")

    # `plotscat` #############################################################
    v.plotscat(np.random.randn(4), show=1, xlims=(0, 1), do_gscale=1)

    # `energy_profile_jtfs` ##################################################
    N = 128
    x = np.random.randn(N)
    jtfs = TimeFrequencyScattering1D(N, out_type='dict:list')
    jmeta = jtfs.meta()
    Scx = jtfs(x)

    _, pair_energies_a = v.energy_profile_jtfs(Scx, jmeta, x=x)


def test_core():
    # `scattering1d` #########################################################
    # `do_U_1_hat = False`
    N = 256
    x = np.random.randn(N)
    sc = Scattering1D(N, average=0, max_order=1, vectorized=0, out_type='list')
    _ = sc(x)


def test_gen_utils():
    gu = wavespin.utils.gen_utils

    # `is_real` ##############################################################
    xr = np.random.randn(15)
    xi = xr + 1j
    assert gu.is_real(xr, is_time=True)
    assert gu.is_real(fft(xr), is_time=False)
    assert not gu.is_real(xi, is_time=True)
    assert not gu.is_real(fft(xi), is_time=False)

    # `print_table` ##########################################################
    # `has_headers = True`
    _ = gu.print_table({'a': [1, 2]}, {'b': [3, 4]})

    # `npy` ##################################################################
    x = [1, 2]
    out = gu.npy(x)
    assert isinstance(out, np.ndarray), type(out)
    assert np.allclose(out, x)

    x = [torch.tensor([1])]
    out = gu.npy(x)
    assert isinstance(out, np.ndarray), type(out)
    assert out.ndim == 2, out.shape
    assert np.allclose(out, 1)

    x = 5.
    out = gu.npy(x)
    assert not isinstance(out, np.ndarray)
    assert out == x


def test_measures():
    ms = wavespin.utils.measures

    # `compute_spatial_width` ################################################
    # `not complete_decay and not fast`
    sigma0 = .14
    pf = gauss_1d(128, sigma0 / 64)
    _ = ms.compute_spatial_width(pf, sigma0=sigma0, fast=False)


# run tests ##################################################################
if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_configs()
        test_toolkit()
        test_visuals()
        test_core()
        test_gen_utils()
        test_measures()
    else:
        pytest.main([__file__, "-s"])
