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
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

import wavespin
from wavespin import visuals as v
from wavespin import toolkit as tkt
from wavespin import TimeFrequencyScattering1D, Scattering1D
from wavespin.scattering1d.filter_bank import gauss_1d
from utils import FORCED_PYTEST, cant_import, tempdir

# set True to execute all test functions without pytest
run_without_pytest = 1


# pre-imports
if not cant_import('torch'):
    import torch
else:
    torch = None
if not cant_import('tensorflow'):
    import tensorflow as tf
else:
    tf = None


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

    # `validate_filterbank` ##################################################
    # sines detection: make arbitrary sines
    psi_fs = [np.zeros(64) for _ in range(8)]
    for i in range(len(psi_fs)):
        psi_fs[i][i + 1] = 1
    report = tkt.validate_filterbank(psi_fs, verbose=0)
    assert len(report['sine']) == len(psi_fs), report['sine']

    # `Decimate` #############################################################
    x = np.random.randn(256 - 32)
    xp = np.pad(x, 16)

    # test padding & frequential input for all dtypes
    for dtype in (None, 'float32', 'float64'):
        dec = tkt.Decimate(dtype=dtype)

        o0 = dec(x,  4)
        o1 = dec(xp, 4)
        assert np.allclose(o0, o1[4:-4])

        o0 = dec(fft(x),  4, x_is_fourier=True)
        o1 = dec(fft(xp), 4, x_is_fourier=True)
        assert np.allclose(o0, o1[4:-4])


def test_visuals():
    # `plot` #################################################################
    v.plot(np.array([1, 2]) + 1j, complex=2)
    v.plot(np.array([1, 2]) + 1j, abs=1)

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

    # `make_gif` #############################################################
    with tempdir() as savedir:
        plt.plot([1, 2])
        plt.savefig(os.path.join(savedir, 'im0.png'))
        plt.savefig(os.path.join(savedir, 'im1.png'))
        plt.close()

        savepath = os.path.join(savedir, 'ims.gif')
        ckw = dict(loaddir=savedir, savepath=savepath, start_end_pause=1,
                   ext='.png', delimiter='im', HD=0)
        v.make_gif(**ckw, delete_images=False)
        v.make_gif(**ckw, delete_images=True, overwrite=True)

    # `make_jtfs_pair` #######################################################
    _ = v.make_jtfs_pair(32, N_time=16)


def test_core():
    for vectorized in (True, False):
        _test_core(vectorized)

def _test_core(vectorized):
    # `scattering1d` #########################################################
    # `do_U_1_hat = False`
    N = 256
    x = np.random.randn(N)
    sc = Scattering1D(N, average=0, max_order=1, vectorized=vectorized,
                      out_type='list')
    _ = sc(x)

    # `pad_mode` as function
    pad_fn = lambda x, pad_left, pad_right: np.pad(
        x, [[0, 0], [0, 0], [pad_left, pad_right]])
    sc = Scattering1D(N, pad_mode=pad_fn, vectorized=vectorized)
    _ = sc(x)

    # `timefrequency_scattering1d` ###########################################
    # `k1 != k1_avg`, need `log2_T < j1`
    J = int(np.log2(N)) - 1
    T = 2**(J - 2)
    # also bundle in `do_energy_correction`
    for do_ec in (True, False):
        jtfs = TimeFrequencyScattering1D(
            N, T=T, J=J, vectorized=vectorized, average=0, out_type='list',
            do_energy_correction=do_ec)
        _ = jtfs(x)


def test_gen_utils():
    gu = wavespin.utils.gen_utils

    # `fill_default_args` ####################################################
    cfg = dict(a=1, b=dict(c=3))
    defaults = dict(a=1, b=dict(c=3, d=4))
    out = gu.fill_default_args(cfg, defaults)
    assert out['b']['d'] == defaults['b']['d'], (out, defaults)

    # `print_table` ##########################################################
    # fill missing headers; ensure same number of rows
    col0 = {'a': [1, 2]}
    col1 = [3]
    _ = gu.print_table(col0, col1, show=False)

    # `ExtendedUnifiedBackend` ###############################################
    B = gu.ExtendedUnifiedBackend('numpy')
    x = np.ones(12)
    _ = B.norm(x, ord=3)
    _ = B.median(x)

    if not cant_import('torch'):
        B = gu.ExtendedUnifiedBackend('torch')
        _ = B.min(torch.randn(5), axis=0, keepdims=True)
        x = torch.randn(5)
        x.requires_grad = True
        _ = B.numpy(x)

    if not cant_import('tensorflow'):
        B = gu.ExtendedUnifiedBackend('tensorflow')
        _ = B.sum(tf.random.normal((5, 1)))

    # other backend-related ##################################################
    if not cant_import('jax'):
        import jax
        x = jax.numpy.array([1.])
        _ = gu._infer_backend(x)
        _ = gu.get_wavespin_backend('jax')

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

    if not cant_import('torch'):
        x = [torch.tensor([1])]
        out = gu.npy(x)
        assert isinstance(out, np.ndarray), type(out)
        assert out.ndim == 2, out.shape
        assert np.allclose(out, 1)

    x = 5.
    out = gu.npy(x)
    assert not isinstance(out, np.ndarray)
    assert out == x


def test_scat_utils():
    # su = wavespin.scattering1d.scat_utils

    # `_handle_smart_paths` ##################################################
    _ = Scattering1D(128, smart_paths=(.01, 1))


def test_measures():
    ms = wavespin.utils.measures

    # `compute_spatial_width` ################################################
    # `not complete_decay and not fast`
    sigma0 = .14
    pf = gauss_1d(128, sigma0 / 64)
    _ = ms.compute_spatial_width(pf, sigma0=sigma0, fast=False)

    # `compute_analyticity` ##################################################
    _ = ms.compute_analyticity(np.ones(1), is_time=True)
    _ = ms.compute_analyticity(np.ones(2))
    _ = ms.compute_analyticity(np.zeros(3))


def test_backends():
    # numpy ##################################################################
    sc = Scattering1D(128)
    with pytest.raises(TypeError) as e:
        _ = sc(None)
    assert "not empty" in e.value.args[0]

    # torch ##################################################################
    if not cant_import('torch'):
        from wavespin.backend import torch_backend
        import wavespin.torch

        B = torch_backend.TorchBackend
        _ = B.sqrt(torch.tensor([1.]), dtype=torch.float32)
        _ = B.reshape(torch.arange(6), (2, 3))
        _ = B.try_squeeze(torch.arange(6))
        _ = B.try_squeeze(torch.arange(6), axis=0)

    # tensorflow #############################################################
    if not cant_import('tensorflow'):
        import tensorflow as tf
        from wavespin.backend import tensorflow_backend
        import wavespin.tensorflow

        B = tensorflow_backend.TensorFlowBackend
        _ = B.sqrt(tf.constant([1.]), dtype=tf.float32)
        _ = B.reshape(tf.range(6), (2, 3))

        x = tf.constant([1.])
        _ = B.assign_slice(x, tf.constant([2.]), [0])
        _ = B.assign_slice(x, tf.constant([2.]), range(0, 1))
        _ = B.assign_slice(tf.expand_dims(x, -1), tf.constant([2.]), [0])
        _ = B.try_squeeze(x)
        _ = B.try_squeeze(x, axis=0)

    # jax ####################################################################
    if not cant_import('jax'):
        import jax
        from wavespin.backend import jax_backend
        import wavespin.jax

# run tests ##################################################################
if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_configs()
        test_toolkit()
        test_visuals()
        test_core()
        test_gen_utils()
        test_measures()
        test_backends()
    else:
        pytest.main([__file__, "-s"])
