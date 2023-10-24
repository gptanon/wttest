# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Tests related to Scattering1D and utilities."""
import pytest
import numpy as np
import scipy.signal

from wavespin import Scattering1D
from wavespin.scattering1d.backend.agnostic_backend import (
    pad, stride_axis, unpad_dyadic, _emulate_get_conjugation_indices)
from wavespin.scattering1d.refining import smart_paths_exclude
from wavespin.toolkit import rel_l1, rel_l2, energy, bag_o_waves, fft_upsample
from utils import cant_import, FORCED_PYTEST, get_wavespin_backend

# set True to execute all test functions without pytest
run_without_pytest = 1
# will run most tests with this backend
default_frontend = ('numpy', 'torch', 'tensorflow', 'jax')[0]
# precision to use for all but precision-sensitive tests
default_precision = 'single'


#### Scattering tests ########################################################
def test_smart_paths():
    """Test that `smart_paths=True` delivers as promised on practical worst-cases.
    """
    def test_x(sc, x, name, e_loss, sp_idx=0):
        # generally we shouldn't change `pad_mode` after creation
        # but here it's fine
        sc.update(pad_mode='reflect' if name != 'impulse' else
                  'zero')

        sc.update(paths_exclude={})
        # sc.paths_exclude = primitive_paths_exclude(sc.psi1_f, sc.psi2_f)
        out_full = sc(x)
        meta_full = sc.meta()
        ckw = dict(psi1_f=sc.psi1_f, psi2_f=sc.psi2_f, e_loss=e_loss)
        sp = smart_paths_exclude(**ckw)
        sc.update(paths_exclude=sp)

        out_smart = sc(x)
        meta_smart = sc.meta()

        out_o2_full  = out_full[meta_full['order'] == 2]
        out_o2_smart = out_smart[meta_smart['order'] == 2]
        eo2_full  = energy(out_o2_full)
        eo2_smart = energy(out_o2_smart)

        o2_preserved_ratio = eo2_smart / eo2_full
        o_preserved_ratio = energy(out_smart) / energy(out_full)
        if DEBUG:
            print(o2_preserved_ratio,
                  o_preserved_ratio,
                  eo2_full / energy(out_full),
                  len(sc.paths_exclude['n2, n1']),
                  len(out_o2_full) - len(out_o2_smart),
                  len(out_o2_full),
                  len(out_o2_smart),
                  )
        return (out_full, out_smart, meta_full, meta_smart, sp,
                out_o2_full, out_o2_smart, eo2_full, eo2_smart,
                o_preserved_ratio)

    # prints debug info
    DEBUG = 0

    # Scattering1D ###########################################################
    N = 2048 + 1
    e_loss = .04
    th = e_loss

    J = int(np.log2(N) - 3)
    np.random.seed(0)
    x = np.random.randn(N)
    x += np.random.uniform(-1, 1, N)**2
    t = np.linspace(0, 1, N, 1)
    a = np.cos(2*np.pi * (N//150) * t)*4
    c = np.cos(2*np.pi * (N//2.4) * t)
    x += a * c

    # generate distinct time-frequency geometries to cover worst cases
    names = ('randn', 'pink', 'impulse', 'adversarial-impulse-train',
             'adversarial-am')
    names_nonadv = [nm for nm in names if not nm.startswith('adversarial')]
    x_all = bag_o_waves(N, names_nonadv)

    for Q in (4, 8, 16, 24):
        ckw = dict(shape=N, J=J, Q=(Q, 1), T='global', out_type='array',
                   max_pad_factor=0, frontend=default_frontend,
                   precision=default_precision)
        sc = Scattering1D(**ckw, smart_paths='primitive', analytic=True)

        if DEBUG:
            print()
        for name in names:
            if name.startswith('adversarial'):
                e_loss_adj = e_loss * 4  # see note in `adversarial_am`
                x = bag_o_waves(N, name, sc=sc, e_th=e_loss_adj)[name]
            else:
                x = x_all[name]
            xs = x if isinstance(x, list) else [x]

            for i, x in enumerate(xs):
                (out_full, out_smart, meta_full, meta_smart, sp,
                 out_o2_full, out_o2_smart, eo2_full, eo2_smart,
                 o_preserved_ratio) = test_x(sc, x, name, e_loss, sp_idx=0)

                test_name = (name if name != 'adversarial-am' else
                             name + f'_{i}')
                assert o_preserved_ratio >= 1 - th, (o_preserved_ratio, th, Q,
                                                     test_name)

            # debug code -----------------------------------------------------
            if DEBUG and 0:
                from wavespin.visuals import plot, imshow
                ikw = dict(abs=1, interpolation='none')
                imshow(out_full[1:np.sum(meta_full['order']==1)+1],  **ikw)
                imshow(out_full[np.sum(meta_full['order']==1)+1:],   **ikw)
                imshow(out_smart[np.sum(meta_smart['order']==1)+1:], **ikw)
            # ----------------------------------------------------------------

        # debug code ---------------------------------------------------------
        if DEBUG and 0:#Q == 24:
            energies = []
            ix2 = np.where(meta_full['order'] == 2)[0]
            ns = meta_full['n'][ix2]
            out_full_o2 = out_full[ix2]

            for i, c in enumerate(out_full_o2):
                n1, n2 = ns[i]
                if 1:#n2 == 3 and n1 >= 7:
                    energies.append(energy(c))
                # print("({}, {}): {:.4f}".format(n2, n1, energy(c)))
            energies = np.array(energies)
            print(energies.sum())
            plot(energies, show=1)
            esum = energies.sum()
            assert np.allclose(esum, eo2_full)

            ##################################################################
            peak1_all = [p['peak_idx'][0] for p in sc.psi1_f]
            bw1_all = [p['bw'][0] for p in sc.psi1_f]
            j1_all = [p['j'] for p in sc.psi1_f]

            ##################################################################
            i = 0
            for n1, p1 in enumerate(sc.psi1_f):
                j1 = p1['j']
                for n2, p2 in enumerate(sc.psi2_f):
                    freq_max2 = p2['bw_idxs'][0][1]
                    peak2 = p2['peak_idx'][0]
                    bin2 = (peak2 + freq_max2) / 2
                    j2 = p2['j']
                    if not (j2 != 0 and j2 > j1):
                        continue

                    peak1, bw1, j1 = [_x[int(n1)] for _x in
                                      (peak1_all, bw1_all, j1_all)]
                    rperc = energies[i] / esum * 100
                    th0 = 4
                    th_effective = th0 / (len(energies) / 3)

                    A = int(not (bin2 < min(peak1, bw1))
                            and j2 >= j1)
                    B = int(rperc < th_effective)

                    if (n2, n1) in sc.paths_exclude['n2, n1'] and not B:
                    # if A and not B:
                        print(("({}, {}): {}, {}, {:.3f} | {} < min({}, {}) "
                                "and {} >= {}"
                                ).format(n2, n1, A, B,
                                         rperc, bin2, peak1, bw1, j2, j1))
                    i += 1
        # --------------------------------------------------------------------


def test_smart_paths_subsetting():
    """Test that output of lower `e_th` subsets higher `e_th`."""
    N = 4096
    J = int(np.log2(N))
    Qs = (1, 8, 16)
    e_loss_max = .5
    n_trials = 20

    ckw = dict(shape=N, J=J, analytic=True, smart_paths=0,
               precision=default_precision)
    e_losses = np.logspace(np.log10(e_loss_max / 100), np.log10(e_loss_max),
                           n_trials, endpoint=False)

    for Q in Qs:
        # minimal padding to test worst case (can't `0` per `J=log2(N)`)
        sc = Scattering1D(**ckw, Q=Q, max_pad_factor=1)

        sp0 = smart_paths_exclude(sc.psi1_f, sc.psi2_f, e_th_direct=e_loss_max
                                  )['n2, n1']
        for e_loss in e_losses:
            sp1 = smart_paths_exclude(sc.psi1_f, sc.psi2_f, e_th_direct=e_loss
                                      )['n2, n1']
            for p in sp1:
                assert p in sp0, ("Q={}, e_loss={}\n{}\n{}"
                                  ).format(Q, e_loss, sp0, sp1)


def test_T():
    """Test that `T` controls degree of invariance as intended: double `T`
    should halve relative distance.
    """
    # configure scattering & signal
    J = 10
    Q = 16
    N = 2048
    width = N//8
    shift = N//8
    T0, T1 = N//2, N//4
    freq_fracs = (4, 8, 16, 32)

    # make signal & shifted
    window = scipy.signal.tukey(width, alpha=0.5)
    window = np.pad(window, (N - width) // 2)
    t  = np.linspace(0, 1, N, endpoint=False)
    x  = np.sum([np.cos(2*np.pi * N/ff * t) for ff in freq_fracs], axis=0
                ) * window
    xs = np.roll(x, shift)

    # make scattering objects
    kw = dict(J=J, Q=Q, shape=N, average=1, out_type="array", pad_mode="zero",
              max_pad_factor=None, frontend=default_frontend)
    ts0 = Scattering1D(T=T0, **kw)
    ts1 = Scattering1D(T=T1, **kw)

    # scatter
    ts0_x  = ts0(x)
    ts0_xs = ts0(xs)
    ts1_x  = ts1(x)
    ts1_xs = ts1(xs)

    # compare distances
    l2_00_xxs = rel_l2(ts0_x, ts0_xs)
    l2_11_xxs = rel_l2(ts1_x, ts1_xs)

    th0 = .09
    th1 = 2*th0
    assert l2_00_xxs < th0, "{} > {}".format(l2_00_xxs, th0)
    assert l2_11_xxs > th1, "{} < {}".format(l2_11_xxs, th1)


def test_aliasing():
    """Test that `oversampling=0` and `oversampling=99` agree.
    Measure aliasing due to

      (1) lowpass filter by DFT-upsampling `oversampling=0`.
      (2) strided wavelet convolutions by subsampling `oversampling=99`.

    Unpadding is accounted for by disabling padding, which works as well as
    disablind unpadding for our purposes.
    Relative L1 distance is used as the stricter alt to relative Euclidean.

    Details
    -------
    The benefits of subsampling aren't to be at expense of correctness.
    The completely unsubsampled case is used as the ground ground truth - meaning,
    if unsubsampled and subsampled carry the same information, we're happy.

    The idea with (2) is, pre-lowpass, the smaller output is a subset of the
    larger output - hence we take the subset of the larger output so it's equal
    in size to the smaller output, and compare. If strided wavelet convolutions
    are aliased, the unsubsampled and subsampled outputs will disagree.

    The idea with (1) is, the lowpass filter should become negligible before the
    `len(x)//2//T + 1`-th DFT bin ("negligible" defined by `criterion_amplitude`).
    If that's not the case, there will be aliasing, and the recovered full-length
    transform will disagree with the unsubsampled transform.

    Since unpadding aliases, it should be disabled to avoid confounding.
    Additionally, the smaller output no longer subsets the larger output due to
    center-padding, as stride offsets cause misalignment, invalidating the
    comparison in (2).
    This doesn't make the measures "unrealistic" - rather, it eliminates a factor
    we don't control, and isolates sources of error to aliasing, as intended.

    Note, WGN is used - while it provides a good general measure, it doesn't
    reflect the practical worst case, so this test can be improved
    (e.g. `bag_o_waves`, but this wasn't explored).
    """
    # setup ##################################################################
    # configure
    N = 1024
    J = int(np.log2(N) - 2)
    T = 2**J
    Q = 16

    # create signal & scat object
    np.random.seed(0)
    x = np.random.randn(N)
    # `max_pad_factor=0` is equivalently no unpadding - we could instead pad and
    # explicitly not unpad but results for our purposes are same
    cfg = dict(shape=N, J=J, Q=Q, T=T, max_pad_factor=0, out_type='list')
    sc0 = Scattering1D(**cfg, oversampling=0)
    sc1 = Scattering1D(**cfg, oversampling=99)

    # Compute, without unpadding
    out0 = sc0(x)
    out1 = sc1(x)

    # measure errors (1) #####################################################
    # recover `out1` from `out0`
    errs1 = np.zeros(len(out0))
    for i, (o0, o1) in enumerate(zip(out0, out1)):
        c0, c1 = o0['coef'], o1['coef']
        c0up = fft_upsample(c0, factor=len(c1)//len(c0),
                            time_to_time=True, real=True)
        errs1[i] = rel_l1(c1, c0up)

    # measure errors (2) #####################################################
    # obtain `out0` from `out1`
    errs2 = np.zeros(len(out0))
    for i, (o0, o1) in enumerate(zip(out0, out1)):
        c0, c1 = o0['coef'], o1['coef']
        c1dn = c1[::len(c1)//len(c0)]
        errs2[i] = rel_l1(c1dn, c0)

    # assert against thresholds ##############################################
    # much lower for higher `N`, as lesser `N` inherently aliases more
    # due to modulus
    th1, th2 = 0.01, 0.01
    assert errs1.max() < th1, (errs1.max(), errs1.mean())
    assert errs2.max() < th2, (errs2.max(), errs2.mean())


#### Primitives tests ########################################################
def _test_padding(backend_name):
    """Test that agnostic implementation matches numpy's."""
    def _arange(N):
        if backend_name == 'tensorflow':
            return backend.range(N)
        return backend.arange(N)

    backend = _get_backend(backend_name)

    for N in (128, 129):  # even, odd
        x = backend.reshape(_arange(6 * N), (2, 3, N))
        for pad_factor in (1, 2, 3, 4):
            pad_left = (N // 2) * pad_factor
            pad_right = int(np.ceil(N / 4) * pad_factor)

            for pad_mode in ('zero', 'reflect'):
                out0 = pad(x, pad_left, pad_right, pad_mode=pad_mode)
                out1 = np.pad(x,
                              [[0, 0]] * (x.ndim - 1) + [[pad_left, pad_right]],
                              mode=pad_mode if pad_mode != 'zero' else 'constant')

                out0 = out0.numpy() if hasattr(out0, 'numpy') else out0
                assert np.allclose(out0, out1), (
                    "{} | (N, pad_mode, pad_left, pad_right) = ({}, {}, {}, {})"
                    ).format(backend_name, N, pad_mode, pad_left, pad_right)


def _test_pad_axis(backend_name):
    """Test that padding any N-dim axis works as expected."""
    backend = _get_backend(backend_name)
    x = backend.zeros((5, 6, 7, 8, 9, 10, 11))

    pad_left, pad_right = 4, 5
    kw = dict(pad_left=pad_left, pad_right=pad_right, pad_mode='reflect')

    for axis in range(x.ndim):
        if backend_name == 'tensorflow' and axis != x.ndim - 1:
            # implemented only for last axis
            continue
        shape0 = list(x.shape)
        shape0[axis] += (pad_left + pad_right)
        shape1 = pad(x, axis=axis, **kw).shape
        shape2 = pad(x, axis=axis - x.ndim, **kw).shape  # negative axis version

        assert np.allclose(shape0, shape1)
        assert np.allclose(shape0, shape2)


def _test_subsample_fourier_axis(backend_name):
    """Test that subsampling an arbitrary axis works as expected."""
    backend = _get_backend(backend_name)
    B = get_wavespin_backend(backend_name)
    x = np.random.randn(4, 8, 16, 32)

    if backend_name == 'torch':
        xb = backend.tensor(x)
    elif backend_name == 'tensorflow':
        xb = backend.cast(backend.convert_to_tensor(x), backend.complex64)
    else:
        xb = x

    for sub in (2, 4):
        for axis in range(x.ndim):
            if (backend_name == 'tensorflow' and
                    axis not in (x.ndim - 1, x.ndim - 2)):
                # not implemented
                continue
            xf = B.fft(xb, axis=axis)
            outf = B.subsample_fourier(xf, sub, axis=axis)
            out = B.ifft(outf, axis=axis)

            xref = xb[stride_axis(sub, axis, xb.ndim)]
            if backend_name != 'numpy':
                out = out.numpy()
            out = out.real
            assert np.allclose(xref, out, atol=0), np.abs(xref - out).max()


def test_pad_numpy():
    _test_padding('numpy')
    _test_pad_axis('numpy')


def test_pad_torch():
    if cant_import('torch'):
        return
    _test_padding('torch')
    _test_pad_axis('torch')


def test_pad_tensorflow():
    if cant_import('tensorflow'):
        return
    _test_padding('tensorflow')
    _test_pad_axis('tensorflow')


def test_subsample_fourier_numpy():
    _test_subsample_fourier_axis('numpy')


def test_subsample_fourier_torch():
    if cant_import('torch'):
        return
    _test_subsample_fourier_axis('torch')


def test_subsample_fourier_tensorflow():
    if cant_import('tensorflow'):
        return
    _test_subsample_fourier_axis('tensorflow')


def test_emulate_get_conjugation_indices():
    """Test that `conj_reflections` indices fetcher works correctly."""
    for N in (123, 124, 125, 126, 127, 128, 129, 159, 180, 2048, 9589, 65432):
      for K in (1, 2, 3, 4, 5, 6):
        for pad_factor in (1, 2, 3, 4):
          for trim_tm in (0, 1, 2, 3, 4):
              if trim_tm > pad_factor:
                  continue
              test_params = dict(N=N, K=K, pad_factor=pad_factor)
              test_params_str = '\n'.join([f'{k}={v}' for k, v in
                                           test_params.items()])

              padded_len = 2**pad_factor * int(2**np.ceil(np.log2(N)))
              pad_left = int(np.ceil((padded_len - N) / 2))
              pad_right = padded_len - pad_left - N

              kw = dict(N=N, K=K, pad_left=pad_left, pad_right=pad_right,
                        trim_tm=trim_tm)
              out0, rp = _get_conjugation_indices(**kw)
              out1 = _emulate_get_conjugation_indices(**kw)

              errmsg = "{}\n{}\n{}".format(out0, out1, test_params_str)

              # first validate the original output ###########################
              # check by sign flipping at obtained slices and adding to
              # original then seeing if it sums to zero with original
              # in at least as many places as there are indices
              rp0 = rp[::2**K]
              rp1 = rp0.copy()
              for slc in out0:
                  rp1[slc] *= -1

              n_zeros_min = sum((s.stop - s.start) for s in out0)
              n_zeros_sum = np.sum((rp0 + rp1) == 0)

              if n_zeros_sum < n_zeros_min:
                  raise AssertionError("{} < {}\n{}".format(
                      n_zeros_sum, n_zeros_min, errmsg))

              # now assert equality with the emulation #######################
              for s0, s1 in zip(out0, out1):
                  assert s0.start == s1.start, errmsg
                  assert s0.stop  == s1.stop,  errmsg


#### utilities ###############################################################
def _get_backend(backend_name):
    if backend_name == 'numpy':
        backend = np
    elif backend_name == 'torch':
        import torch
        backend = torch
    elif backend_name == 'tensorflow':
        import tensorflow as tf
        backend = tf
    return backend


def _get_conjugation_indices(N, K, pad_left, pad_right, trim_tm):
    """Ground truth of the algorithm. Not tested for extreme edge cases, but
    those are impossible in implementation (stride > signal length).
    """
    import numpy as np

    # compute boundary indices from simulated reflected ramp at original length
    r = np.arange(N)
    rp = np.pad(r, [pad_left, pad_right], mode='reflect')
    if trim_tm > 0:
        rp = unpad_dyadic(rp, N, len(rp), len(rp) // 2**trim_tm)

    # will conjugate where sign is negative
    rpdiffo = np.diff(rp)
    rpdiffo = np.hstack([rpdiffo[0], rpdiffo])

    # mark rising ramp as +1, including bounds, and everything else as -1;
    # -1 will conjugate. This instructs to not conjugate: non-reflections, bounds.
    # diff will mark endpoints with sign opposite to rise's; correct manually
    rpdiffo2 = rpdiffo.copy()
    rpdiffo2[np.where(np.diff(rpdiffo) > 0)[0]] = 1

    rpdiff = rpdiffo2[::2**K]

    idxs = np.where(rpdiff == -1)[0]

    # convert to slices ######################################################
    if idxs.size == 0:
        slices_contiguous = []
    else:
        ic = [0, *(np.where(np.diff(idxs) > 1)[0] + 1)]
        ic.append(None)
        slices_contiguous = []
        for i in range(len(ic) - 1):
            s, e = ic[i], ic[i + 1]
            start = idxs[s]
            end = idxs[e - 1] + 1 if e is not None else idxs[-1] + 1
            slices_contiguous.append(slice(start, end))

    out = slices_contiguous
    return out, rp


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_smart_paths()
        test_smart_paths_subsetting()
        test_T()
        test_aliasing()
        test_pad_numpy()
        test_pad_torch()
        test_pad_tensorflow()
        test_subsample_fourier_numpy()
        test_subsample_fourier_torch()
        test_subsample_fourier_tensorflow()
        test_emulate_get_conjugation_indices()
    else:
        pytest.main([__file__, "-s"])
