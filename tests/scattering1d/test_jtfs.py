# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Joint Time-Frequency Scattering related tests."""
import os
import pytest
import numpy as np
import warnings
from pathlib import Path
from copy import deepcopy

from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.utils.gen_utils import npy, ExtendedUnifiedBackend, backend_has_gpu
from wavespin import toolkit as tkt
from wavespin.toolkit import echirp, energy
from wavespin.visuals import (coeff_distance_jtfs, compare_distances_jtfs,
                              energy_profile_jtfs, plot, plotscat)
from utils import (cant_import, IgnoreWarnings,
                   SKIPS, TEST_DATA_DIR, FORCED_PYTEST)

# backend to use for all tests (except `test_backends`),
# run one at a time since the tests take very long.
# NOTE: non-'numpy' skips `test_meta()` and `test_lp_sum()`
default_backend = ('numpy', 'torch', 'tensorflow', 'jax')[0]
# precision to use for all but precision-sensitive tests
default_precision = 'single'
# set True to execute all test functions without pytest
run_without_pytest = 1
# set True to print assertion errors rather than raising them in `test_output()`
output_test_print_mode = 1
# set True to print assertion values of certain tests
metric_verbose = 1
# set True to visualize certain assertion errors
viz = 1
# set True to skip this file entirely
SKIP_ALL = SKIPS['jtfs']
# same as `SKIP_ALL` but passed from command line
CMD_SKIP_ALL = bool(os.environ.get('CMD_SKIP_JTFS', '0') == '1')
# set True to skip longest tests (should be False before merging with main)
SKIP_LONG = SKIPS['long_in_jtfs']
# same as `SKIP_LONG` but passed from command line
CMD_SKIP_LONG = bool(os.environ.get('CMD_SKIP_LONG_JTFS', '0') == '1')

# former default, used for testing with sufficient padding.
# if they're typed manually, it means the test requires frequential padding None
pad_kw = dict(max_pad_factor=2, max_pad_factor_fr=None)
# former default
pad_mode_kw = dict(pad_mode_fr='conj-reflect-zero')


def test_alignment():
    """Ensure A.M. cosine's peaks are aligned across joint slices.

    Additionally tests `coeff2meta_jtfs` and that the highest activating
    coefficient's `xi1` meta matches the carrier's frequency.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    N = 1025
    J = 7
    Q = 16
    Q_fr = 2
    F = 4

    # generate A.M. cosine ###################################################
    f1, f2 = 256, 8
    t = np.linspace(0, 1, N, 1)
    c = np.cos(2*np.pi * f1 * t)
    a = (np.cos(2*np.pi * f2 * t) + 1) / 2
    x = a * c

    # scatter ################################################################
    for out_3D in (True, False):
      for sampling_psi_fr in ('resample', 'exclude'):
        for J_fr in (3, 5):
          out_type = ('dict:array' if out_3D else
                      'dict:list')  # for convenience
          test_params = dict(out_3D=out_3D,
                             sampling_filters_fr=(sampling_psi_fr, 'resample'))
          test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                      test_params.items())
          jtfs = TimeFrequencyScattering1D(
              N, J, Q, J_fr=J_fr, Q_fr=Q_fr, F=F, average=True, average_fr=True,
              aligned=True, out_type=out_type, frontend=default_backend,
              pad_mode='zero', pad_mode_fr='zero', smart_paths='primitive',
              precision=default_precision, **pad_kw, **test_params)

          Scx_orig = jtfs(x)
          Scx = tkt.drop_batch_dim_jtfs(Scx_orig)
          Scx = tkt.jtfs_to_numpy(Scx)

          # assert peaks share an index #################################
          def max_row_idx(c):
              coef = c['coef'] if 'list' in out_type else c
              return np.argmax(np.sum(coef**2, axis=-1))

          # select reference coefficient. NOTE: selection is simple and not
          # foulproof, the first coeff has shortest fr dim, which may not
          # contain `a`'s freq at all - future changes to coeff shapes may
          # hence cause failure.
          first_coef = Scx['psi_t * psi_f_up'][0]
          mx_idx_ref = max_row_idx(first_coef)
          for pair in Scx:
              if pair in ('S0', 'S1'):  # joint only
                  continue

              for i, c in enumerate(Scx[pair]):
                  mx_idx_i = max_row_idx(c)
                  # current max index must be within 1 of reference
                  assert abs(mx_idx_i - mx_idx_ref) <= 1, (
                      "{} != {} -- Scx[{}][{}]\n{}").format(
                          mx_idx_i, mx_idx_ref, pair, i, test_params_str)

          if J_fr == 3:
              # assert not all J_pad_frs are same so test covers this case
              assert_pad_difference(jtfs, test_params_str)

          # check `coeff2meta_jtfs` with 'array'
          jtfs.out_type = 'array'
          Scx = jtfs(x)
          if out_3D:
              Scx = (npy(Scx[0]), npy(Scx[1]))
          else:
              Scx = npy(Scx)
          jmeta = jtfs.meta()

          # find max energy row
          if out_3D:
              Scx_joint = Scx[1]
          else:
              Scx_joint = Scx[:, 1 + len(jtfs.psi1_f):]
          c = Scx_joint.reshape(-1, Scx_joint.shape[-1])
          ce = energy(c, axis=-1)
          if out_3D:
              out_idx = (1, np.argmax(ce))
          else:
              out_idx = np.argmax(ce) + 1 + len(jtfs.psi1_f)
          meta_at_idx = tkt.coeff2meta_jtfs(Scx, jmeta, out_idx=out_idx)
          # assert that obtained meta xi1 agrees with f1
          r_th = .03
          x_xi1 = f1 / N
          m_xi1 = meta_at_idx['xi'].squeeze()[-1]
          r = abs(1 - m_xi1 / x_xi1)
          assert r < r_th, (r, r_th, x_xi1, m_xi1)


def test_shapes():
    """Ensure `out_3D=True` joint coeff slices have same shape."""
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    N = 1024
    J = 6
    Q = 16

    x = np.random.randn(N)

    # scatter ################################################################
    for oversampling in (0, 1):
      for oversampling_fr in (0, 1):
        for aligned in (True, False):
          test_params = dict(oversampling=oversampling,
                             oversampling_fr=oversampling_fr, aligned=aligned)
          test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                      test_params.items())

          jtfs = TimeFrequencyScattering1D(
              N, J, Q, J_fr=4, Q_fr=2, average=True, average_fr=True,
              out_type='dict:array', out_3D=True, aligned=aligned,
              oversampling=oversampling, oversampling_fr=oversampling_fr,
              frontend=default_backend, precision=default_precision,
              **pad_kw, **pad_mode_kw)
          try:
              _ = jtfs(x)  # shapes must equal for this not to error
          except Exception as e:
              print(test_params_str)
              raise e


def test_jtfs_vs_ts():
    """Test JTFS sensitivity to FDTS (frequency-dependent time shifts), and that
    time scattering is insensitive to it.

    Here a global L2 measure is used, i.e. Euclidean distance between all
    time scattering and (spinned only) JTFS coefficients. "Spinned only" since
    only they are responsible for FDTS sensitivity.

      - This measure is flawed and may mislead, but is adequate here; see
        https://github.com/kymatio/kymatio/discussions/730
      - Better measures are on per-coeff or spatial-localized basis; former is
        done in `test_freq_tp_invar()`.
      - In case of global averaging in time, L2 is an excellent measure.


    Sensitivity factors
    -------------------

    JTFS:

        1. U1: unaveraged first order is subject to various discretization errors,
           both from the filterbank and the input signal. These include

           A. low Q1: fewer filters = larger LP sum oscillations and more
              artifacts when convolving with spinned wavelets. However, also

           B. low Q1 = superior temporal localization. May outweigh A.

           C. interferences: partials close in frequency, when shifted, won't
              perfectly shift respective scalogram bins, per wavelet
              non-compactness in frequency. Countered by sufficient partial
              separation, which makes wavelets equivalently compact per decaying
              below machine epsilon.

        2: Input:

           A. finite support boundary effects: see
           https://github.com/kymatio/kymatio/discussions/
           752#discussioncomment-864234

           B. insufficient sampling rate to capture full time-frequency geometry

        3: Padding:

           A. With large J and 'reflect', kernels may draw from padded portions,
              but this isn't a sensible "prior" (in this case of isolated sines)
              and does worsen the representation for FDTS purposes.

    S1:

        4. U1: 1C applies

        5. Input: 2A & 2B apply

        6. Padding:

            A. 3A applies

            B. insufficient padding relative to T will inflate phi_f's invariance
               and reduce L2 distance

        7. T: larger is better, ==N is best. The most significant factor by far;
           anything below `N//2` both has much less invariance and requires
           measures other than L2. It's why 'practical' won't do smaller T.

    Other factors exist.

    'perfect' variant
    -----------------
    Attains greatest possible ratio.

    Figuring out how it's done is informative, and a recommended exercise
    for the reader; it's a few changes to 'ideal'. Explanation, and a derived
    insight, follow.

    Infinite ratio is impossible*.

        - The bottlenecking culprit is non-compactness - namely, inability
          to be compact in both, time and frequency domains, simultaneously.

        - A practical culprit is spectral leakage. Windowed sinusoids aren't
          compact in frequency domain, nor pseudo-compact: unlike Morlets,
          they do not decay to float epsilon (float zero) quickly enough.

          This is consequential via *nonlinearity*: windowed sinusoids
          (partials) of different frequencies, perfectly aligned in time,
          contribute to each other in time-frequency: the wavelet at one
          partial doesn't correlate to zero with the other partial at the
          same time step. If one partial is shifted, this contribution is
          also shifted in time (equivariance), now no longer overlapping
          the other partial. Without the modulus, `sum(, axis=-1)` is
          identical in both cases, since total correlations per wavelet
          don't change. However, with the modulus, overlapped != non-overlapped:

              - `|(1 + 2j)| + |(3 + 4j)| = 7.24`
                # sum of modulus of wavelet transform at two time steps
              - `|(1 + 2j) + (3 + 4j)| + |(0 + 0j)| = 7.21`
                # sum of modulus of wavelet transform at the two time steps,
                # with the second time step shifted to the first

        - Above can be circumvented by forcing compactness of `x` in frequency
          domain (`brick_spectrum=True`). Note, this makes `x` non-compact in
          time-domain.

          Further, to make it a perfect shift, we must do the shift in frequency
          domain via complex rotation: shifting the continuous-time expression,
          and then sampling it, doesn't guarantee a pure shift of the resulting
          samples. (Alternatively, generate partials separately and shift the
          samples themselves in time domain).

          Lastly, note that we cannot perfectly isolate the two partials in
          frequency domain, as that amounts to making them disappear entirely.
          Instead, we make the *wavelets* compact by ensuring that no wavelet
          ever draws from both partials at once - i.e., the wavelet drops
          below float epsilon.

          This achieves the "float infinity" ratio, `1e16`.

    * Infinity is possible, but not in any useful manner:

        - Make wavelets drop below float tiny (`np.finfo(np.float64).tiny`),
          which should register them as exact zeros after multiplying with
          `fft(x)`, achieving perfect float compactness. This is a cheat.
        - Make wavelets compact in frequency domain directly. This eliminates
          overlap under modulus, making shifting of `x` a perfect shifting
          of modulus of wavelet transform of `x`. (Note, `x` must still be
          tweaked via `brick_spectrim`). This is a legitimate infinity.
        - Hence, infinity can't be accomplished with Morlets - not ever in
          continuous time, but in code only per float limitations.

    The "insight" is that, modulus actually helps *preserve* information under
    averaging. Scattering is only fully invariant to *global* shifts of `x`,
    not relative shifts of its partials. However, it can be very insensitive
    to such shifts (as shown in 'ideal' variant), which is same as invariance
    in almost any application - hence why we need JTFS.

    Note, the "legitimate infinity" is a bit questionable, as it contradicts
    the previous paragraph. Possibly, scattering requires non-compact wavelets
    in both domains. Or, there's an unexplored factor

    'ideal' variant
    ---------------
    Is essentially the 'perfect' variant minus the "hack" on `x`. The purpose
    is to demonstrate a realistic best-case.

    'practical' variant
    -------------------
    Drops global averaging as it's not always desired. Ideally we'd go farther
    from it than only half of global averaging, but that wrecks the ratio
    according to global L2 distance (which, again, isn't the best metric - but
    it still stays above `1`). We also add more partials as it's more realistic.

    'noisy' variant
    ---------------
    Is the 'practical' variant with noise and global averaging. Performs on par
    with 'practical', showing importance of global averaging to time invariance.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    VARIANTS_ALL = ('perfect', 'ideal', 'practical', 'noisy')
    variants = VARIANTS_ALL
    N = (2048, 4096)[0]
    N_scale = int(np.log2(N))

    # in case forgot to switch back to fast version
    if N != 2048:
        warnings.warn("test_jtfs_vs_ts: N != 2048")
    # in case forgot to switch back to complete version
    if variants != VARIANTS_ALL:
        warnings.warn("test_jtfs_vs_ts: variants != VARIANTS_ALL")

    # "practical" serves to reproduce practical transform imperfections rather
    # than reflect practical signals or transform configurations.
    # in all below examples, 'zero' padding (for both tm & fr) always wins, but
    # we test others just to account for them.
    cfgs = {}
    cfgs['ts_jtfs_common'] = dict(shape=N, frontend=default_backend,
                                  precision='double')
    cfgs['ts_common'] = dict(out_type='array', average=True, analytic=False)
    cfgs['jtfs_common'] = dict(sampling_filters_fr=('resample', 'resample'),
                               out_type='dict:array', max_pad_factor_fr=None,
                               out_3D=True, average_fr=True)
    cfgs['fdts'] = {
        'perfect': dict(
            f0=N//12,
            n_partials=2,
            partials_f_sep=5,
            total_shift=N//6,
            seg_len=N//6,
            global_shift=-N//5,
            brick_spectrum=True,
            ),
        'ideal': dict(
            f0=N//12,
            n_partials=2,
            partials_f_sep=5,
            total_shift=N//6,
            seg_len=N//6,
            global_shift=-N//5,  # approx center after doing FDTS; bound effs
            ),
        'practical': dict(
            f0=N//20,
            n_partials=5,
            partials_f_sep=1.4,
            total_shift=N//6,
            seg_len=N//6,
            ),
        'noisy': dict(
            f0=N//20,
            n_partials=5,
            partials_f_sep=1.4,
            total_shift=N//6,
            seg_len=N//6,
            )
    }
    cfgs['ts'] = {
        'perfect': dict(
            T=2**N_scale,
            J=N_scale - 3,
            Q=(16, 1),
            pad_mode='zero',
            oversampling=99,  # anti alias
            max_pad_factor=0,
        ),
        'ideal': dict(
            T=2**N_scale,
            J=N_scale - 3,
            Q=(8, 1),
            pad_mode='zero',
            max_pad_factor=None,
            ),
        'practical': dict(
            T=2**(N_scale - 1),  # <- greatest influence on ratio
            J=N_scale - 2,
            Q=(12, 1),
            pad_mode='zero',
            max_pad_factor=None,
            ),
        'noisy': dict(
            T=2**N_scale,
            J=N_scale - 3,
            Q=(16, 1),
            pad_mode='reflect',
            out_type='array',
            max_pad_factor=None,
            )
    }
    cfgs['jtfs'] = {
        'perfect': dict(
            Q_fr=4,
            J_fr=4,
            pad_mode_fr='zero',
            F=4,
            ),
        'ideal': dict(
            Q_fr=4,
            J_fr=4,
            pad_mode_fr='zero',
            F=4,
            ),
        'practical': dict(
            Q_fr=2,
            J_fr=4,
            pad_mode_fr='conj-reflect-zero',
            ),
        'noisy': dict(
            Q_fr=2,
            J_fr=4,
            pad_mode_fr='zero',
            )
    }
    # better results with larger N
    cfgs['th_ts'] = {
        2048: {'perfect': 2.5e-16,
               'ideal': 1e-5,
               'practical': 8e-2,
               'noisy': 8e-2},
        4096: {'perfect': 1.5e-16,
               'ideal': 8e-6,
               'practical': 8e-2,
               'noisy': 5e-2}
        }[N]
    cfgs['th_ratio'] = {
        2048: {'perfect': 1e15 * .95,  # remote builds have lower precision..?
               'ideal': 4e4,
               'practical': 4.5,
               'noisy': 3.5},
        4096: {'perfect': 1e15,
               'ideal': 6.5e4,
               'practical': 5,
               'noisy': 5.5}
    }[N]

    if metric_verbose:
        print("\nFDTS sensitivity (global L2):")
    for variant in variants:
        if default_backend == 'tensorflow' and variant in ('perfect', 'ideal'):
            # TF has numeric precision issues with small values
            continue
        _test_jtfs_vs_ts(N, cfgs, variant)


def _test_jtfs_vs_ts(N, cfgs, variant):
    # unpack configs
    C = deepcopy(cfgs)
    C = {k: (v[variant] if variant in v else v) for k, v in C.items()}
    C['ts_for_jtfs'] = {k: v for k, v in C['ts'].items()
                        if k not in ('out_type',)}

    cfg_ts = {k: v for name in ('ts', 'ts_common', 'ts_jtfs_common')
              for k, v in C[name].items()}
    cfg_jtfs = {k: v for name in ('jtfs', 'jtfs_common', 'ts_jtfs_common',
                                  'ts_for_jtfs')
                for k, v in C[name].items()}
    # handle special case
    if variant == 'perfect':
        cfg_jtfs['oversampling'] = 0  # no need for this, too slow

    # make signal
    x, xs = tkt.fdts(N=N, **C['fdts'])
    if variant == 'noisy':
        np.random.seed(0)  # notable dependence, ideally try several seeds
        noise = np.sqrt(.2) * np.random.randn(N)
        snr = 10 * np.log10(x.var() / noise.var())  # in decibels
        x += noise
        xs += noise
    else:
        snr = -1
    # # scale up to avoid working near float epsilon for `l2_ts`
    x, xs = x*10, xs*10

    # make scattering objects
    ts   = Scattering1D(**cfg_ts)
    jtfs = TimeFrequencyScattering1D(**cfg_jtfs)

    # scatter
    ts_x  = ts(x)
    ts_xs = ts(xs)
    jtfs_x_all  = jtfs(x)
    jtfs_xs_all = jtfs(xs)

    # unpack
    jtfs_x_all  = tkt.jtfs_to_numpy(jtfs_x_all)
    jtfs_xs_all = tkt.jtfs_to_numpy(jtfs_xs_all)
    jtfs_x  = concat_joint(jtfs_x_all,  spinned_only=True)
    jtfs_xs = concat_joint(jtfs_xs_all, spinned_only=True)

    # compute distance
    rl2_ts   = float(tkt.rel_l2(ts_x, ts_xs))
    rl2_jtfs = float(tkt.rel_l2(jtfs_x, jtfs_xs))

    # assert
    assert rl2_jtfs / rl2_ts > C['th_ratio'], (
        "\nvariant={}\nJTFS/TS: {} <= {} \nTS: {}\nJTFS: {}"
        ).format(variant, rl2_jtfs / rl2_ts, C['th_ratio'], rl2_ts, rl2_jtfs)
    assert rl2_ts < C['th_ts'], (
        "\nvariant={}\nTS: {} >= {}"
        ).format(variant, rl2_ts, C['th_ts'])

    # report
    if metric_verbose:
        title = {'perfect': 'Perfect',
                 'ideal': 'Ideal',
                 'practical': 'Practical (non global avg, other)',
                 'noisy': 'Noisy (%.2fdB SNR)' % snr}[variant]
        print(("\n{}:\n"
               "JTFS/TS = {:.1f}\n"
               "JTFS    = {:.2f}\n"
               "TS      = {:.2e}"
               ).format(title, rl2_jtfs / rl2_ts, rl2_jtfs, rl2_ts))


def test_freq_tp_invar():
    """Test frequency transposition invariance.

    Note, greater `F` won't necessarily lower global Euclidean distance.
    This is per `x1` not being a perfect log-frequency shift of `x0`, instead
    differing slightly in norm (particularly on per temporal slice basis, along
    frequency), violating scattering theorem assumptions.
    It follows discretization artefacts in both the signal and the transform*
    (note, not studied thoroughly), but most coefficients behave as intended,
    hence motivating an alternative, per-coefficient distance measure.

    See https://github.com/kymatio/kymatio/discussions/730

    * should be mediated by higher sampling rates in both, namely greater
    `N` for signal, and greater `Q1` and `r_psi` for transform to increase
    the number of wavelets without changing the quality factor, `xi/sigma`.
    If `x1` crosses into non-CQT, then docs on `max_noncqt_fr` become relevant.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    # design signal
    N = 2048
    f0 = N // 12
    f1 = f0 / np.sqrt(2)
    n_partials = 5
    seg_len = N//8

    x0 = tkt.fdts(N, n_partials, f0=f0, seg_len=seg_len)[0]
    x1 = tkt.fdts(N, n_partials, f0=f1, seg_len=seg_len)[0]

    # make scattering objects
    J = int(np.log2(N) - 1)  # have 2 time units at output
    J_fr = 5
    F_all = [32, 64]

    pair_distances, global_distances = [], []
    for F in F_all:
        jtfs = TimeFrequencyScattering1D(
            J=J, Q=16, Q_fr=1, J_fr=J_fr, shape=N, F=F, average_fr=True,
            out_3D=False, out_type='dict:array', out_exclude=('S0', 'S1'),
            pad_mode='reflect', pad_mode_fr='conj-reflect-zero',
            sampling_filters_fr=('resample', 'resample'), max_pad_factor_fr=None,
            max_pad_factor=2, frontend=default_backend, precision='double')
        # scatter
        jtfs_x0_all = jtfs(x0)
        jtfs_x1_all = jtfs(x1)
        jtfs_x0_all = tkt.jtfs_to_numpy(jtfs_x0_all)
        jtfs_x1_all = tkt.jtfs_to_numpy(jtfs_x1_all)

        # compute & append distances
        _, pair_dist = coeff_distance_jtfs(jtfs_x0_all, jtfs_x1_all,
                                           jtfs.meta(), plots=False)
        pair_distances.append(pair_dist)

        jtfs_x0 = concat_joint(jtfs_x0_all)
        jtfs_x1 = concat_joint(jtfs_x1_all)
        global_distances.append(float(tkt.rel_l2(jtfs_x0, jtfs_x1)))

    if metric_verbose:
        print("\nFrequency transposition invariance stats:")

    # compute stats & assert
    _, stats = compare_distances_jtfs(*pair_distances, plots=0,
                                      verbose=metric_verbose, title="F: 32 vs 64")
    maxs, means = zip(*[(s['max'], s['mean']) for s in stats.values()])
    max_max, mean_mean = max(maxs), np.mean(means)
    # best case must attain at least twice the invariance
    assert max_max > 2, max_max
    # global mean ratio should exceed unity
    assert mean_mean > 1.4, mean_mean

    if metric_verbose:
        print("max_max, mean_mean = {:.2f}, {:.2f}".format(max_max, mean_mean))
        print("Global L2: (F=32, F=64, ratio) = ({:.3f}, {:.3f}, {:.3f})".format(
            *global_distances, global_distances[0] / global_distances[1]))


def test_up_vs_down():
    """Test that echirp yields significant disparity in up vs down energies.

    Related discussions:
    https://github.com/kymatio/kymatio/discussions/752#discussioncomment-864234
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    N = 2048
    x = echirp(N, fmin=64)

    if metric_verbose:
        print("\nFDTS directional sensitivity; E_dn / E_up:")

    # note: if a design change requires making these numbers substantially lower,
    # it's not necessarily bad, but always worth understanding why the numbers
    # changed. They shouldn't change as much as to cut in half, however.
    m_th = (220, 600)
    l2_th = (300, 810)
    for i, pad_mode in enumerate(['reflect', 'zero']):
        pad_mode_fr = 'conj-reflect-zero' if pad_mode == 'reflect' else 'zero'
        jtfs = TimeFrequencyScattering1D(
            shape=N, J=(8, 6), Q=16, J_fr=4, F=4, Q_fr=2, average_fr=True,
            out_type='dict:array', pad_mode=pad_mode, max_pad_factor=None,
            max_pad_factor_fr=None, sampling_filters_fr=('resample', 'resample'),
            pad_mode_fr=pad_mode_fr, frontend=default_backend,
            precision='double')
        Scx = jtfs(x)
        Scx = tkt.jtfs_to_numpy(Scx)
        jmeta = jtfs.meta()

        r = tkt.coeff_energy_ratios(Scx, jmeta)
        r_m = r.mean()

        E_up = tkt.coeff_energy(Scx, jmeta, pair='psi_t * psi_f_up')
        E_dn = tkt.coeff_energy(Scx, jmeta, pair='psi_t * psi_f_dn')
        r_l2 = E_dn / E_up

        if metric_verbose:
            print(("Global:     {0:<6.1f} -- '{1}' pad\n"
                   "Slice mean: {2:<6.1f} -- '{1}' pad").format(
                       r_l2, pad_mode, r_m))
        assert r_l2 > l2_th[i], "{} < {} | '{}'".format(r_l2, l2_th[i], pad_mode)
        assert r_m  > m_th[i],  "{} < {} | '{}'".format(r_m,  m_th[i],  pad_mode)


def test_sampling_psi_fr_exclude():
    """Test that outputs of `sampling_psi_fr=='exclude'` are a subset of
    `'resample'` (i.e. equal wherever both exist).
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    N = 2048
    x = echirp(N)

    params = dict(shape=N, J=11, Q=8, J_fr=4, Q_fr=2, F=4, average_fr=True,
                  max_pad_factor_fr=None, max_pad_factor=2,
                  out_type='dict:list', frontend=default_backend,
                  precision=default_precision)
    test_params_str = '\n'.join(f'{k}={v}' for k, v in params.items())
    jtfs0 = TimeFrequencyScattering1D(
        **params, sampling_filters_fr=('resample', 'resample'))
    jtfs1 = TimeFrequencyScattering1D(
        **params, sampling_filters_fr=('exclude', 'resample'))

    # required otherwise 'exclude' == 'resample'
    assert_pad_difference(jtfs0, test_params_str)
    # reproduce case with different J_pad_fr
    assert jtfs0.J_pad_frs != jtfs1.J_pad_frs, jtfs0.J_pad_frs

    Scx0 = jtfs0(x)
    Scx1 = jtfs1(x)
    Scx0 = tkt.jtfs_to_numpy(Scx0)
    Scx1 = tkt.jtfs_to_numpy(Scx1)

    # assert equality where `n` metas match
    # if `n` don't match, assert J_pad_fr is below maximum
    for pair in Scx1:
        i1 = 0
        for i0, c in enumerate(Scx1[pair]):
            s0, s1 = Scx0[pair][i0], Scx1[pair][i1]
            n0, n1 = s0['n'], s1['n']
            c0, c1 = s0['coef'], s1['coef']
            info = "{}, (i0, i1)=({}, {}); (n0, n1)=({}, {})".format(
                pair, i0, i1, n0, n1)

            is_joint = bool(pair not in ('S0', 'S1'))
            if is_joint:
                pad = jtfs1.J_pad_frs[jtfs1.scale_diffs[n0[0]]]
                pad_max = jtfs1.J_pad_frs_max
            if n0 != n1:
                assert is_joint, (
                    "found mismatch in time scattered coefficients\n%s" % info)
                # Mismatched `n` should only happen for mismatched `pad_fr`.
                # Check 1's pad as indexed by 0, since n0 lags n1 and might
                # have e.g. pad1[n0=5]==(max-1), pad[n1=6]==max, but we're still
                # iterating n==5 so appropriate comparison is at 5
                assert pad != pad_max, (
                    "{} == {} | {}\n(must have sub-maximal `J_pad_fr` for "
                    "mismatched `n`)").format(pad, pad_max, info)
                continue

            assert c0.shape == c1.shape, "shape mismatch: {} != {} | {}".format(
                c0.shape, c1.shape, info)
            ae = tkt.rel_ae(c0, c1)
            # due to different energy renorms (LP sum)
            atol = 1e-8 if (not is_joint or pad == pad_max) else 1e-2
            assert np.allclose(c0, c1, atol=atol), (
                "{} | MeanAE={:.2e}, MaxAE={:.2e}").format(
                    info, ae.mean(), ae.max())
            i1 += 1


def test_max_pad_factor_fr():
    """Test that low and variable `max_pad_factor_fr` works, and that
    `unrestricted_pad_fr` works with large `F`."""
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    skip_long = bool(CMD_SKIP_LONG or SKIP_LONG)

    N = 1024
    x = echirp(N)

    # configure params based on `SKIP_LONG` flag
    C = dict(
        aligned=(True, False),
        sampling_filters_fr=('resample', 'exclude', 'recalibrate'),
        max_pad_factor_fr=([1, 2, 0], 0, 1, None),
    )
    if skip_long:
        for k, v in C.items():
            C[k] = v[:1]  # keep only 1

    # test
    for aligned in C['aligned']:
        for sampling_filters_fr in C['sampling_filters_fr']:
          for max_pad_factor_fr in C['max_pad_factor_fr']:
            # 127 to stay large but avoid F='global' per 'recalibrate'.
            # 32 since otherwise None takes too long, and we just want to test
            # log2_F > J_fr.
            F_large = (127 if max_pad_factor_fr is not None else
                       32)
            for F in (16, F_large):
                if sampling_filters_fr == 'recalibrate' and aligned:
                    # otherwise, invalid option
                    sampling_filters_fr = ('recalibrate', 'resample')
                test_params = dict(aligned=aligned, F=F,
                                   sampling_filters_fr=sampling_filters_fr,
                                   max_pad_factor_fr=max_pad_factor_fr)
                test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                            test_params.items())

                try:
                    with IgnoreWarnings("boundary effects"):
                        jtfs = TimeFrequencyScattering1D(
                            shape=N, J=9, Q=12, J_fr=4, Q_fr=1, average_fr=True,
                            out_3D=True, max_pad_factor=2,
                            **test_params, frontend=default_backend,
                            precision=default_precision, **pad_mode_kw)
                except Exception as e:
                    if not ("same `J_pad_fr`" in str(e) and
                            sampling_filters_fr == 'recalibrate'):
                        print("Failed on %s with" % test_params_str)
                        raise e
                    else:
                        continue
                assert_pad_difference(jtfs, test_params_str)

                try:
                    _ = jtfs(x)
                except Exception as e:
                    print("Failed on %s with" % test_params_str)
                    raise e

    if skip_long:
        warnings.warn("Skipped most of `test_max_pad_factor_fr()` per "
                      "`SKIPS['long_in_jtfs']` or `--SKIP_LONG_jtfs`")


def test_out_exclude():
    """Test that `out_exclude` works as expected.

    Also partially tests that `average=False` doesn't error.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    N = 512
    params = dict(shape=N, J=4, Q=4, J_fr=4, average=False, average_fr=True,
                  out_type='dict:list', frontend=default_backend, **pad_kw,
                  precision=default_precision)
    x = np.random.randn(N)

    all_pairs = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                 'psi_t * phi_f', 'psi_t * psi_f_up', 'psi_t * psi_f_dn')
    out_excludes = [
        ('S0', 'psi_t * psi_f_up'),
        ('psi_t * psi_f_dn', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f'),
        ('S1', 'psi_t * psi_f_up', 'psi_t * psi_f_dn'),
    ]
    for out_exclude in out_excludes:
        jtfs = TimeFrequencyScattering1D(**params, out_exclude=out_exclude)
        out = jtfs(x)
        jmeta = jtfs.meta()

        for pair in out:
            assert pair not in out_exclude, pair
            assert pair in all_pairs, pair  # ensure nothing else was inserted

        for field in jmeta:
            for pair in jmeta[field]:
                assert pair not in out_exclude, (field, pair)
                assert pair in all_pairs, (field, pair)

    # ensure invalid pair is caught
    with pytest.raises(ValueError) as record:
        jtfs = TimeFrequencyScattering1D(**params, out_exclude=('banana',))
    assert "invalid coefficient" in record.value.args[0]


def test_global_averaging():
    """Test that `T==N` and `F==pow2(N_frs_max)` doesn't error, and outputs
    close to `T==N-1` and `F==pow2(N_frs_max)-1`
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    np.random.seed(0)
    N = 512
    params = dict(
        shape=N, J=9, Q=4, J_fr=5, Q_fr=2,
        average=True, average_fr=True, out_type='dict:array',
        pad_mode='reflect', pad_mode_fr='conj-reflect-zero',
        max_pad_factor=None, max_pad_factor_fr=None,
        sampling_filters_fr=('resample', 'resample'),
        frontend=default_backend, precision=default_precision,
    )
    x = echirp(N)
    x += np.random.randn(N)

    outs = {}
    metas = {}
    Ts, Fs = (N - 1, N), (2**6 - 1, 2**6)
    for T in Ts:
        # N_frs_max ~= Q*max(p2['j'] for p2 in psi2_f); found 29 at runtime
        for F in Fs:
            jtfs = TimeFrequencyScattering1D(**params, T=T, F=F)
            assert (jtfs.average_fr_global if F == Fs[-1] else
                    not jtfs.average_fr_global)
            assert (jtfs.average_global if T == Ts[-1] else
                    not jtfs.average_global)

            out = jtfs(x)
            out = tkt.jtfs_to_numpy(out)
            outs[ (T, F)] = out
            metas[(T, F)] = jtfs.meta()

    T0F0 = tkt.coeff_energy(outs[(Ts[0], Fs[0])], metas[(Ts[0], Fs[0])])
    T0F1 = tkt.coeff_energy(outs[(Ts[0], Fs[1])], metas[(Ts[0], Fs[1])])
    T1F0 = tkt.coeff_energy(outs[(Ts[1], Fs[0])], metas[(Ts[1], Fs[0])])
    T1F1 = tkt.coeff_energy(outs[(Ts[1], Fs[1])], metas[(Ts[1], Fs[1])])

    if metric_verbose:
        print("\nGlobal averaging reldiffs:")

    th = .15
    for pair in T0F0:
        ref = T0F0[pair]
        reldiff01 = abs(T0F1[pair] - ref) / ref
        reldiff10 = abs(T1F0[pair] - ref) / ref
        reldiff11 = abs(T1F1[pair] - ref) / ref
        assert reldiff01 < th, "%s > %s | %s" % (reldiff01, th, pair)
        assert reldiff10 < th, "%s > %s | %s" % (reldiff10, th, pair)
        assert reldiff11 < th, "%s > %s | %s" % (reldiff11, th, pair)

        if metric_verbose:
            print("(01, 10, 11) = ({:.2e}, {:.2e}, {:.2e}) | {}".format(
                reldiff01, reldiff10, reldiff11, pair))


def test_lp_sum():
    """Test that filterbank energy renormalization works as expected.

        - analytic-only filterbanks aim for peaking at `2`
        - analytic & anti-analytic aim for peaking at `1`
        - peak targets are thresholded, separately for
          - CQT region
          - non-CQT region
          - DC bin
          - bins 1-2, which, with max padding, tend to have minimal signal
            energy, making it okay to under-tile
          - bins between highest peak frequency and Nyquist
        - LP sums of analytic-only filterbanks, with `analytic=False`, account
          for negative frequencies (LP sum sum)

    Tests encompass a wide variety of filterbank behaviors:
        - wide and narrow wavelet bandwidth (`Q`, `r_psi`)
        - leaking/non-leaking across Nyquist (`analytic`)
        - sufficient/insufficient filterbank lengths (`max_pad_factor`,
          `max_pad_factor_fr`)
        - frequential design (`sampling_psi_fr`)

    Future implementations may wish to set `check_up_to_nyquist=False`;
    see "Nyquist correction note" in `energy_norm_filter_bank`.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    skip_long = bool(CMD_SKIP_LONG or SKIP_LONG)

    if default_backend != 'numpy':
        # filters don't change
        warnings.warn("`test_lp_sum()` skipped per non-'numpy' `default_backend`")
        return None if run_without_pytest else pytest.skip()

    def maybe_viz_err(lp, psis, peak_target=None):
        if viz:
            halflen = len(psis[0]) // 2
            # last_peak_idx = np.argmax(psis[0])
            first_peak_idx = np.argmax(psis[-1])
            is_analytic = bool(first_peak_idx < halflen)

            t_idxs0 = (slice(0, halflen) if is_analytic else
                       slice(halflen, None))
            if halflen > 50:
                # low freq portion may still be relevant in is_cqt=True
                t_idxs_zoom = (slice(0, 50) if is_analytic else
                               slice(halflen*2 - 50, None))
            else:
                t_idxs_zoom = None

            for t_idxs in (t_idxs0, t_idxs_zoom):
                if t_idxs is None:
                    continue

                pkw = dict(show=1, ylims=(0, None))
                plotscat(lp[t_idxs], ylims=(0, None))
                if peak_target is not None:
                    hlines = (peak_target,
                              {'color': 'tab:red', 'linestyle': '--'})
                else:
                    hlines = None
                plot([], hlines=hlines, show=1)

                plot(np.array(psis)[:, t_idxs].T, **pkw)
                plotscat(psis[-1][t_idxs], **pkw)

    def _get_peak_idxs(lp, test_params_str, warned_pad, psi_fs, k, psi_id,
                       sampling_filters_fr, max_pad, is_cqt=True, below=None):
        # unpack filterbank
        def cond(psi_is_cqt, n1_fr):
            return ((is_cqt and psi_is_cqt[n1_fr]) or
                    (not is_cqt and not psi_is_cqt[n1_fr]))

        # has no `is_cqt==True`
        is_recalibrate = bool(psi_id is not None and psi_id > 0 and
                              sampling_filters_fr == 'recalibrate')

        # get `psis`
        if psi_id is not None:
            psi_is_cqt = psi_fs['is_cqt'][psi_id]
            psis = [p for n1_fr, p in enumerate(psi_fs[psi_id])
                    if cond(psi_is_cqt, n1_fr)]

        elif k is not None:
            psi_is_cqt = [p['is_cqt'] for p in psi_fs]
            psis = [p[k] for n1_fr, p in enumerate(psi_fs)
                    if k in p and cond(psi_is_cqt, n1_fr)]

        # in non-CQT, include one CQT so there isn't a blindspot
        # between CQT & non-CQT
        if not is_cqt and not is_recalibrate and len(psis) != 0:
            last_cqt_idx = -(len(psis) + 1)
            if psi_id is not None:
                last_cqt = psi_fs[psi_id][last_cqt_idx]
                assert psi_fs['is_cqt'][psi_id][last_cqt_idx] is True
            elif k is not None:
                if k in psi_fs[last_cqt_idx]:
                    last_cqt = psi_fs[last_cqt_idx][k]
                    assert psi_fs[last_cqt_idx]['is_cqt'] is True
                else:
                    last_cqt = None
            if last_cqt is not None:
                psis.insert(0, last_cqt)

        # get peak indices
        if len(psis) == 0:
            # all CQT or non-CQT, pass
            cqt_but_no_k = bool(is_cqt and
                                k is not None and
                                not any(k in p and p['is_cqt'] for p in psi_fs))

            assert (not is_cqt or is_recalibrate or cqt_but_no_k), (
                "\n%s\nis_cqt=%s" % (test_params_str, is_cqt))
            first_peak, last_peak = None, None
        else:
            pk0, pk1 = np.argmax(psis[0]), np.argmax(psis[-1])
            # spin down (analytic) will be (pk1, pk0), up will be (pk0, pk1)
            first_peak, last_peak = min(pk0, pk1), max(pk0, pk1)

            # for later
            is_analytic_part = bool(first_peak < len(lp) // 2)

            if not is_cqt:
                # include bin 3 (bins 1 & 2 checked separately)
                if is_analytic_part:
                    first_peak = 3
                    if first_peak >= last_peak:
                        last_peak = first_peak + 1
                else:
                    last_peak = len(lp) - 3
                    if last_peak <= first_peak:
                        first_peak = last_peak - 1

            # in CQT, include Nyquist so there isn't a blindspot between
            # highest freq peak and Nyquist
            if check_up_to_nyquist and not below and (
                    _should_check_nyquist(k, sampling_filters_fr, is_cqt)):
                if is_analytic_part:
                    last_peak = len(lp) // 2
                else:
                    first_peak = len(lp) // 2

        # handle edge case
        if first_peak is not None and first_peak >= last_peak:
            # fr we already skip if poorly padded
            # len(psis) == 1 should only occur in tm if poorly padded
            warned_pad_or_is_not_time = bool(k is None or
                                             (k is not None and warned_pad['tm']))
            # account for separately since warning isn't thrown for >=2 wavelets
            if not (len(psis) == 1 or warned_pad_or_is_not_time):
                maybe_viz_err(lp, psis)
                raise AssertionError((
                    "{} <= {}\n{}\nis_cqt={}\npsi_id={}\nk={}\n"
                    ).format(last_peak, first_peak, test_params_str, is_cqt,
                             psi_id, k))

        n_psis = len(psis)
        return first_peak, last_peak, psis, n_psis

    def _should_check_nyquist(k, sampling_filters_fr, is_cqt):
        if_tm_then_full_filterbank = bool((k is not None and k == 0) or
                                          k is None)
        return bool(((is_cqt and if_tm_then_full_filterbank) or
                     sampling_filters_fr == 'recalibrate'))

    def _get_th_and_text(loc, k, psi_id, th_loc, sampling_filters_fr, max_pad,
                         is_cqt, n_psis):
        is_recalibrate = bool(sampling_filters_fr == 'recalibrate' and
                              psi_id is not None and psi_id > 0)
        if loc == 'above':
            th_loc = (th_above if (is_cqt or is_recalibrate) else
                      th_above_non_cqt)
        elif loc == 'below':
            if n_psis == 1:
                th_loc = th_below_one_psi
            elif is_cqt or is_recalibrate:
                th_loc = (th_below if max_pad else
                          th_below_nonmax_pad)
            else:
                th_loc = th_below_non_cqt

        if k is not None:  # temporal
            s = "k=%s" % k
            peak_target = 2
        elif psi_id is not None:  # frequential
            s = "psi_id=%s" % psi_id
            peak_target = 1
        th = peak_target * th_loc
        return th, peak_target, s

    def check_above(lp, test_params_str, warned_pad, psi_fs, k=None,
                    psi_id=None, sampling_filters_fr=None, max_pad=None):
        for is_cqt in (True, False):
            _check_above(lp, test_params_str, warned_pad, psi_fs,
                         k, psi_id, sampling_filters_fr, max_pad, is_cqt)

    def _check_above(lp, test_params_str, warned_pad, psi_fs, k,
                     psi_id, sampling_filters_fr, max_pad, is_cqt=True):
        # get peak indices
        first_peak, last_peak, psis, n_psis = _get_peak_idxs(
            lp, test_params_str, warned_pad, psi_fs, k, psi_id,
            sampling_filters_fr, max_pad, is_cqt)
        if first_peak is None and last_peak is None:
            return

        # get thresholds & assert
        # use CQT bounds for 'recalibrate' since it's always non-CQT but also
        # not fixed in bandwidth and linear-spaced in frequency
        is_recalibrate = bool(sampling_filters_fr == 'recalibrate' and
                              psi_id is not None and psi_id > 0)
        th_loc = (th_above if (is_cqt or is_recalibrate)
                  else th_above_non_cqt)
        th, peak_target, s = _get_th_and_text('above', k, psi_id, th_loc,
                                              sampling_filters_fr, max_pad,
                                              is_cqt, n_psis)

        lp_max = lp[first_peak:last_peak + 1].max()
        if not lp_max - peak_target < th:
            maybe_viz_err(lp, psis, peak_target)
            raise AssertionError(("{} - {} >= {} | \n{}\n{}\nis_cqt={}"
                                 ).format(lp_max, peak_target, th, s,
                                          test_params_str, is_cqt))

    def check_below(lp, test_params_str, warned_pad, psi_fs, analytic, k=None,
                    psi_id=None, sampling_filters_fr=None, max_pad=None):
        for is_cqt in (True, False):
            _check_below(lp, test_params_str, warned_pad, psi_fs, analytic, k,
                         psi_id, sampling_filters_fr, max_pad, is_cqt)

    def _check_below(lp, test_params_str, warned_pad, psi_fs, analytic, k,
                     psi_id, sampling_filters_fr, max_pad, is_cqt=True):
        # get peak indices
        first_peak, last_peak, psis, n_psis = _get_peak_idxs(
            lp, test_params_str, warned_pad, psi_fs, k, psi_id,
            sampling_filters_fr, max_pad, is_cqt, below=True)
        if first_peak is None and last_peak is None:
            return

        # get thresholds
        th_loc = th_below if is_cqt else th_below_non_cqt
        th, peak_target, s = _get_th_and_text('below', k, psi_id, th_loc,
                                              sampling_filters_fr, max_pad,
                                              is_cqt, n_psis)

        def _assert(first_peak, last_peak, peak_target, th):
            lp_min = lp[first_peak:last_peak + 1].min()
            if not peak_target - lp_min < th:
                maybe_viz_err(lp, psis, peak_target)
                raise AssertionError(
                    ("{} - {} >= {} | between peaks {} and {}\n{}\n{}"
                     "\nis_cqt={}\npsi_id={}").format(
                         peak_target, lp_min, th, first_peak, last_peak, s,
                         test_params_str, is_cqt, psi_id))

        # exclude Nyquist upon `analytic=True`
        if analytic:
            if first_peak == len(lp) // 2:
                first_peak += 1
            elif last_peak == len(lp) // 2:
                last_peak -= 1

        # lp min must be checked between peaks since it drops to 0 elsewhere
        _assert(first_peak, last_peak, peak_target, th)

        # special case: bins 1 & 2
        is_analytic_part = bool(first_peak < len(lp) // 2)
        if not is_cqt or (first_peak in (2, 3) or
                          last_peak in (len(lp) - 1, len(lp) - 2)):
            if is_analytic_part:
                first_peak = 1
            else:
                first_peak = len(lp) - 3
            last_peak = first_peak + 2
            _assert(first_peak, last_peak, peak_target,
                    th_below_bin_1 * peak_target)

        # special case: between highest freq & Nyquist
        if _should_check_nyquist(k, sampling_filters_fr, is_cqt):
            if is_analytic_part:
                first_peak = last_peak
                last_peak = (len(lp) // 2 - 1 if analytic else
                             len(lp) //2 )
            else:
                last_peak = first_peak
                first_peak = (len(lp) // 2 + 1 if analytic else
                              len(lp) // 2)
            _assert(first_peak, last_peak, peak_target,
                    th_below_nyquist_region * peak_target)

    def check_lp_sum_sum(psi_fs, test_params_str, Q, analytic, psi_id=None):
        def _maybe_viz_err(lp_sum):
            if psi_id is None:
                maybe_viz_err(lp_sum, psi_fs)
            else:
                # viz up & down separately
                maybe_viz_err(lp_sum, psi_fs[:len(psi_fs)//2])
                maybe_viz_err(lp_sum, psi_fs[len(psi_fs)//2:])

        lp_sum = 0
        for psi_f in psi_fs:
            lp_sum += np.abs(psi_f)**2
        lp_sum_sum = lp_sum.sum()

        # `1` per bin, minus
        #   - DC bin, since no phi
        #   - half of Nyquist bin, since `analytic=True` cannot ever get a full
        #     Nyquist (Nyquist bin is halved, so even in best case of the peak
        #     placed at Nyquist, we get 0.5). This is intentional, since any
        #     real-valued signal's bin shifted to Nyquist is doubled, which is
        #     then halved in the analytic signal representation. Energy conserves.
        if analytic:
            expected_sum = len(lp_sum) - 1 - 0.5
        else:
            expected_sum = len(lp_sum) - 1
        # scale according to tolerance
        expected_above = expected_sum * (1 + th_sum_above)
        expected_below = expected_sum * (1 - th_sum_below[Q])

        # upper bound
        if not lp_sum_sum < expected_above:
            _maybe_viz_err(lp_sum)
            raise AssertionError((
                "{} >= {}, \n{}\npsi_id={}\nQ_chk={}".format(
                    lp_sum_sum, expected_above, test_params_str, psi_id, Q)))
        # lower bound
        if not lp_sum_sum > expected_below:
            _maybe_viz_err(lp_sum)
            raise AssertionError((
                "{} <= {}, \n{}\npsi_id={}\nQ_chk={}".format(
                    lp_sum_sum, expected_below, test_params_str, psi_id, Q)))

    def get_test_params_str(test_params):
        return '\n'.join(f'{k}={v}' for k, v in test_params.items())

    # run tests ##############################################################

    # in case we decide against
    check_up_to_nyquist = True

    N = 1024
    J = int(np.log2(N))
    T = 2**J  # match J so we can include phi in lp sum

    # test is precision-sensitive but instead of using 'double' we tweak the
    # thresholds since otherwise the test is very long, and results are still
    # valid as the wavelets are precision-insensitive enough in freq domain
    # for purposes here.
    common_params = dict(shape=N, J=J, T=T, Q_fr=2, frontend=default_backend,
                         **pad_mode_kw, precision=default_precision)
    # thresholding empirically chosen to balance reasonable expectations against
    # implementation complexity or achievability
    th_above = .025
    th_below = .5
    th_above_non_cqt = .030
    th_below_non_cqt = .7
    th_below_one_psi = .5
    th_below_nonmax_pad = .5
    # this one should likely depend on `max_pad_factor*` as it's worst for `None`
    # but not in an alarming manner.
    th_below_bin_1 = .985
    th_below_nyquist_region = .65
    th_sum_above = .01
    th_sum_below = {1: .15, 2: .15, 8: .075, 16: .062}
    # assume subsampling works as intended past this;
    # hard to account for all edge cases with incomplete filterbanks
    max_k = 5

    # configure params based on `SKIP_LONG` flag
    C = dict(
        Q=(1, 8, 16),
        r_psi=(np.sqrt(.5), .85),
        max_pad_factor=(None, 1),
        max_pad_factor_fr=(None, 1),
        sampling_filters_fr=('resample', 'exclude', 'recalibrate'),
        analytic=(True, False)
    )
    if skip_long:
        for k, v in C.items():
            C[k] = v[:1]  # keep only 1

    # test loop ##############################################################
    for Q in C['Q']:
      for r_psi in C['r_psi']:
        for max_pad_factor in C['max_pad_factor']:
          for max_pad_factor_fr in C['max_pad_factor_fr']:
            for sampling_filters_fr in C['sampling_filters_fr']:
              # others are duplicate in time; used to skip some tests
              tm_not_duplicate = bool(
                  sampling_filters_fr == 'resample' and max_pad_factor_fr is None)
              for analytic in C['analytic']:
                if not analytic and not tm_not_duplicate:
                    continue
                aligned = bool(sampling_filters_fr != 'recalibrate')
                test_params = dict(Q=Q, r_psi=(r_psi, r_psi), r_psi_fr=r_psi,
                                   analytic=analytic, analytic_fr=analytic,
                                   max_pad_factor=max_pad_factor,
                                   max_pad_factor_fr=max_pad_factor_fr,
                                   sampling_filters_fr=sampling_filters_fr)
                try:
                    with warnings.catch_warnings(record=True) as _:
                        # first determine N_frs to then set max F & J_fr
                        jtfs = TimeFrequencyScattering1D(
                            **common_params, **test_params, aligned=aligned)
                        N_fr_scales_max = jtfs.N_fr_scales_max
                        J_fr = (N_fr_scales_max
                                if sampling_filters_fr != 'recalibrate' else
                                N_fr_scales_max - 1)
                        F = 2**J_fr
                        test_params['F'] = F
                        test_params['J_fr'] = J_fr

                    with warnings.catch_warnings(record=True) as ws:
                        jtfs = TimeFrequencyScattering1D(
                            **common_params, **test_params, aligned=aligned)
                except Exception as e:
                    print(get_test_params_str(test_params))
                    raise e

                # set reused params
                test_params_str = get_test_params_str(test_params)

                warned = bool(any("filter distortion" in str(w.message)
                                  for w in ws))
                if max_pad_factor is None and max_pad_factor_fr is None:
                    assert not warned, test_params_str
                warned_pad = {}
                warned_pad['fr'] = bool(max_pad_factor_fr == 1 and warned)
                warned_pad['tm'] = bool(max_pad_factor == 1 and warned)

                max_pad = {'tm': max_pad_factor is None,
                           'fr': max_pad_factor_fr is None}

                # temporal filterbank ######################################
                if tm_not_duplicate:
                    for psis_idx, psi_fs in enumerate([jtfs.psi1_f,
                                                       jtfs.psi2_f]):
                        for k in psi_fs[-1]:
                            if not isinstance(k, int) or k > max_k:
                                continue
                            # psi & phi
                            lp = tkt.compute_lp_sum(
                                [p[k] for p in psi_fs if k in p],
                                phi_f=None, fold_antianalytic=True)

                            kw = dict(psi_fs=psi_fs, k=k, max_pad=max_pad['tm'])
                            check_above(lp, test_params_str, warned_pad, **kw)
                            check_below(lp, test_params_str, warned_pad,
                                        analytic=analytic, **kw)

                # frequential filterbank ###################################
                if warned_pad['fr']:
                    # bad behavior obtained but warned about
                    pass
                else:
                    for psi_id in jtfs.psi1_f_fr_up:
                        if not isinstance(psi_id, int):
                            continue
                        elif (sampling_filters_fr == 'exclude' and
                                  max_pad_factor_fr == 1 and
                                  len(jtfs.psi1_f_fr_up[psi_id]) <= 3):
                            # implem can't account for this case
                            continue
                        psi_fs_up = jtfs.psi1_f_fr_up[psi_id]
                        psi_fs_dn = jtfs.psi1_f_fr_dn[psi_id]

                        lp = tkt.compute_lp_sum(psi_fs_up + psi_fs_dn, phi_f=None)
                        for s1_fr, psi_fs in enumerate([jtfs.psi1_f_fr_up,
                                                        jtfs.psi1_f_fr_dn]):
                            ckw = dict(psi_fs=psi_fs, psi_id=psi_id,
                                       sampling_filters_fr=sampling_filters_fr,
                                       max_pad=max_pad['fr'])
                            check_above(lp, test_params_str, warned_pad, **ckw)
                            check_below(lp, test_params_str, warned_pad,
                                        analytic=analytic, **ckw)

                # assert same peak values (symmetry) #######################
                for psi_id in jtfs.psi1_f_fr_up:
                    if not isinstance(psi_id, int):
                        continue
                    for i, (p_up, p_dn
                            ) in enumerate(zip(jtfs.psi1_f_fr_up[psi_id],
                                               jtfs.psi1_f_fr_dn[psi_id])):
                        up_mx, dn_mx = p_up.max(), p_dn.max()
                        assert up_mx == dn_mx, (i, up_mx, dn_mx,
                                                "\n%s" % test_params_str)

                # complete filterbank energy conservation ##################
                # assert lp sum sum indicates non-expansiveness.
                # In particular, `analytic=False` may only account for
                # negatives-only, while non-zero positives yield energy
                # expansion.
                # 'exclude' and `k > 0` are exceptions as they by definition
                # compute subsets (while, if the "prior" is accurate, still
                # conserving energy)
                # temporal #######################################
                if tm_not_duplicate:
                    for psis_idx, psi_fs in enumerate([jtfs.psi1_f,
                                                       jtfs.psi2_f]):
                        psi_fs = [p[0] for p in psi_fs]
                        check_lp_sum_sum(psi_fs, test_params_str,
                                         Q=jtfs.Q[psis_idx], analytic=analytic)

                # frequential ####################################
                for psi_id in jtfs.psi1_f_fr_up:
                    if (not isinstance(psi_id, int) or
                        # incomplete filterbank, won't pass
                        (sampling_filters_fr == 'exclude' and psi_id > 0) or
                        # too poorly behaved, don't bother
                        (sampling_filters_fr == 'recalibrate' and psi_id > 1
                         and warned_pad['fr'])):
                        continue
                    psi_fs = (jtfs.psi1_f_fr_up[psi_id] +
                              jtfs.psi1_f_fr_dn[psi_id])
                    check_lp_sum_sum(psi_fs, test_params_str, Q=jtfs.Q_fr,
                                     analytic=analytic, psi_id=psi_id)

    if skip_long:
        warnings.warn("Skipped most of `test_lp_sum()` per SKIPS['long_in_jtfs']")


def test_pack_coeffs_jtfs():
    """Test coefficients are packed as expected."""
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    def validate_n2s(o, info, spin):
        info = info + "\nspin={}".format(spin)
        # ensure 4-dim
        assert o.ndim == 4, "{}{}".format(o.shape, info)

        # pack here directly via arrays, see if they match
        n2s = o[:, :, :, 0]

        # fetch unpadded
        n2s_no_pad = n2s[np.where(n2s != -2)]
        # if phi_t is present, ensure it's the last (and only the last)
        if -1 in n2s:
            n2s0_no_pad = n2s[-1:][np.where(n2s[-1:] != -2)]
            assert np.all(n2s0_no_pad == -1), "{}{}".format(n2s, info)
            assert -1 not in n2s[:-1], "{}{}".format(n2s, info)

        # should never require to pad along `n2` (i.e. fully empty coeff)
        for o_n2 in o:
            assert not np.all(o_n2 == -2), "{}{}".format(o_n2, info)

        # exclude phis
        n2s = n2s[n2s != -1]

        # ensure `n2` is sorted low-to-high (high-to-low freq)
        # no pad & no lowpass
        n2s_np_nlp = n2s_no_pad[np.where(n2s_no_pad != -1)]
        assert np.all(n2s_np_nlp == sorted(n2s_np_nlp, reverse=False)
                      ), "{}{}".format(n2s, info)

    def validate_n1s(o, info, spin, reverse_n1):
        info = info + "\nspin={}".format(spin)
        # ensure `n1`s ordered low to high (high-to-low freq) for every n2, n1_fr
        for n2 in range(len(o)):
            for n1_fr in range(len(o[n2])):
                n1s = o[n2, n1_fr, :, 2]
                if -2 in n1s:
                    # assert right-padded for `reverse_n1=False`, else left-padded
                    n_pad = sum(n1s == -2)
                    n1s_expected_pad = (n1s[-n_pad:] if not reverse_n1 else
                                        n1s[:n_pad])
                    assert np.all(n1s_expected_pad == -2), (
                        "{}, {}{}").format(n_pad, n1s, info)
                # remove padded
                n1s = np.array([n1 for n1 in n1s if n1 != -2])
                assert np.all(n1s == sorted(n1s, reverse=reverse_n1)), (
                    "{}, {}, {}{}").format(n2, n1_fr, n1s, info)

    def validate_spin(out_s, structure, info, up=True):
        info = info + "\nspin={}".format("up" if up else "down")
        dn_is_n1_fr_reversed = bool(structure in (1, 2))
        # check every n1_fr
        for n2 in range(len(out_s)):
            n1_frs = out_s[n2, :, 0, 1]

            # if phi_f is present, ensure it's centered (and len(n1_frs) is odd)
            if -1 in n1_frs:
                if up or not dn_is_n1_fr_reversed:
                    assert n1_frs[-1] == -1, "%s\n%s" % (n1_frs, info)
                else:
                    assert n1_frs[0] == -1, "%s\n%s" % (n1_frs, info)
                # exclude phi_f
                n1_frs = np.array([n1_fr for n1_fr in n1_frs if n1_fr != -1])

            # ensure only psi_f pairs present
            assert -1 not in n1_frs, (n1_frs, info)

            # check padding
            if -2 in n1_frs:
                n_pad = sum(n1_frs == -2)
                if up or not dn_is_n1_fr_reversed:
                    # ensure right-padded
                    assert np.all(n1_frs[-n_pad:] == -2), "%s, %s\n%s" % (
                        n_pad, n1_frs, info)
                else:
                    # ensure left-padded
                    assert np.all(n1_frs[:n_pad]  == -2), "%s, %s\n%s" % (
                        n_pad, n1_frs, info)
                # exclude pad values
                n1_frs = np.array([n1_fr for n1_fr in n1_frs if n1_fr != -2])

            errmsg = "{}{}".format(n1_frs, info)
            if up or not dn_is_n1_fr_reversed:
                assert np.all(n1_frs == sorted(n1_frs)), errmsg
            else:
                assert np.all(n1_frs == sorted(n1_frs, reverse=True)), errmsg

    def validate_packing(out, separate_lowpass, structure, out_exclude,
                         reverse_n1, t, info):
        # unpack into `out_up, out_dn, out_phi`
        out_phi_f, out_phi_t = None, None
        if structure in (1, 2):
            if separate_lowpass:
                out, out_phi_f, out_phi_t = out

            if structure == 1:
                out = out.transpose(1, 0, 2, 3)
                if separate_lowpass:
                    if out_phi_f is not None:
                        out_phi_f = out_phi_f.transpose(1, 0, 2, 3)
                    if out_phi_t is not None:
                        out_phi_t = out_phi_t.transpose(1, 0, 2, 3)

            s = out.shape
            out_up = (out[:, :s[1]//2 + 1] if not separate_lowpass else
                      out[:, :s[1]//2])
            out_dn = out[:, s[1]//2:]

        elif structure == 3:
            if not separate_lowpass:
                out_up, out_dn, out_phi_f = out
            else:
                out_up, out_dn, out_phi_f, out_phi_t = out

        elif structure == 4:
            if not separate_lowpass:
                out_up, out_dn = out
            else:
                out_up, out_dn, out_phi_t = out

        elif structure == 5:
            out_up, out_dn, out_phi_f, out_phi_t, out_phi = out

        # ensure nothing omitted or included by mistake
        phi_t_cond = lambda: (out_phi_t is not None if separate_lowpass else
                              out_phi_t is None)
        if out_exclude is None:
            if structure in (1, 2):
                assert (out_phi_f is not None if separate_lowpass else
                        out_phi_f is None), info
                assert phi_t_cond(), info
            elif structure == 3:
                assert out_phi_f is not None, info
                assert phi_t_cond(), info
            elif structure == 4:
                assert out_phi_f is None, info
                assert phi_t_cond(), info
            elif structure == 5:
                assert out_phi_f is not None, info
                assert out_phi_t is not None, info

        # ensure sliced properly
        assert out_up.shape == out_dn.shape, (
            "{} != {}{}").format(out_up.shape, out_dn.shape, info)

        # do validation ######################################################
        # n1s and n2s
        outs = (out_up, out_dn, out_phi_f, out_phi_t)
        for spin, o in zip([1, -1, 0, 0], outs):
            if o is not None:
                assert o.shape[-1] == t, (o.shape, t)
                validate_n2s(o, info, spin)
                validate_n1s(o, info, spin, reverse_n1)

        # n1_frs
        validate_spin(out_up, structure, info, up=True)
        validate_spin(out_dn, structure, info, up=False)

        # `phi_f`
        if out_phi_f is not None:
            out_phi_f_n1_fr = out_phi_f[:, :, :, 1]
            # exclude pad
            out_phi_f_n1_fr[out_phi_f_n1_fr == -2] = -1
            assert np.all(out_phi_f_n1_fr == -1), (out_phi_f_n1_fr, info)

        # `phi_t`
        if out_phi_t is not None:
            out_phi_t_n2 = out_phi_t[:, :, :, 0]
            # exclude pad
            out_phi_t_n2[out_phi_t_n2 == -2] = -1
            assert np.all(out_phi_t_n2 == -1), (out_phi_t_n2, info)

    # end of helper methods ##################################################
    # test
    tests_params = {
        1: dict(average=True, average_fr=True,  aligned=True,  out_3D=True),
        0: dict(average=True, average_fr=True,  aligned=False, out_3D=False),
        4: dict(average=True, average_fr=False, aligned=True,  out_3D=False),
    }

    for test_num, test_params in tests_params.items():
        _, out_stored, params, _, meta = load_data(test_num)
        ref_key = list(out_stored)[0]
        t = out_stored[ref_key][0].shape[-1]

        # ensure match
        for k in test_params:
            if k != 'average':
                assert test_params[k] == params[k]
        test_params['sampling_psi_fr'] = ('resample' if test_num != 0 else
                                          'exclude')

        # pack into 'dict:list'
        paired_flat0 = {pair: [{'coef': c} for c in v]
                        for pair, v in out_stored.items()}

        for separate_lowpass in (False, True):
          for structure in (1, 2, 3, 4, 5):
            if structure == 5 and separate_lowpass:
                continue  # invalid
            for reverse_n1 in (False, True):
              for out_exclude in (None, True):
                  if out_exclude is not None:
                      if not separate_lowpass:
                          continue  # invalid
                      else:
                          pairs_lp = ('psi_t * phi_f', 'phi_t * psi_f',
                                      'phi_t * phi_f')
                          out_exclude = (pairs_lp[-3:] if structure != 4 else
                                         pairs_lp[-2:])

                  kw = {k: v for k, v in test_params.items()
                        if k in ('sampling_psi_fr',)}
                  test_params_str = "\n".join(f'{k}={v}'
                                              for k, v in test_params.items())
                  info = ("\nstructure={}\nseparate_lowpass={}\nreverse_n1={}\n{}"
                          ).format(structure, separate_lowpass, reverse_n1,
                                   test_params_str)

                  # exclude pairs if needed
                  if out_exclude is None:
                      paired_flat = paired_flat0
                  else:
                      paired_flat = {}
                      for pair in paired_flat0:
                          if pair not in out_exclude:
                              paired_flat[pair] = paired_flat0[pair]

                  # pack & validate
                  out = tkt.pack_coeffs_jtfs(
                      paired_flat, meta, structure=structure,
                      separate_lowpass=separate_lowpass,
                      reverse_n1=reverse_n1,
                      out_3D=test_params['out_3D'],
                      **kw, debug=True)
                  validate_packing(out, separate_lowpass, structure, out_exclude,
                                   reverse_n1, t, info)


def test_energy_conservation():
    """E_out ~= E_in.

    Refer to:
     - https://github.com/kymatio/kymatio/discussions/732#discussioncomment-864097
     - https://github.com/kymatio/kymatio/discussions/732#discussioncomment-855701
     - https://github.com/kymatio/kymatio/discussions/732#discussioncomment-855702

    Limited by `out_3D=False`; see `help(toolkit.est_energy_conservation)`.
    For bounding energy ratios above by `1`, also limited to `aligned=True`
    per unpad aliasing; see `_energy_correction()` in `core`.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    np.random.seed(0)
    # 8.5 on dyadic scale to test time unpad correction
    N = 362
    x = np.random.randn(N)

    # configure ~tight frame; configured also for speed, not optimized for I/O
    # https://github.com/kymatio/kymatio/discussions/732#discussioncomment-855703
    # https://github.com/kymatio/kymatio/discussions/732#discussioncomment-968384
    r_psi = (.9, .9)
    r_psi_fr = .9
    J = int(np.ceil(np.log2(N)) - 2)
    J_fr = 3
    F = 2**J_fr
    T = 2**J
    Q = (8, 3)
    Q_fr = 4
    params = dict(
        shape=N, J=J, J_fr=J_fr, Q=Q, Q_fr=Q_fr, F=F, T=T,
        r_psi=r_psi, r_psi_fr=r_psi_fr, average_fr=False, out_3D=False,
        sampling_filters_fr='resample', max_pad_factor=None,
        max_pad_factor_fr=None, pad_mode='reflect', aligned=True,
        pad_mode_fr='conj-reflect-zero', out_type='dict:list',
        frontend=default_backend, precision=default_precision,
    )
    with IgnoreWarnings("`r_psi2`"):
        jtfs_a = TimeFrequencyScattering1D(**params, average=True)
        jtfs_u = TimeFrequencyScattering1D(**params, average=False)
    jmeta_a = jtfs_a.meta()
    jmeta_u = jtfs_u.meta()

    # scatter
    Scx_a = jtfs_a(x)
    Scx_u = jtfs_u(x)

    # compute energies
    pairs = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
             'psi_t * psi_f_up', 'psi_t * psi_f_dn')
    kw = dict(kind='l2', plots=False)
    _, pair_energies_a = energy_profile_jtfs(Scx_a, jmeta_a, **kw, pairs=pairs)
    _, pair_energies_u = energy_profile_jtfs(Scx_u, jmeta_u, **kw, pairs=pairs)

    pe_a, pe_u = [{pair: np.sum(pe[pair]) for pair in pe}
                  for pe in (pair_energies_a, pair_energies_u)]

    # compute energy relations ###############################################
    E = {}
    E['in'] = energy(x)
    E['out'] = (np.sum([v for pair, v in pe_u.items()
                        if pair not in ('S0', 'S1')]) +
                pe_a['S0'])

    E['S0'] = pe_a['S0']
    E['S1'] = pe_a['S1']
    E['U1'] = pe_u['S1']
    # U2 + S2 = U1 - S1 --> U2 = (U1 - S1) - S2
    E['U2'] = (E['U1'] - E['S1']) - pe_u['psi_t * phi_f']
    E['S1_joint'] = (pe_u['phi_t * phi_f'] +
                     pe_u['phi_t * psi_f'])
    E['U2_joint'] = (pe_u['psi_t * psi_f_up'] +
                     pe_u['psi_t * psi_f_dn'])

    r = {}
    r['out / in'] = E['out'] / E['in']
    r['(S0 + U1) / in'] = (E['S0'] + E['U1']) / E['in']
    r['S1_joint / S1'] = E['S1_joint'] / E['S1']
    r['U2_joint / U2'] = E['U2_joint'] / E['U2']

    if metric_verbose:
        print("\nEnergy conservation (w/ ~tight frame):")
        for k, v in r.items():
            print("{:.4f} -- {}".format(v, k))

    # run assertions #########################################################
    assert .93 < r['out / in']       < 1., r['out / in']
    assert .97 < r['(S0 + U1) / in'] < 1., r['(S0 + U1) / in']
    assert .95 < r['S1_joint / S1']  < 1., r['S1_joint / S1']
    assert .91 < r['U2_joint / U2']  < 1., r['U2_joint / U2']


def test_est_energy_conservation():
    """Tests that `toolkit.est_energy_conservation` doesn't error, and that
    values are sensible.

    Limited by `out_3D=False`; see `help(toolkit.est_energy_conservation)`.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    N = 256
    x = np.random.randn(N)

    kw = dict(analytic=1, backend=default_backend, precision=default_precision)
    with IgnoreWarnings("`r_psi2`"):
        print()
        ESr0 = tkt.est_energy_conservation(x, jtfs=0, **kw, verbose=1)
        print()
        ESr1 = tkt.est_energy_conservation(x, jtfs=1, aligned=1, **kw, verbose=1)

    for ESr in (ESr0, ESr1):
        for k, v in ESr.items():
            tol = .03  # random with torch cuda default
            assert 0 < v < 1 + tol, (k, v)

    # coverage
    kw.pop('backend')
    with IgnoreWarnings("`r_psi2`"):
        _ = tkt.est_energy_conservation(x, **kw, verbose=0)


def test_implementation():
    """Test that every `implementation` option works."""
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    N = 512
    x = echirp(N)

    for implementation in range(1, 6):
        jtfs = TimeFrequencyScattering1D(shape=N, J=4, Q=2,
                                         implementation=implementation,
                                         frontend=default_backend,
                                         precision=default_precision)
        assert jtfs.implementation == implementation, (
            jtfs.implementation, implementation)
        _ = jtfs(x)


def test_pad_mode_fr():
    """Test that functional `pad_mode_fr` works."""
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    from wavespin.scattering1d.core.timefrequency_scattering1d import _right_pad
    N = 512
    x = echirp(N)

    kw = dict(shape=N, J=4, Q=2, frontend=default_backend, out_type='array',
              max_pad_factor_fr=1, precision=default_precision)
    jtfs0 = TimeFrequencyScattering1D(**kw, pad_mode_fr='zero')
    jtfs1 = TimeFrequencyScattering1D(**kw, pad_mode_fr=_right_pad)

    out0 = jtfs0(x)
    out1 = jtfs1(x)
    assert np.allclose(out0, out1)


def test_no_second_order_filters():
    """Reproduce edge case: configuration yields no second-order wavelets
    so can't do JTFS.

    Also tests for insufficient number of first-order filters for `smart_paths`.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    ckw = dict(shape=8192, r_psi=(.9, .9), frontend=default_backend,
               precision=default_precision)

    # improved `j` design made raising this exception difficult without breaking
    # something else first, and without modifying criterion_amplitude,
    # so just keep the code here for future reference
    # with pytest.raises(ValueError) as record:
    #     _ = TimeFrequencyScattering1D(**ckw, Q=(3, 1))
    # assert "no second-order filters" in record.value.args[0]

    with pytest.raises(Exception) as record:
        with IgnoreWarnings("`r_psi2`"):
            _ = TimeFrequencyScattering1D(**ckw, J=1, Q=(3, 3))
    msg = record.value.args[0]
    assert "two first-order CQT filters" in msg, msg

    with pytest.raises(Exception) as record:
        with IgnoreWarnings("`r_psi2`"):
            _ = TimeFrequencyScattering1D(**ckw, J=2, Q=(1, 3))
    msg = record.value.args[0]
    assert "failed to produce any joint" in msg, msg


def test_backends():
    """Test basic functionality with PyTorch and TensorFlow backends.

    Also test that `pack_coeffs_jtfs`
      - works with batched inputs
      - conserves energy
      - does not modify input (leaves original JTFS output tensor values unchanged)
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    for backend_name in ('torch', 'tensorflow', 'jax'):
        if cant_import(backend_name):
            continue
        elif backend_name == 'torch':
            import torch
        elif backend_name == 'tensorflow':
            import tensorflow as tf
        elif backend_name == 'jax':
            import jax

        N = 2048
        x = echirp(N)
        x = np.vstack([x, x, x])
        if backend_name == 'torch':
            x = torch.from_numpy(x)
        elif backend_name == 'tensorflow':
            x = tf.convert_to_tensor(x)
        elif backend_name == 'jax':
            x = jax.numpy.array(x)

        jtfs = TimeFrequencyScattering1D(
            shape=N, J=(8, 6), Q=8, J_fr=3, Q_fr=1,
            average_fr=True, out_type='dict:array', out_3D=True,
            max_pad_factor=2, max_pad_factor_fr=None,
            pad_mode='reflect', pad_mode_fr='conj-reflect-zero',
            frontend=backend_name, precision=default_precision)

        Scx = jtfs(x)
        jmeta = jtfs.meta()

        # test batched packing for convenience ###############################
        for structure in (1, 2, 3, 4):
            for separate_lowpass in (False, True):
                kw = dict(meta=jmeta, structure=structure,
                          separate_lowpass=separate_lowpass,
                          sampling_psi_fr=jtfs.sampling_psi_fr)
                # keep original copy
                Scxnc  = deepcopy(tkt.jtfs_to_numpy(Scx))
                outs   = tkt.pack_coeffs_jtfs(Scx, **kw)
                outs0  = tkt.pack_coeffs_jtfs(Scx, **kw, sample_idx=0)
                outsn  = tkt.pack_coeffs_jtfs(tkt.jtfs_to_numpy(Scx), **kw)
                outs0n = tkt.pack_coeffs_jtfs(tkt.jtfs_to_numpy(Scx), **kw,
                                              sample_idx=0)
                outs   = outs   if isinstance(outs,  tuple) else [outs]
                outs0  = outs0  if isinstance(outs0, tuple) else [outs0]
                outsn  = outsn  if isinstance(outs,  tuple) else [outsn]
                outs0n = outs0n if isinstance(outs0, tuple) else [outs0n]

                # ensure methods haven't altered original array ##############
                # (with e.g. inplace ops) ####################################
                for pair in Scx:
                    coef = npy(Scx[pair])
                    assert np.allclose(coef, Scxnc[pair]), pair

                # shape and value checks #####################################
                for o, o0, on, o0n in zip(outs, outs0, outsn, outs0n):
                    assert o.ndim == 5, o.shape
                    assert len(o) == len(x), (len(o), len(x))
                    assert o.shape[-1] == Scx['S0'].shape[-1], (
                        o.shape, Scx['S0'].shape)
                    assert o.shape[1:] == o0.shape, (o.shape, o0.shape)
                    assert np.allclose(npy(o), on)
                    assert np.allclose(npy(o0), o0n)

                # E_in == E_out ##############################################
                _test_packing_energy_io(Scx, outs, structure, separate_lowpass)

        ######################################################################

        Scx = tkt.jtfs_to_numpy(Scx)
        E_up = tkt.coeff_energy(Scx, jmeta, pair='psi_t * psi_f_up')
        E_dn = tkt.coeff_energy(Scx, jmeta, pair='psi_t * psi_f_dn')
        th = 32
        assert E_dn / E_up > th, "{:.2f} < {}".format(E_dn / E_up, th)


def _test_packing_energy_io(Scx, outs, structure, separate_lowpass):
    def phi_t_energies(out_phi_t, structure, separate_lowpass):
        if structure == 1:
            B = ExtendedUnifiedBackend(out_phi_t)
            out_phi_t = B.transpose(out_phi_t, (0, 2, 1, 3, 4))
        s = tuple(out_phi_t.shape)
        # out_phi_t includes `phi_t * phi_f`
        phi_t_packed_twice = bool((structure in (1, 2)) or
                                  (structure in (3, 4) and not separate_lowpass))
        if phi_t_packed_twice:
            E_out_phi = energy(out_phi_t[:, :, s[2]//2])
        else:
            E_out_phi = energy(out_phi_t[:, :, -1])
        E_out_phi_t = energy(out_phi_t) - E_out_phi
        return E_out_phi, E_out_phi_t

    def check_phi_t_energies(out_phi_t, structure, separate_lowpass):
        E_out_phi, E_out_phi_t = phi_t_energies(out_phi_t, structure,
                                                separate_lowpass)
        assert np.allclose(E_in_phi_t, E_out_phi_t), (E_in_phi_t, E_out_phi_t)
        assert np.allclose(E_in_phi, E_out_phi), (E_in_phi, E_out_phi)
        return E_out_phi, E_out_phi_t

    E_in_up = energy(Scx['psi_t * psi_f_up'])
    E_in_dn = energy(Scx['psi_t * psi_f_dn'])
    E_in_spinned = E_in_up + E_in_dn
    E_in_phi_t = energy(Scx['phi_t * psi_f'])
    E_in_phi_f = energy(Scx['psi_t * phi_f'])
    E_in_phi   = energy(Scx['phi_t * phi_f'])
    E_in = E_in_spinned + E_in_phi_t + E_in_phi_f + E_in_phi

    if structure in (1, 2):
        if separate_lowpass:
            out_spinned, out_phi_f, out_phi_t = outs
            E_out_spinned = energy(out_spinned)
            assert np.allclose(E_in_spinned, E_out_spinned
                               ), (E_in_spinned, E_out_spinned)

            E_out_phi, E_out_phi_t = check_phi_t_energies(out_phi_t, structure,
                                                          separate_lowpass)

            E_out_phi_f = energy(out_phi_f)
            assert np.allclose(E_in_phi_f, E_out_phi_f), (E_in_phi_f, E_out_phi_f)

            E_out = E_out_spinned + E_out_phi_f + E_out_phi + E_out_phi_t
        else:
            E_out = energy(outs[0])

    elif structure == 3:
        if separate_lowpass:
            """E(outs) == E_in_spinned + E_in_phi_t + E_in_phi_f + 2*E_in_phi"""
            out_up, out_dn, out_phi_f, out_phi_t = outs

            # method properly splits `phi_t * phi_f` and `phi_t * psi_f` energies
            # so no duplication in `E_out`
            E_out_phi, E_out_phi_t = check_phi_t_energies(out_phi_t, structure,
                                                          separate_lowpass)
        else:
            """E(outs) == E_in_spinned + E_in_phi_t + E_in_phi_f + 2*E_in_phi"""
            out_up, out_dn, out_phi_f = outs
            # `phi_t * psi_f` packed along `psi_t * psi_f`
            E_in_spinned += E_in_phi_t

        E_out_spinned = energy(out_up) + energy(out_dn)
        assert np.allclose(E_in_spinned, E_out_spinned
                           ), (E_in_spinned, E_out_spinned)

        E_out_phi_f = energy(out_phi_f)
        # `phi_t * phi_f` is packed with `psi_t * phi_f`
        E_in_phi_f += E_in_phi
        assert np.allclose(E_in_phi_f, E_out_phi_f), (E_in_phi_f, E_out_phi_f)

        E_out = E_out_spinned + E_out_phi_f
        if separate_lowpass:
            E_out += E_out_phi_t

    elif structure == 4:
        if separate_lowpass:
            out_up, out_dn, out_phi_t = outs
            # `psi_t * phi_f` packed for each spin
            E_in_spinned += 2*E_in_phi_f
            E_out_spinned = energy(out_up) + energy(out_dn)

            E_out_phi, E_out_phi_t = check_phi_t_energies(out_phi_t, structure,
                                                          separate_lowpass)
        else:
            out_up, out_dn = outs
            # phi_t and phi_f pairs included
            # only `phi_t * phi_f` and `psi_t * phi_f` energy duped
            E_in_spinned += E_in_phi_t + 2*(E_in_phi + E_in_phi_f)
            E_out_spinned = energy(out_up) + energy(out_dn)

        assert np.allclose(E_in_spinned, E_out_spinned
                           ), (E_in_spinned, E_out_spinned)

        E_out = E_out_spinned
        E_in = E_in_spinned
        if separate_lowpass:
            E_out += (E_out_phi + E_out_phi_t)
            E_in  += (E_in_phi + E_in_phi_t)

    elif structure == 5:
        out_up, out_dn, out_phi_f, out_phi_t, out_phi = outs
        E_out_up, E_out_dn, E_out_phi_f, E_out_phi_t, E_out_phi = [
            energy(o) for o in (out_up, out_dn, out_phi_f, out_phi_t, out_phi)]

        assert np.allclose(E_out_up,    E_in_up),    (E_out_up,    E_in_up)
        assert np.allclose(E_out_dn,    E_in_dn),    (E_out_dn,    E_in_dn)
        assert np.allclose(E_out_phi_f, E_in_phi_f), (E_out_phi_f, E_in_phi_f)
        assert np.allclose(E_out_phi_t, E_in_phi_t), (E_out_phi_t, E_in_phi_t)
        assert np.allclose(E_out_phi,   E_in_phi),   (E_out_phi,   E_in_phi)

        E_out = E_out_up + E_out_dn + E_out_phi_f + E_out_phi_t + E_out_phi

    assert np.allclose(E_in, E_out), (E_in, E_out, structure, separate_lowpass)


def test_differentiability_torch():
    """Tests whether JTFS is differentiable in PyTorch backend.
    Does NOT test whether the gradients are correct.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    if cant_import('torch'):
        return None if run_without_pytest else pytest.skip()
    import torch
    if backend_has_gpu('torch'):
        devices = ['cuda', 'cpu']
    else:
        devices = ['cpu']

    J = 6
    Q = 8
    N = 2**12
    for device in devices:
        jtfs = TimeFrequencyScattering1D(N, J, Q, frontend='torch',
                                         out_type='array').to(device)
        x = torch.randn(2, N, requires_grad=True, device=device)

        s = jtfs.forward(x)
        loss = torch.sum(torch.abs(s))
        loss.backward()
        assert torch.max(torch.abs(x.grad)) > 0.


def test_reconstruction_torch():
    """Test that input reconstruction via backprop has decreasing loss."""
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    if cant_import('torch'):
        return None if run_without_pytest else pytest.skip()
    import torch
    device = 'cuda' if backend_has_gpu('torch') else 'cpu'

    J = 6
    Q = 8
    N = 1024
    n_iters = 30
    jtfs = TimeFrequencyScattering1D(N, J, Q, J_fr=4, Q_fr=2, average_fr=False,
                                     frontend='torch', out_type='array',
                                     sampling_filters_fr=('exclude', 'resample'),
                                     max_pad_factor=1, max_pad_factor_fr=2,
                                     pad_mode_fr='conj-reflect-zero',
                                     ).to(device)

    y = torch.from_numpy(echirp(N, fmin=1).astype('float32')).to(device)
    Sy = jtfs(y)
    div = Sy.max()
    Sy /= div

    torch.manual_seed(0)
    x = torch.randn(N, device=device)
    x /= torch.max(torch.abs(x))
    x.requires_grad = True
    optimizer = torch.optim.SGD([x], lr=140000, momentum=.9, nesterov=True)
    loss_fn = torch.nn.MSELoss()

    losses, losses_recon = [], []
    for i in range(n_iters):
        optimizer.zero_grad()
        Sx = jtfs(x)
        Sx /= div
        loss = loss_fn(Sx, Sy)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().numpy()))
        xn, yn = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        losses_recon.append(float(tkt.rel_l2(yn, xn)))

    # unsure why CPU's worse
    th           = 1.5e-5 if device == 'cuda' else 2e-5
    th_end_ratio = 45     if device == 'cuda' else 30
    th_recon = 1.06
    end_ratio = losses[0] / losses[-1]
    assert end_ratio > th_end_ratio, end_ratio
    assert min(losses) < th, "{:.2e} > {}".format(min(losses), th)
    assert min(losses_recon) < th_recon, "{:.2e} > {}".format(min(losses_recon),
                                                              th_recon)
    if metric_verbose:
        print(("\nReconstruction (torch):\n(end_start_ratio, min_loss, "
               "min_loss_recon) = ({:.1f}, {:.2e}, {:.2f})").format(
                   end_ratio, min(losses), min(losses_recon)))


def test_batch_shape_agnostic():
    """Test that tuple `batch_size` in `x.shape == (*batch_size, N)` works."""
    if cant_import(default_backend):
        return None if run_without_pytest else pytest.skip()

    # get backend module and library object
    if default_backend == 'torch':
        import torch as backend_obj
    elif default_backend == 'tensorflow':
        import tensorflow as backend_obj
    else:
        backend_obj = np

    batch_size = (2, 3, 1, 3)
    N = 128
    x = np.random.randn(*batch_size, N)
    common_params = dict(shape=N, J=3, Q=4, Q_fr=2,
                         max_pad_factor=0, max_pad_factor_fr=0,
                         frontend=default_backend, precision=default_precision)

    for out_3D in (False, True):
      for average_fr in (False, True):
        if out_3D and not average_fr:
            continue  # invalid
        for pad_mode_fr in ('zero', 'conj-reflect-zero'):
          with IgnoreWarnings("boundary effects"):
              jtfs = TimeFrequencyScattering1D(
                  **common_params, out_3D=out_3D, pad_mode_fr=pad_mode_fr,
                  average_fr=average_fr)

          for out_type in ('array', 'list', 'dict:array', 'dict:list'):
            # take JTFS ######################################################
            jtfs.out_type = out_type
            test_params = dict(out_3D=out_3D, pad_mode_fr=pad_mode_fr,
                               out_type=out_type, average_fr=average_fr)
            test_params_str = '\n'.join(f"{k}={v}"
                                        for k, v in test_params.items())

            out = jtfs(x)

            # check ##########################################################
            def assert_fn(coef, info=''):
                errmsg = "coef.shape={}, batch_size={}, {}\n{}".format(
                    coef.shape, batch_size, info, test_params_str)
                assert coef.shape[:len(batch_size)] == batch_size, errmsg

            if out_type == 'array':
                if not out_3D:
                    out = [out]
                for o in out:
                    assert_fn(o)
            elif out_type == 'list':
                if not out_3D:
                    out = [out]
                for o in out:
                    for i, c in enumerate(o):
                        assert_fn(c['coef'], info=f"i={i}")
            elif out_type == 'dict:array':
                for pair in out:
                    assert_fn(out[pair], info=f"pair={pair}")
            elif out_type == 'dict:list':
                for pair in out:
                    for i, c in enumerate(out[pair]):
                        assert_fn(out[pair][i]['coef'],
                                  info=f"pair={pair}, i={i}")

            # test `out_structure` ###########################################
            if 'dict' in out_type:
                # first undo batch reshaping
                bs_ndim = len(batch_size)
                for pair in out:
                    coeffs = out[pair]
                    if out_type == 'dict:array':
                        out[pair] = backend_obj.reshape(
                            coeffs, (-1, *coeffs.shape[bs_ndim:]))
                    elif out_type == 'dict:list':
                        for i, c in enumerate(coeffs):
                            c = c['coef']
                            out[pair][i]['coef'] = backend_obj.reshape(
                                c, (-1, *c.shape[bs_ndim:]))
                out_orig = out

                for out_structure in (1, 2, 3, 4, 5):
                    jtfs.out_structure = out_structure
                    out = jtfs._post_scattering(out_orig, batch_size, backend_obj)
                    if isinstance(out, tuple):
                        for o in out:
                            assert_fn(o, f"out_structure={out_structure}")
                    else:
                        assert_fn(out, f"out_structure={out_structure}")
                # restore `out_structure`
                jtfs.out_structure = None


def test_out_type():
    """Test that 'list' and 'array' `out_type` return structures that work as
    expected, including coefficient values and shapes agreement, and shapes
    agreement with metas.

    'dict:list' and 'dict:array' are tested extensively in `test_meta()`,
    minus the coefficient values agreement since the two output modes differ
    only in concatenation (same here but we check anyway).
    """
    def test_array_vs_list(jtfs):
        jtfs.out_type = 'list'
        s_ls = jtfs(x)
        jtfs.out_type = 'array'
        s_arr = jtfs(x)

        jmeta = jtfs.meta()
        # bundled test: info()
        _ = jtfs.info(show=False)

        if jtfs.out_3D:
            # list -> arr
            s_ls_order01, s_ls_order2 = [[npy(c['coef']) for c in s_ls_o]
                                         for s_ls_o in s_ls]
            # concatenate since joining along existing dim:
            # (batch_size, n_freqs0 + n_freqs1, time)
            ls_arr01 = np.concatenate(s_ls_order01, axis=1)
            # stack since must extend dim:
            # (batch_size, n_freqs, time) ->
            # (batch_size, mixed, n_freqs, time)
            ls_arr2 = np.stack(s_ls_order2, axis=1)

            # arr: check, numpify
            assert isinstance(s_arr, tuple) and len(s_arr) == 2, (
                type(s_arr), len(s_arr))
            arr01, arr2 = npy(s_arr[0]), npy(s_arr[1])

            # compare output shapes & values
            assert ls_arr01.shape == arr01.shape, (ls_arr01.shape, arr01.shape)
            assert ls_arr2.shape == arr2.shape, (ls_arr2.shape, arr2.shape)
            assert np.allclose(ls_arr01, arr01)
            assert np.allclose(ls_arr2, arr2)

            # output shapes == meta shapes
            jmeta01, jmeta2 = jmeta
            for field in jmeta01:
                if field == 'key':
                    continue
                assert arr01.shape[1] == jmeta01[field].shape[0], (
                    arr01.shape, jmeta01[field].shape, field)
                assert arr2.shape[1] == jmeta2[field].shape[0], (
                    arr2.shape, jmeta2[field].shape, field)

        else:
            # list -> arr
            s_ls_values = [npy(c['coef']) for c in s_ls]
            ls_arr = np.concatenate(s_ls_values, axis=1)

            # arr: numpify
            arr = npy(s_arr)

            # compare output shapes & values
            assert ls_arr.shape == arr.shape, (ls_arr.shape, arr.shape)
            assert np.allclose(ls_arr, arr)

            # output shape == meta shape
            for field in jmeta:
                if field == 'key':
                    continue
                assert arr.shape[1] == jmeta[field].shape[0], (
                    arr.shape, jmeta[field].shape, field)

    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    N = 512
    x = np.random.randn(N)

    # make scattering objects
    ckw = dict(shape=N, J=int(np.log2(N)) - 2, Q=8, J_fr=5, Q_fr=2, F=4,
               **pad_mode_kw, frontend=default_backend,
               precision=default_precision)

    for out_3D in (False, True):
        for average_fr in (True, False):
            if out_3D and not average_fr:
                continue  # invalid configuration
            for aligned in (True, False):
                test_params = dict(out_3D=out_3D, average_fr=average_fr,
                                   aligned=aligned)
                jtfs = TimeFrequencyScattering1D(**ckw, **test_params)
                test_array_vs_list(jtfs)


def test_meta():
    """Tests that meta values and structures match those of output for all
    combinations of
        - out_3D (True only with average_fr=True and sampling_psi_fr != 'exclude')
        - average_fr
        - average
        - aligned
        - sampling_psi_fr
        - sampling_phi_fr
    and, partially
        - average_global
        - average_global_phi
        - average_global_fr
        - average_global_fr_phi
    a total of 60 tests.

    Not tested:
        - oversampling
        - oversampling_fr
        - max_pad_factor
        - max_pad_factor_fr

    For compute convenience, also tests that `validate_filterbank_tm` and
    `validate_filterbank_fr` run without error on each configuration.
    """
    def assert_equal_lengths(Scx, jmeta, field, pair, out_3D, test_params_str,
                             jtfs):
        """Assert that number of coefficients and frequency rows for each match"""
        if out_3D:
            out_n_coeffs  = len(Scx[pair])
            out_n_freqs   = sum(len(c['coef'][0]) for c in Scx[pair])
            meta_n_coeffs = len(jmeta[field][pair])
            meta_n_freqs  = np.prod(jmeta[field][pair].shape[:2])

            assert out_n_coeffs == meta_n_coeffs, (
                "len(out[{0}]), len(jmeta[{1}][{0}]) = {2}, {3}\n{4}"
                ).format(pair, field, out_n_coeffs, meta_n_coeffs,
                         test_params_str)
        else:
            out_n_freqs  = sum(c['coef'].shape[1] for c in Scx[pair])
            meta_n_freqs = len(jmeta[field][pair])

        assert out_n_freqs == meta_n_freqs, (
            "out vs meta n_freqs mismatch for {}, {}: {} != {}\n{}".format(
                pair, field, out_n_freqs, meta_n_freqs, test_params_str))

    def assert_equal_values(Scx, jmeta, field, pair, i, meta_idx, out_3D,
                            test_params_str, test_params, jtfs):
        """Assert that non-NaN values are equal."""
        a, b = Scx[pair][i][field], jmeta[field][pair][meta_idx[0]]
        errmsg = ("(out[{0}][{1}][{2}], jmeta[{2}][{0}][{3}]) = ({4}, {5})\n{6}"
                  ).format(pair, i, field, meta_idx[0], a, b, test_params_str)

        meta_len = b.shape[-1] if field != 'spin' else 1
        zeroth_order_unaveraged = bool(pair == 'S0' and
                                       not test_params['average'])
        if field not in ('spin', 'stride'):
            assert meta_len == 3, ("all meta fields (except spin, stride) must "
                                   "pad to length 3: %s" % errmsg)
            if not zeroth_order_unaveraged:
                assert len(a) > 0, ("all computed metas (except spin) must "
                                    "append something: %s" % errmsg)

        if field == 'stride':
            assert meta_len == 2, ("'stride' meta length must be 2 "
                                   "(got meta: %s)" % b)
            if pair in ('S0', 'S1'):
                if pair == 'S1' or test_params['average']:
                    assert len(a) == 1, errmsg
                if pair == 'S0' and not test_params['average']:
                    assert a == (), errmsg
                    assert np.all(np.isnan(b)), errmsg
                else:
                    assert a == b[..., 1], errmsg
                    assert np.isnan(b[..., 0]), errmsg
            else:
                assert len(a) == 2, errmsg
                assert np.all(a == b), errmsg
                assert not np.any(np.isnan(b)), errmsg

        elif (field == 'spin' and pair in ('S0', 'S1')) or zeroth_order_unaveraged:
            assert len(a) == 0 and np.all(np.isnan(b)), errmsg

        elif len(a) == meta_len:
            assert np.all(a == b), errmsg

        elif len(a) < meta_len:
            # S0 & S1 have one meta entry per coeff so we pad to joint's len
            if np.all(np.isnan(b[:2])):
                assert pair in ('S0', 'S1'), errmsg
                assert a[0] == b[..., 2], errmsg
            # joint meta is len 3 but at compute 2 is appended
            elif len(a) == 2 and meta_len == 3:
                assert pair not in ('S0', 'S1'), errmsg
                assert np.all(a[:2] == b[..., :2]), errmsg

        else:
            # must meet one of above behaviors
            raise AssertionError(errmsg)

        # increment meta_idx for next check (manual 'key')
        meta_idx_prev = meta_idx[0]  # store for ref
        if pair in ('S0', 'S1') or out_3D:
            meta_idx[0] += 1
        else:
            # increment by number of frequential rows (i.e. `n1`) since
            # these n1-meta aren't appended in computation
            n_freqs = Scx[pair][i]['coef'].shape[1]
            meta_idx[0] += n_freqs

        # check 'key'
        if pair in ('S0', 'S1'):
            n_freqs_coeff = 1
        else:
            n_freqs_coeff = Scx[pair][i]['coef'].shape[1]
        key = jmeta['key'][pair][i]
        expected = (meta_idx_prev if not out_3D else
                    meta_idx_prev * n_freqs_coeff)  # since (n_slices, n_freqs, t)
        assert key.start == expected, (key.start, expected, pair, i)
        n_freqs_key = key.stop - key.start
        assert n_freqs_key == n_freqs_coeff, (n_freqs_key, n_freqs_coeff, pair, i)

    def assert_aligned_stride(Scx, test_params_str, jtfs):
        """Assert all frequential strides are equal in `aligned`."""
        ref_stride = Scx['psi_t * psi_f_up'][0]['stride'][0]
        for pair in Scx:
            # skip non second order
            if pair in ('S0', 'S1'):
                continue
            # 'phi_f' exempt from alignment in this case
            elif 'phi_f' in pair and (not jtfs.average_fr and
                                      jtfs.scf.average_fr_global_phi):
                continue
            for i, c in enumerate(Scx[pair]):
                s = c['stride'][0]
                assert s == ref_stride, (
                    "Scx[{}][{}]['stride'] = {} != ref_stride == {}\n{}".format(
                        pair, i, s, ref_stride, test_params_str))

    def run_test(params, test_params, assert_fn=None):
        jtfs = TimeFrequencyScattering1D(**params, **test_params, **pad_kw,
                                         frontend=default_backend,
                                         precision=default_precision)
        if assert_fn is not None:
            assert_fn(jtfs)

        test_params_str = '\n'.join(f'{k}={v}' for k, v in test_params.items())
        _ = tkt.validate_filterbank_tm(jtfs, verbose=0)
        _ = tkt.validate_filterbank_fr(jtfs, verbose=0)

        sampling_psi_fr = test_params['sampling_filters_fr'][0]
        if sampling_psi_fr in ('recalibrate', 'exclude'):
            # assert not all J_pad_fr are same so test covers this case
            # psi is dominant here as `2**J_fr > F`
            assert_pad_difference(jtfs, test_params_str)

        try:
            Scx = jtfs(x)
            jmeta = jtfs.meta()
        except Exception as e:
            print("Failed at:\n%s" % test_params_str)
            raise e

        # ensure no output shape was completely reduced
        for pair in Scx:
            for i, c in enumerate(Scx[pair]):
                assert not np.any(c['coef'].shape == 0), (pair, i)

        # meta test
        out_3D = test_params['out_3D']
        for field in ('j', 'n', 'spin', 'stride'):
          for pair in jmeta[field]:
            assert_equal_lengths(Scx, jmeta, field, pair, out_3D,
                                 test_params_str, jtfs)
            meta_idx = [0]
            for i in range(len(Scx[pair])):
                assert_equal_values(Scx, jmeta, field, pair, i, meta_idx,
                                    out_3D, test_params_str, test_params, jtfs)

        # check stride if `aligned`
        if jtfs.aligned:
            assert_aligned_stride(Scx, test_params_str, jtfs)

        # save compute and test this method for thoroughness
        if jtfs.average:
            for structure in (1, 2, 3, 4):
                for separate_lowpass in (False, True):
                    try:
                        _ = tkt.pack_coeffs_jtfs(
                            Scx, jmeta, structure=structure,
                            separate_lowpass=separate_lowpass,
                            sampling_psi_fr=jtfs.sampling_psi_fr,
                            out_3D=jtfs.out_3D)
                    except Exception as e:
                        print(test_params_str)
                        raise e
        # bundled test: info()
        _ = jtfs.info(show=False)

    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    if default_backend != 'numpy':
        # meta doesn't change
        warnings.warn("`test_meta()` skipped per non-'numpy' `default_backend`")
        return None if run_without_pytest else pytest.skip()

    # minimal global averaging testing #######################################
    N = 512
    out_type = 'dict:list'
    x = np.random.randn(N)
    params = dict(shape=N, J=9, Q=9, J_fr=5, Q_fr=1, F=2**7, out_type=out_type,
                  **pad_mode_kw)

    # ensure global averaging is attained
    def assert_fn(jtfs):
        assert jtfs.average_global_phi, (jtfs.shape, jtfs.T)
        assert jtfs.scf.average_fr_global_phi, (jtfs.N_frs_max, jtfs.F)

        if jtfs.average:
            assert jtfs.average_global, (jtfs.shape, jtfs.T)
        if jtfs.average_fr:
            assert jtfs.average_fr_global, (jtfs.N_frs_max, jtfs.F)

    # run tests
    for average_fr in (True, False):
        for average in (True, False):
            test_params = dict(average_fr=average_fr, average=average,
                               sampling_filters_fr='resample', out_3D=False)
            run_test(params, test_params, assert_fn)

    # main tests #############################################################
    skip_long = bool(CMD_SKIP_LONG or SKIP_LONG)
    if skip_long:
        warnings.warn("Skipped most of `test_meta()`")
        return
    N = 512
    x = np.random.randn(N)

    # make scattering objects
    _J = int(np.log2(N) - 1)  # have 2 time units at output
    J = (_J, _J - 1)
    Q = (16, 1)
    J_fr = 5
    Q_fr = 2
    F = 4
    params = dict(shape=N, J=J, Q=Q, J_fr=J_fr, Q_fr=Q_fr, F=F, out_type=out_type,
                  **pad_mode_kw)

    for out_3D in (False, True):
      for average_fr in (True, False):
        for average in (True, False):
          if out_3D and not (average_fr and average):
              continue  # invalid option
          for aligned in (True, False):
            for sampling_psi_fr in ('resample', 'exclude', 'recalibrate'):
              for sampling_phi_fr in ('resample', 'recalibrate'):
                  if aligned and sampling_phi_fr == 'recalibrate':
                      continue  # invalid option
                  test_params = dict(
                      out_3D=out_3D, average_fr=average_fr, average=average,
                      aligned=aligned,
                      sampling_filters_fr=(sampling_psi_fr, sampling_phi_fr))
                  run_test(params, test_params)

    # reproduce this case separately; above doesn't test where 'exclude' fails
    # also test `paths_exclude`
    N = 1024
    x = np.random.randn(N)
    _J = int(np.log2(N) - 1)
    J = (_J, _J + 1)
    J_fr = 3
    params = dict(shape=N, J=J, Q=Q, J_fr=J_fr, Q_fr=Q_fr, F=F, out_type=out_type,
                  pad_mode_fr='zero')  # 'zero' just to account for this case
    _paths_exclude = {'n2': -1, 'n1_fr': 0, 'j2': 1, 'j1_fr': -1}

    sampling_psi_fr = 'exclude'
    out_3D = False
    for average_fr in (True, False):
      for average in (True, False):
        for aligned in (True, False):
          for sampling_phi_fr in ('resample', 'recalibrate'):
            for paths_exclude in (None, _paths_exclude):
              if sampling_phi_fr == 'recalibrate' and aligned:
                  continue  # invalid option
              test_params = dict(
                  out_3D=out_3D, average_fr=average_fr, average=average,
                  aligned=aligned,
                  sampling_filters_fr=(sampling_psi_fr, sampling_phi_fr),
                  paths_exclude=paths_exclude)
              run_test(params, test_params)


def test_output():
    """Applies JTFS on a stored signal to make sure its output agrees with
    a previously calculated version. Tests for:

          (aligned, average_fr, out_3D,   F,        sampling_filters_fr)
        0. False    True        False     32        ('exclude', 'recalibrate')
        1. True     True        True      4         ('resample', 'resample')
        2. False    True        True      16        ('resample', 'resample')
        3. True     True        False     'global'  ('resample', 'resample')
        4. True     False       False     8         ('resample', 'resample')
        5. False    True        True      16        ('recalibrate', 'recalibrate')

        6. special: params such that `scf._J_pad_fr_fo > scf.J_pad_frs_max_init`
            - i.e. all first-order coeffs pad to greater than longest set of
            second-order, as in `U1 * phi_t * phi_f` and
            `(U1 * phi_t * psi_f) * phi_t * phi_f`.

    For complete info see `data['code']` (`_load_data()`).

    Note, relative errors as great as 0.1 are acceptable for TensorFlow backend.
    They're confirmed to be due to its numeric imprecision, likely with FFT,
    particularly with low values.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    num_tests = sum(p.suffix == '.npz' and p.stem.endswith('coeffs')
                    for p in Path(TEST_DATA_DIR, 'test_jtfs').iterdir())

    for test_num in range(num_tests):
        # if test_num != 4:
        #     continue
        x, out_stored, params, params_str, _ = load_data(test_num)

        jtfs = TimeFrequencyScattering1D(**params, frontend=default_backend)
        jmeta = jtfs.meta()
        out = jtfs(x)
        out = tkt.jtfs_to_numpy(out)

        # assert equal total number of coefficients
        if params['out_type'] == 'dict:list':
            n_coef_out = sum(len(o) for o in out.values())
        elif params['out_type'] == 'dict:array':
            n_coef_out = sum(o.shape[1] for o in out.values())
        n_coef_out_stored = sum(len(o) for o in out_stored.values())
        assert n_coef_out == n_coef_out_stored, (
            "out vs stored number of coeffs mismatch ({} != {})\n{}"
            ).format(n_coef_out, n_coef_out_stored, params_str)

        # edge case test
        if test_num == 6:
            assert jtfs.scf._J_pad_fr_fo > jtfs.scf.J_pad_frs_max_init, (
                jtfs.scf._J_pad_fr_fo, jtfs.scf.J_pad_frs_max_init)

        mean_aes, max_aes = [0], [0]
        already_printed_test_info, max_mean_info, max_max_info = False, None, None
        for pair in out:
            out_stored_pair = out_stored[pair]
            for i, o in enumerate(out[pair]):
                n = jmeta['n'][pair][i]
                while n.squeeze().ndim > 1:
                    n = n[0]
                # assert equal shapes
                o = o if params['out_type'] == 'dict:array' else o['coef']
                o_stored = out_stored_pair[i]
                errmsg = ("out[{0}][{1}].shape != out_stored[{0}][{1}].shape | "
                          "n={2}\n({3} != {4})\n").format(
                              pair, i, n, o.shape, o_stored.shape)
                if not already_printed_test_info:
                    errmsg += params_str

                if output_test_print_mode and o.shape != o_stored.shape:
                    # print(errmsg)
                    already_printed_test_info = True
                    continue
                else:
                    assert o.shape == o_stored.shape, errmsg

                # store info for printing
                adiff = tkt.rel_ae(o_stored, o, ref_both=True)
                mean_ae, max_ae = adiff.mean(), adiff.max()
                if mean_ae > max(mean_aes):
                    max_mean_info = "out[%s][%s] | n=%s" % (pair, i, n)
                if max_ae > max(max_aes):
                    max_max_info  = "out[%s][%s] | n=%s" % (pair, i, n)
                mean_aes.append(mean_ae)
                max_aes.append(max_ae)

                # assert equal values
                errmsg = ("out[{0}][{1}] != out_stored[{0}][{1}] | n={2}\n"
                          "(MeanRAE={3:.2e}, MaxRAE={4:.2e})\n"
                          ).format(pair, i, n,
                                   mean_aes[-1], max_aes[-1],)
                if not already_printed_test_info:
                    errmsg += params_str

                if output_test_print_mode and not np.allclose(o, o_stored):
                    # print(errmsg)
                    already_printed_test_info = True
                else:
                    assert np.allclose(o, o_stored), errmsg

        if output_test_print_mode:
            if max_mean_info is not None:
                print("{}: // max_meanRAE = {:.2e} | {}\n".format(
                    test_num, max(mean_aes), max_mean_info))
            if max_max_info is not None:
                print("{}: // max_maxRAE  = {:.2e} | {}\n".format(
                    test_num, max(max_aes), max_max_info))

### helper methods ###########################################################
def load_data(test_num):
    """Also see data['code']."""
    def is_meta(k):
        return k.startswith('meta:')
    def is_coef(k):
        return (':' in k and k.split(':')[-1].isdigit()) and not is_meta(k)
    def not_param(k):
        return k in ('code', 'x') or is_coef(k) or is_meta(k)

    base_name = f'{test_num}_'

    def _load(name, as_list=False, as_dict=False):
        data = np.load(Path(TEST_DATA_DIR, 'test_jtfs',
                            base_name + name + '.npz'),
                       allow_pickle=True)
        if as_dict:
            return {k: data[k] for k in data.files}
        return data

    _coeffs = _load('coeffs', as_dict=True)
    _out_stored = {k: v for k, v in _coeffs.items() if not k.endswith('_idxs')}
    unpack_idxs = {k: v for k, v in _coeffs.items() if k.endswith('_idxs')}
    # e.g.
    #   - i0, i1 = unpack_idxs['psi_t * psi_f_up'][5]
    #   - out_stored[5] == _out_stored['psi_t * psi_f_up'][:, i0*i1]
    out_stored = {pair: [v[:, slice(*idxs)]
                         for idxs in unpack_idxs[pair + '_idxs']]
                  for pair, v in _out_stored.items()}

    _params = _load('params', as_dict=True)
    params = {k: v for k, v in _params.items() if k not in ('x', 'code')}
    x = _params['x']

    _meta = _load('meta')
    meta = packed_meta_into_arr(_meta)

    for k in params:
        try:
            if k in ('average', 'average_fr', 'aligned'):
                params[k] = bool(params[k])
            elif k == 'sampling_filters_fr':
                params[k] = (bool(params[k]) if len(params[k]) == 1 else
                             tuple(params[k]))
            elif k == 'F':
                params[k] = (str(params[k]) if str(params[k]) == 'global' else
                             int(params[k]))
            elif k in ('out_type', 'pad_mode', 'pad_mode_fr'):
                params[k] = str(params[k])
            elif (k == 'max_noncqt_fr' and hasattr(params[k], 'dtype') and
                  params[k].dtype == np.int32):
                params[k] = int(params[k])
            else:
                if params[k].ndim == 0 and params[k].size == 1:
                    params[k] = np.atleast_1d(params[k])[0]  # strip array
                else:
                    params[k] = int(params[k])
        except Exception as e:
            raise Exception(("failed to load param {}, test_num={}\n{}".format(
                k, test_num, e)))

    params_str = "Test #%s:\n" % test_num
    for k, v in params.items():
        params_str += "{}={}\n".format(k, str(v))
    return x, out_stored, params, params_str, meta


def packed_meta_into_arr(data):
    meta_arr = {}
    for k in data.files:
        _, field, pair = k.split(':')
        if field not in meta_arr:
            meta_arr[field] = {}
        meta_arr[field][pair] = data[k]
    return meta_arr


def concat_joint(Scx, spinned_only=False):
    assert isinstance(Scx, dict), type(Scx)
    Scx = tkt.drop_batch_dim_jtfs(Scx)
    k = list(Scx)[0]
    out_type = ('list' if (isinstance(Scx[k], list) and 'coef' in Scx[k][0]) else
                'array')

    def cond(pair):
        if spinned_only:
            return bool('up' in pair or ('d' in pair and 'n' in pair))
        return bool(pair not in ('S0', 'S1'))

    if out_type == 'array':
        return np.vstack([c for pair, c in Scx.items() if cond(pair)])
    return np.vstack([c['coef'] for pair, coeffs in Scx.items()
                      for c in coeffs if cond(pair)])


def assert_pad_difference(jtfs, test_params_str):
    assert not all(
        J_pad_fr == jtfs.J_pad_frs_max
        for J_pad_fr in jtfs.J_pad_frs if J_pad_fr != -1
        ), "\n{}\nJ_pad_fr={}\nN_frs={}".format(
            test_params_str, jtfs.J_pad_frs, jtfs.N_frs)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_alignment()
        test_shapes()
        test_jtfs_vs_ts()
        test_freq_tp_invar()
        test_up_vs_down()
        test_sampling_psi_fr_exclude()
        test_max_pad_factor_fr()
        test_out_exclude()
        test_global_averaging()
        test_lp_sum()
        test_pack_coeffs_jtfs()
        test_energy_conservation()
        test_est_energy_conservation()
        test_implementation()
        test_pad_mode_fr()
        test_no_second_order_filters()
        test_backends()
        test_differentiability_torch()
        test_reconstruction_torch()
        test_batch_shape_agnostic()
        test_out_type()
        test_meta()
        test_output()
    else:
        pytest.main([__file__, "-s"])
