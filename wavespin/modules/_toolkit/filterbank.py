# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Tools for filterbank introspection/manipulation."""
import os
import numpy as np
import scipy.signal
import warnings
from scipy.fft import fft, ifft

from ...scattering1d.refining import smart_paths_exclude
from ...utils.gen_utils import npy
from ...utils.measures import compute_filter_redundancy, compute_bandwidth
from .misc import energy

try:
    from tqdm import trange
except:  # no-cov
    trange = None


def fit_smart_paths(sc, x_all, e_loss_goal=.01, outs_dir=None, verbose=True):
    """Configures `paths_exclude` to guarantee energy loss doesn't exceed set
    threshold on the provided dataset.

    Iterates `x_all` and dynamically adjusts `e_th` that's passed into
    `smart_paths()`, refining estimates to match `e_loss_goal`.

    Note, ignores existing `sc.paths_exclude`. If execution is forcibly
    interrupted repeatedly, the following attributes will be overwritten:
    `paths_exclude, out_type`. By default, the program will restore
    them to original values even upon interruption via `try-finally`.

    Parameters
    ----------
    sc : Scattering1D
        Time Scattering instance.

        Use to also compute for JTFS, by instantiating with same time scattering
        params (`J`, `Q`, etc). See "JTFS example" below.

    x_all : tensor / list[tensor] / generator
        List of 1D tensors, 2D tensor shaped `(n_samples, time)`,
        or a generator (see "Generator example" below).

        For machine learning, this should *not* include samples from the test
        set; that leaks the test set. If one wants to be safe, a slightly
        smaller `e_loss_goal` can be chosen.

    e_loss_goal : float[>0, <1]
        Energy loss goal; this shall not be exceeded by `sc` over `x_all`.

    outs_dir : str / None
        Path to pre-computed full transform's outputs.
        See "Performance tip" below.

    verbose : bool (default True)
        Whether to print progress reports.

    Returns
    -------
    e_th_optimal_est : float
        Optimal `e_th` as passed to `smart_paths()`, as estimated by the
        search procedure.

        The output of such `smart_paths()` is already set to `sc.paths_exclude`,
        so this output can be discarded.

    Performance tip
    ---------------
    Simply use `outs_dir` as below; only a single pass over the dataset will take
    place. Requires `out_type='array'`.

    ::
        wavespin.toolkit._compute_e_fulls(sc, x_all, outs_dir)
        fit_smart_paths(..., outs_dir=outs_dir)

    JTFS example
    ------------
    ::
        jtfs = TimeFrequencyScattering1D(2048)
        sc = Scattering1D(**{k: getattr(jtfs, k) for k in
                             ('shape', 'J', 'Q', 'T', 'max_pad_factor')})
        fit_smart_paths(sc, x_all)

    Generator example
    -----------------
    Must supply `__getitem__` and `__len__` methods.
    Below loads numpy arrays from a directory.

    ::
        class MyGen():
            def __init__(self, directory):
                from pathlib import Path
                self.paths = [p for p in Path(directory).iterdir()
                              if p.suffix == '.npy']

            def __getitem__(self, idx):
                if idx >= len(self):  # needed if method doesn't throw IndexError
                    raise IndexError  # so here not needed per `paths[idx]`
                return np.load(self.paths[idx])

            def __len__(self):
                return len(self.paths)

        x_all = MyGen(r"C:/Desktop/my_data//")

    JTFS vs Scattering1D
    --------------------
    Scattering1D enjoys the property, `e_loss(e_th1) >= e_loss(e_th2)`, for all
    `e_th1 >= e_th2` and `x`. This means we can find the right `e_th` for a
    given `e_loss` from a single pass over the dataset:

        1. Start with high `e_th`.
        2. If new `x` violates `e_loss`, lower `e_th`. This `e_th` is guaranteed
           to not violate `e_loss` for any preceding `x` due to said property:
           the energy loss for those `x` can only go *lower* from lowering `e_th`.
        3. Repeat for all `x`. The final `e_th` is the optimal `e_th`, within
           the search increment (currently `e_th *= .99`).

    JTFS doesn't enjoy this property. Changing the number of first-order rows
    for any given `n2` has non-linear interactions with second-order coefficients
    - namely, it changes the length of convolution and potentially padding, so
    *fewer* `n1`'s may *increase* joint coefficients' energies. This is probable
    with `pad_mode_fr='zero'`, but not with `'conj-reflect-zero'`, as latter
    strives for energy conservation.

    Arguably, it's better to use Scattering1D to compute for JTFS. Ideally, JTFS
    second-order energy equals Scattering1D second-order energy, hence we can
    use one to get the other - but ideal doesn't work per above, and the
    differences are sometimes meaningful. I we decide they aren't meaningful,
    then Scattering1D provides true energy loss measures.

    Why isn't JTFS directly supported?
    ----------------------------------
    Implementation and testing complexity too great given its benefits.
    Also quite slow.

    Originally this method supported all but one thing JTFS needed: it required
    completely re-instantiating JTFS with its original configurations each time
    `paths_exclude` was updated, so one would have to pass in the configs or
    they'd need to be fetched automatically. This isn't that hard but it is quite
    slow and with everything else requires a fair bit of testing code.
    """
    # wrap in `try-finally` to restore changed parameters
    pe = sc.paths_exclude.copy()
    ot = sc.out_type
    try:
        sc.out_type = 'array'
        out = _fit_smart_paths(sc, x_all, e_loss_goal, outs_dir, verbose)
    finally:
        sc.out_type = ot
        sc.paths_exclude.update(pe)
    return out


def _fit_smart_paths(sc, x_all, e_loss_goal, outs_dir, verbose):
    # handle args ############################################################
    # sanity checks
    if hasattr(x_all, 'ndim'):
        assert x_all.ndim == 2, x_all.shape
    elif isinstance(x_all, list):
        assert all(x.ndim == 1 for x in x_all)
    assert 0 < e_loss_goal < 1, e_loss_goal

    # prepare to loop ########################################################
    # first collect full transform's energies
    if outs_dir is not None:
        e_fulls = np.load(os.path.join(outs_dir, 'e_fulls.npy'))
        if verbose:
            print("Using pre-computed `e_fulls`...")
    else:
        e_fulls = _compute_e_fulls(sc, x_all, verbose=verbose)

    # initialize paths_exclude to a high since we'll only be lowering it
    e_th_pseudo_max = .5
    e_th_init = e_th_pseudo_max
    sc.paths_exclude = smart_paths_exclude(sc.psi1_f, sc.psi2_f,
                                           e_th_direct=e_th_init)

    # main loop ##############################################################
    e_th_optimal_est = _compute_e_losses(
        sc, x_all, e_fulls, e_th_init, e_loss_goal, e_th_pseudo_max, outs_dir,
        verbose)
    return e_th_optimal_est


def _compute_e_losses(sc, x_all, e_fulls, e_th_init, e_loss_goal=-1,
                      e_th_pseudo_max=.5, outs_dir=None, verbose=1):
    # maybe print status
    if verbose:
        print("Optimizing energy threshold for e_loss_goal=%.3g..." % e_loss_goal)

    # derived params
    ckw = dict(psi1_f=sc.psi1_f, psi2_f=sc.psi2_f)
    if outs_dir is not None:
        outs_dir = os.path.abspath(outs_dir)  # for debug
        # get full meta then restore paths
        sp, sc.paths_exclude = sc.paths_exclude, {}
        ns_full = sc.meta()['n']
        sc.paths_exclude = sp

    # reusable
    def compute_e_loss(idx, e_th_current, x=None, update_paths=False):
        if update_paths:
            sp = smart_paths_exclude(**ckw, e_th_direct=e_th_current)
            sc.paths_exclude = sp

        # either get `x` and scatter, or load precomputed output and trim it
        if outs_dir is not None:
            out = np.load(os.path.join(outs_dir, f'{idx}.npy'))
            if out.ndim == 2:
                out = out[None]  # add batch dim if absent
            out = out.transpose(1, 0, 2)  # prep for iterating coefficients

            out_sp = []
            for row, n in zip(out, ns_full):
                if tuple(n) not in sc.paths_exclude['n2, n1']:
                    out_sp.append(row)
            out_sp = np.array(out_sp).transpose(1, 0, 2)

            # assert expected shapes, as we reasonably can
            A = len(out) - out_sp.shape[1]
            B = len(sc.paths_exclude['n2, n1'])
            assert A == B, (A, B, out.shape, out_sp.shape)
        else:
            if x is None:
                x = x_all[idx]
            x = x[None] if x.ndim == 1 else x  # add batch dim if absent
            out_sp = sc(x)

        # compute energy and corresponding loss
        e_sp = samples_energy(out_sp)
        e_loss = 1 - npy(e_sp) / e_fulls[idx]

        return e_loss, x

    # track thresholds and obtained losses
    e_th_current = e_th_init
    e_ths_all = [e_th_current]
    # loop params
    ranger = _get_ranger(verbose)

    e_losses = []

    # pass loop ##########################################################
    # Scattering1D: `e_th_current` is only lowered inside this loop, and upon
    # lowering, the value is checked to not violate `e_loss_goal`, so whatever
    # its final loop value is, it's guaranteed to not violate `e_loss_goal`.
    for i in ranger(len(x_all)):
        e_loss, x = compute_e_loss(i, e_th_current)

        # if loss exceeds goal, lower e_th_current
        if any(el > e_loss_goal for el in e_loss):
            # check that it doesn't violate `e_loss_goal`, and if it does,
            # adjust lightly until it no longer does so
            eloss, _ = compute_e_loss(i, e_th_current, x, update_paths=True)
            while any(el > e_loss_goal for el in eloss):
                e_th_current *= .99
                eloss, _ = compute_e_loss(i, e_th_current, x,
                                          update_paths=True)
            # track
            e_ths_all.append(e_th_current)

        e_losses.append(e_loss)
    e_th_loop_out = e_th_current

    # finalize ###############################################################
    e_th_optimal_est = e_th_loop_out
    sc.paths_exclude = smart_paths_exclude(**ckw, e_th_direct=e_th_optimal_est)
    return e_th_optimal_est


def _compute_e_fulls(sc, x_all, outs_dir=None, verbose=1):
    if outs_dir is not None:
        outs_dir = os.path.abspath(outs_dir)
    if verbose:
        if outs_dir is None:
            print("Gathering full transform's energies...")
        else:
            print("Gathering full transform's energies, and saving in\n"
                  + outs_dir)
    ranger = _get_ranger(verbose)

    e_fulls = []
    sp, sc.paths_exclude = sc.paths_exclude, {}
    for idx in ranger(len(x_all)):
        x = x_all[idx]
        x = x[None] if x.ndim == 1 else x
        out = sc(x)
        e_out = samples_energy(out)
        e_fulls.append(e_out)
        if outs_dir is not None:
            np.save(os.path.join(outs_dir, f'{idx}.npy'), npy(out))
    sc.paths_exclude = sp

    if verbose:
        print("... done\n")

    e_fulls = npy(e_fulls)
    if e_fulls.ndim == 1:
        e_fulls = e_fulls[:, None]
    if outs_dir is not None:
        np.save(os.path.join(outs_dir, 'e_fulls.npy'), e_fulls)
    return e_fulls


def samples_energy(x):
    """(batch_size, *spatial) -> (batch_size,)"""
    return energy(x, axis=tuple(range(1, x.ndim)))

#### Validating 1D filterbank ################################################
def validate_filterbank_tm(sc=None, psi1_f=None, psi2_f=None, phi_f=None,
                           criterion_amplitude=1e-3, verbose=True):
    """Runs `validate_filterbank()` on temporal filters; supports `Scattering1D`
    and `TimeFrequencyScattering1D`.

    Parameters
    ----------
    sc : `Scattering1D` / `TimeFrequencyScattering1D` / None
        If None, then `psi1_f_fr_up`, `psi1_f_fr_dn`, and `phi_f_fr` must
        be not None.

    psi1_f : list[tensor] / None
        First-order bandpasses in frequency domain.
        Overridden if `sc` is not None.

    psi2_f : list[tensor] / None
        Second-order bandpasses in frequency domain.
        Overridden if `sc` is not None.

    phi_f : tensor / None
        Lowpass filter in frequency domain.
        Overridden if `sc` is not None.

    criterion_amplitude : float
        Used for various thresholding in `validate_filterbank()`.

    verbose : bool (default True)
        Whether to print the report.

    Returns
    -------
    data1, data2 : dict, dict
        Returns from `validate_filterbank()` for `psi1_f` and `psi2_f`.
    """
    if sc is None:  # no-cov
        assert not any(arg is None for arg in (psi1_f, psi2_f, phi_f))
    else:
        psi1_f, psi2_f, phi_f = [getattr(sc, k) for k in
                                 ('psi1_f', 'psi2_f', 'phi_f')]
    psi1_f, psi2_f = [[p[0] for p in ps] for ps in (psi1_f, psi2_f)]
    phi_f = phi_f[0][0] if isinstance(phi_f[0], list) else phi_f[0]

    if verbose:  # no-cov
        print("\n// FIRST-ORDER")
    data1 = validate_filterbank(psi1_f, phi_f, criterion_amplitude,
                                verbose=verbose,
                                for_real_inputs=True, unimodal=True)
    if verbose:  # no-cov
        print("\n\n// SECOND-ORDER")
    data2 = validate_filterbank(psi2_f, phi_f, criterion_amplitude,
                                verbose=verbose,
                                for_real_inputs=True, unimodal=True)
    return data1, data2


def validate_filterbank_fr(sc=None, psi1_f_fr_up=None, psi1_f_fr_dn=None,
                           phi_f_fr=None, psi_id=0, criterion_amplitude=1e-3,
                           verbose=True):
    """Runs `validate_filterbank()` on frequential filters of JTFS.

    Parameters
    ----------
    sc : `TimeFrequencyScattering1D` / None
        JTFS instance. If None, then `psi1_f_fr_up`, `psi1_f_fr_dn`, and
        `phi_f_fr` must be not None.

    psi1_f_fr_up : list[tensor] / None
        Spin up bandpasses in frequency domain.
        Overridden if `sc` is not None.

    psi1_f_fr_dn : list[tensor] / None
        Spin down bandpasses in frequency domain.
        Overridden if `sc` is not None.

    phi_f_fr : tensor / None
        Lowpass filter in frequency domain.
        Overridden if `sc` is not None.

    psi_id : int
        See `psi_id` in `filter_bank_jtfs.psi_fr_factory`.

    criterion_amplitude : float
        Used for various thresholding in `validate_filterbank()`.

    verbose : bool (default True)
        Whether to print the report.

    Returns
    -------
    data_up, data_dn : dict, dict
        Returns from `validate_filterbank()` for `psi1_f_fr_up` and
        `psi1_f_fr_dn`.
    """
    if sc is None:  # no-cov
        assert not any(arg is None for arg in
                       (psi1_f_fr_up, psi1_f_fr_dn, phi_f_fr))
    else:
        psi1_f_fr_up, psi1_f_fr_dn, phi_f_fr = [
            getattr(sc, k) for k in
            ('psi1_f_fr_up', 'psi1_f_fr_dn', 'phi_f_fr')]

    psi1_f_fr_up, psi1_f_fr_dn = psi1_f_fr_up[psi_id], psi1_f_fr_dn[psi_id]
    phi_f_fr = phi_f_fr[0][0][0]

    if verbose:  # no-cov
        print("\n// SPIN UP")
    data_up = validate_filterbank(psi1_f_fr_up, phi_f_fr, criterion_amplitude,
                                  verbose=verbose,
                                  for_real_inputs=False, unimodal=True)
    if verbose:  # no-cov
        print("\n\n// SPIN DOWN")
    data_dn = validate_filterbank(psi1_f_fr_dn, phi_f_fr, criterion_amplitude,
                                  verbose=verbose,
                                  for_real_inputs=False, unimodal=True)
    return data_up, data_dn


def validate_filterbank(psi_fs, phi_f=None, criterion_amplitude=1e-3,
                        for_real_inputs=True, unimodal=True, is_time_domain=False,
                        verbose=True):
    """Checks whether the wavelet filterbank is well-behaved against several
    criterion:

        1. Analyticity:

          - A: Whether analytic *and* anti-analytic filters are present
               (input should contain only one)
          - B: Extent of (anti-)analyticity - whether there's components
               on other side of Nyquist
          - C: Whether the Nyquist bin is halved

        2. Aliasing:

          - A. Whether peaks are sorted (left to right or right to left).
               If not, it's possible aliasing (or sloppy user input).
          - B. Whether peaks are distributed exponentially or linearly.
               If neither, it's possible aliasing. (Detection isn't foulproof.)

        3. Zero-mean: whether filters are zero-mean (in time domain)

        4. Zero-phase: whether filters are zero-phase

        5. Frequency coverage: whether filters capture every frequency,
           and whether they do so excessively or insufficiently.

             - Measured with Littlewood-Paley sum (sum of energies),
               the "energy transfer function".
             - Also measured with sum of LP sum, in case of imperfect
               analyticity not being accounted for (must fold leaked frequencies,
               see `help(toolkit.compute_lp_sum)`, `fold_antianalytic`).

        6. Frequency-bandwidth tiling: whether upper quarters of frequencies
           follow CQT (fixed `xi/sigma = (center freq) / bandwidth`), and
           whether all wavelet peak frequencies are distributed either
           exponentially or linearly.

           Only upper quarters (i.e. not `0 to N//4`) is checked for CQT because
           the non-CQT portion could be in the majority, but unlikely for it to
           ever span the upper half.

        7. Redundancy: whether filters overlap excessively (this isn't
           necessarily bad).

             - Measured as ratio of product of energies to sum of energies
               of adjacent filters
             - Also measured as peak duplication in frequency domain. Note,
               it's possible to exceed redundancy thershold without duplicating
               peaks, and vice versa (but latter is more difficult).

        8. Decay:

          - A: Whether any filter is a pure sine (occurs if we try to sample
               a wavelet at too low of a center frequency)
          - B: Whether filters decay sufficiently in time domain to avoid
               boundary effects
          - C: Whether filters decay sufficiently in frequency domain
               (bandwidth isn't the entire signal), and whether they decay
               permanently (don't rise again after decaying)

          B may fail for same reason as 8A & 8B (see these).

        9. Temporal peaks:

          - A: Whether peak is at t==0
          - B: Whether there is only one peak
          - C: Whether decay is smooth (else will incur inflection points)

          A and B may fail to hold for lowest xi due to Morlet's corrective
          term; this is proper behavior.
          See https://www.desmos.com/calculator/ivd7t3mjn8

    Parameters
    ----------
    psi_fs : list[tensor]
        Wavelet filterbank, by default in frequency domain (if in time domain,
        set `in_time_domain=True`.
        Analytic or pseudo-analytic, or anti- of either; does not support
        real-valued wavelets (in time domain).

        If `psi_fs` aren't all same length, will pad in time domain and
        center about `n=0` (DFT-symmetrically), with original length's center
        placed at index 0.

        Note, if `psi_fs` are provided in time domain or aren't all same length,
        they're padded such that FFT convolution matches
        `np.convolve(, mode='full')`. If wavelets are properly centered for FFT
        convolution - that is, either at `n=0` or within `ifftshift` or `n=0`,
        then for even lengths, `np.convolve` *will not* produce correct
        results - which is what happens with `scipy.cwt`.

    phi_f : tensor
        Lowpass filter in frequency domain, of same length as `psi_fs`.

    criterion_amplitude : float
        Used for various thresholding.

    for_real_inputs : bool (default True)
        Whether the filterbank is intended for real-only inputs.
        E.g. `False` for spinned bandpasses in JTFS.

    unimodal : bool (default True)
        Whether the wavelets have a single peak in frequency domain.
        If `False`, some checks are omitted, and others might be inaccurate.
        Always `True` for Morlet wavelets.

    in_time_domain : bool (default False)
        Whether `psi_fs` are in time domain. See notes in `psi_fs`.

    verbose : bool (default True)
        Whether to print the report.

    Returns
    -------
    data : dict
        Aggregated testing info, along with the report. For keys, see
        `print(list(data))`. Note, for entries that describe individual filters,
        the indexing corresponds to `psi_fs` sorted in order of decreasing
        peak frequency.
    """
    def pop_if_no_header(report, did_atleast_one_header):
        """`did_atleast_one_header` sets to `False` after every `title()` call,
        whereas `did_header` before every subsection, i.e. a possible
        `if not did_header: report += []`. Former is to pop titles, latter
        is to avoid repeatedly appending subsection text.
        """
        if not did_atleast_one_header:
            report.pop(-1)

    # handle `psi_fs` domain and length ######################################
    # squeeze all for convenience
    psi_fs = [p.squeeze() for p in psi_fs]
    # fetch max length
    max_len = max(len(p) for p in psi_fs)

    # take to freq or pad to max length
    _psi_fs = []  # store processed filters
    # also handle lowpass
    if phi_f is not None:
        psi_fs.append(phi_f)

    for p in psi_fs:
        if len(p) != max_len:
            if not is_time_domain:
                p = ifft(p)
            # right-pad
            orig_len = len(p)
            p = np.pad(p, [0, max_len - orig_len])
            # odd case: circularly-center about n=0; equivalent to `ifftshift`
            # even case: center such that first output index of FFT convolution
            # corresponds to `sum(x, p[::-1][-len(p)//2:])`, where `p` is in
            # time domain. This is what `np.convolve` does, and it's *not*
            # equivalent to FFT convolution after `ifftshift`
            center_idx = orig_len // 2
            p = np.roll(p, -(center_idx - 1))
            # take to freq-domain
            p = fft(p)
        elif is_time_domain:
            center_idx = len(p) // 2
            p = np.roll(p, -(center_idx - 1))
            p = fft(p)
        _psi_fs.append(p)
    psi_fs = _psi_fs
    # recover & detach phi_f
    if phi_f is not None:
        phi_f = psi_fs.pop(-1)

    ##########################################################################

    # set reference filter
    psi_f_0 = psi_fs[0]
    # fetch basic metadata
    N = len(psi_f_0)

    # assert all inputs are same length
    # note, above already guarantees this, but we keep the code logic in case
    # something changes in the future
    for n, p in enumerate(psi_fs):
        assert len(p) == N, (len(p), N)
    if phi_f is not None:
        assert len(phi_f) == N, (len(phi_f), N)

    # initialize report
    report = []
    data = {k: {} for k in ('analytic_a_ratio', 'nonzero_mean', 'sine', 'decay',
                            'imag_mean', 'time_peak_idx', 'n_inflections',
                            'redundancy', 'peak_duplicates')}
    data['opposite_analytic'] = []

    def title(txt):
        return ("\n== {} " + "=" * (80 - len(txt)) + "\n").format(txt)
    # for later
    w_pos = np.linspace(0, N//2, N//2 + 1, endpoint=True).astype(int)
    w_neg = - w_pos[1:-1][::-1]
    w = np.hstack([w_pos, w_neg])
    eps = np.finfo(psi_f_0.dtype).eps

    peak_idxs = np.array([np.argmax(np.abs(p)) for p in psi_fs])
    peak_idxs_sorted = np.sort(peak_idxs)
    if unimodal and not (np.all(peak_idxs == peak_idxs_sorted) or
                         np.all(peak_idxs == peak_idxs_sorted[::-1])):
        warnings.warn("`psi_fs` peak locations are not sorted; a possible reason "
                      "is aliasing. Will sort, breaking mapping with input's.")
        data['not_sorted'] = True
        peak_idxs = peak_idxs_sorted

    # Analyticity ############################################################
    # check if there are both analytic and anti-analytic bandpasses ##########
    report += [title("ANALYTICITY")]
    did_header = did_atleast_one_header = False

    peak_idx_0 = np.argmax(psi_f_0)
    if peak_idx_0 == N // 2:  # ambiguous case; check next filter
        peak_idx_0 = np.argmax(psi_fs[1])
    analytic_0 = bool(peak_idx_0 < N//2)
    # assume entire filterbank is per psi_0
    analyticity = "analytic" if analytic_0 else "anti-analytic"

    # check whether all is analytic or anti-analytic
    found_counteranalytic = False
    for n, p in enumerate(psi_fs[1:]):
        peak_idx_n = np.argmax(np.abs(p))
        analytic_n = bool(peak_idx_n < N//2)
        if not (analytic_0 is analytic_n):
            if not did_header:
                report += [("Found analytic AND anti-analytic filters in same "
                            "filterbank! psi_fs[0] is {}, but the following "
                            "aren't:\n").format(analyticity)]
                did_header = did_atleast_one_header = True
            report += [f"psi_fs[{n}]\n"]
            data['opposite_analytic'].append(n)
            found_counteranalytic = True

    # set `is_analytic` based on which there are more of
    if not found_counteranalytic:
        is_analytic = analytic_0
    else:
        n_analytic     = sum(np.argmax(np.abs(p)) <= N//2 for p in psi_fs)
        n_antianalytic = sum(np.argmax(np.abs(p)) >= N//2 for p in psi_fs)
        if n_analytic > n_antianalytic or n_analytic == n_antianalytic:
            is_analytic = True
        else:
            is_analytic = False
        report += [("\nIn total, there are {} analytic and {} anti-analytic "
                    "filters\n").format(n_analytic, n_antianalytic)]

    # determine whether the filterbank is strictly analytic/anti-analytic
    if is_analytic:
        negatives_all_zero = False
        for p in psi_fs:
            # exclude Nyquist as it's both in analytic and anti-analytic
            if not np.allclose(p[len(p)//2 + 1:], 0.):
                break
        else:
            negatives_all_zero = True
        strict_analyticity = negatives_all_zero
    else:
        positives_all_zero = False
        for p in psi_fs:
            # exclude DC, one problem at a time; exclude Nyquist
            if not np.allclose(p[1:len(p)//2], 0.):
                break
        else:
            positives_all_zero = True
        strict_analyticity = positives_all_zero

    # determine whether the Nyquist bin is halved
    if strict_analyticity:
        did_header = False
        pf = psi_fs[0]
        if is_analytic:
            nyquist_halved = bool(pf[N//2 - 1] / pf[N//2] > 2)
        else:
            nyquist_halved = bool(pf[N//2 + 1] / pf[N//2] > 2)
        if not nyquist_halved:
            report += [("Nyquist bin isn't halved for strictly analytic wavelet; "
                        "yields improper analyticity with bad time decay.\n")]
            did_header = did_atleast_one_header = True

    # check if any bandpass isn't strictly analytic/anti- ####################
    did_header = False
    th_ratio = (1 / criterion_amplitude)
    for n, p in enumerate(psi_fs):
        ap = np.abs(p)
        # assume entire filterbank is per psi_0
        if is_analytic:
            # Nyquist is *at* N//2, so to include in sum, index up to N//2 + 1
            a_ratio = (ap[:N//2 + 1].sum() / (ap[N//2 + 1:].sum() + eps))
        else:
            a_ratio = (ap[N//2:].sum() / (ap[:N//2].sum() + eps))
        if a_ratio < th_ratio:
            if not did_header:
                report += [("\nFound not strictly {} filter(s); threshold for "
                            "ratio of `spectral sum` to `spectral sum past "
                            "Nyquist` is {} - got (less is worse):\n"
                            ).format(analyticity, th_ratio)]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}]: {:.1f}\n".format(n, a_ratio)]
            data['analytic_a_ratio'][n] = a_ratio

    # check if any bandpass isn't zero-mean ##################################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("ZERO-MEAN")]
    did_header = did_atleast_one_header = False

    for n, p in enumerate(psi_fs):
        if p[0] != 0:
            if not did_header:
                report += ["Found non-zero mean filter(s)!:\n"]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}][0] == {:.2e}\n".format(n, p[0])]
            data['nonzero_mean'][n] = p[0]

    # Littlewood-Paley sum ###################################################
    def report_lp_sum(report, phi):
        with_phi = not isinstance(phi, int)
        s = "with" if with_phi else "without"
        report += [title("LP-SUM (%s phi)" % s)]
        did_header = did_atleast_one_header = False

        # compute parameters #################################################
        # finish computing lp sum
        lp_sum = lp_sum_psi + np.abs(phi)**2
        lp_sum = (lp_sum[:N//2 + 1] if is_analytic else
                  lp_sum[N//2:])
        if with_phi:
            data['lp'] = lp_sum
        else:
            data['lp_no_phi'] = lp_sum
        if not with_phi and is_analytic:
            lp_sum = lp_sum[1:]  # exclude dc

        # excess / underflow
        diff_over  = lp_sum - th_lp_sum_over
        diff_under = th_lp_sum_under - lp_sum
        diff_over_max, diff_under_max = diff_over.max(), diff_under.max()
        excess_over  = np.where(diff_over  > th_sum_excess)[0]
        excess_under = np.where(diff_under > th_sum_excess)[0]
        if not is_analytic:
            excess_over  += N//2
            excess_under += N//2
        elif is_analytic and not with_phi:
            excess_over += 1
            excess_under += 1  # dc

        # lp sum sum
        lp_sum_sum = lp_sum.sum()
        # `1` per bin, minus
        #   - DC bin, since no phi
        #   - half of Nyquist bin, since `analytic=True` cannot ever get a full
        #     Nyquist (Nyquist bin is halved, so even in best case of the peak
        #     placed at Nyquist, we get 0.5). Unclear if any correction is due
        #     on this.
        # negligible adjustments if `N` is large (JTFS N_frs can be small enough)
        expected_sum = N
        if not with_phi:
            expected_sum -= 1
        if strict_analyticity:
            expected_sum -= .5

        # scale according to tolerance.
        # tolerances determined empirically from the most conservative case;
        # see `tests.test_jtfs.test_lp_sum`
        th_sum_above = .01
        th_sum_below = .15
        expected_above = expected_sum * (1 + th_sum_above)
        expected_below = expected_sum * (1 - th_sum_below)

        # append report entries ##############################################
        input_kind = "real" if for_real_inputs else "complex"
        if len(excess_over) > 0:
            # show at most 30 values
            stride = max(int(round(len(excess_over) / 30)), 1)
            s = f", shown skipping every {stride-1} values" if stride != 1 else ""
            report += [("LP sum exceeds threshold of {} (for {} inputs) by "
                        "at most {:.3f} (more is worse) at following frequency "
                        "bin indices (0 to {}{}):\n"
                        ).format(th_lp_sum_over, input_kind, diff_over_max,
                                 N//2, s)]
            report += ["{}\n\n".format(w[excess_over][::stride])]
            did_header = did_atleast_one_header = True
            if with_phi:
                data['lp_excess_over'] = excess_over
                data['lp_excess_over_max'] = diff_over_max
            else:
                data['lp_no_phi_excess_over'] = excess_over
                data['lp_no_phi_excess_over_max'] = diff_over_max

        if len(excess_under) > 0:
            # show at most 30 values
            stride = max(int(round(len(excess_under) / 30)), 1)
            s = f", shown skipping every {stride-1} values" if stride != 1 else ""
            report += [("LP sum falls below threshold of {} (for {} inputs) by "
                        "at most {:.3f} (more is worse; ~{} implies ~zero "
                        "capturing of the frequency!) at following frequency "
                        "bin indices (0 to {}{}):\n"
                        ).format(th_lp_sum_under, input_kind, diff_under_max,
                                 th_lp_sum_under, N//2, s)]
            # w_show = np.round(w[excess_under][::stride], 3)
            report += ["{}\n\n".format(w[excess_under][::stride])]
            did_header = did_atleast_one_header = True
            if with_phi:
                data['lp_excess_under'] = excess_under
                data['lp_excess_under_max'] = diff_under_max
            else:
                data['lp_no_phi_excess_under'] = excess_under
                data['lp_no_phi_excess_under_max'] = diff_under_max

        if lp_sum_sum > expected_above:
            report += [("LP sum sum exceeds expected: {} > {}. If LP sum "
                        "otherwise has no excess, then there may be leakage due "
                        "to imperfect analyticity, corrected by folding; see "
                        "help(toolkit.fold_lp_sum)\n").format(lp_sum_sum,
                                                              expected_above)]
            did_header = did_atleast_one_header = True
            diff = lp_sum_sum - expected_above
            if with_phi:
                data['lp_sum_sum_excess_over'] = diff
            else:
                data['lp_sum_sum_no_phi_excess_over'] = diff

        if lp_sum_sum < expected_below:
            report += [("LP sum sum falls short of expected: {} < {}. If LP sum "
                        "otherwise doesn't fall short, then there may be leakage "
                        "due to imperfect analyticity, corrected by folding; see "
                        "help(toolkit.fold_lp_sum)\n").format(lp_sum_sum,
                                                              expected_below)]
            did_header = did_atleast_one_header = True
            diff = expected_below - lp_sum_sum
            if with_phi:
                data['lp_sum_sum_excess_under'] = diff
            else:
                data['lp_sum_sum_no_phi_excess_under'] = diff

        if did_header:
            stdev = np.abs(lp_sum[lp_sum >= th_lp_sum_under] -
                           th_lp_sum_under).std()
            report += [("Mean absolute deviation from tight frame: {:.2f}\n"
                        "Standard deviation from tight frame: {:.2f} "
                        "(excluded LP sum values below {})\n").format(
                            np.abs(diff_over).mean(), stdev, th_lp_sum_under)]

        pop_if_no_header(report, did_atleast_one_header)

    pop_if_no_header(report, did_atleast_one_header)
    th_lp_sum_over = 2 if for_real_inputs else 1
    th_lp_sum_under = th_lp_sum_over / 2
    th_sum_excess = (1 + criterion_amplitude)**2 - 1
    lp_sum_psi = np.sum([np.abs(p)**2 for p in psi_fs], axis=0)
    # fold opposite frequencies to ensure leaks are accounted for
    lp_sum_psi = fold_lp_sum(lp_sum_psi, analytic_part=is_analytic)

    # do both cases
    if phi_f is not None:
        report_lp_sum(report, phi=phi_f)
    report_lp_sum(report, phi=0)

    # Redundancy #############################################################
    report += [title("REDUNDANCY")]
    did_header = did_atleast_one_header = False
    max_to_print = 20

    # overlap ####
    th_r = .4 if for_real_inputs else .2

    printed = 0
    for n in range(len(psi_fs) - 1):
        r = compute_filter_redundancy(psi_fs[n], psi_fs[n + 1])
        data['redundancy'][(n, n + 1)] = r
        if r > th_r:
            if not did_header:
                report += [("Found filters with redundancy exceeding {} (energy "
                            "overlap relative to sum of individual energies) "
                            "-- This isn't necessarily bad. Showing up to {} "
                            "filters:\n").format(th_r, max_to_print)]
                did_header = did_atleast_one_header = True
            if printed < max_to_print:
                report += ["psi_fs[{}] & psi_fs[{}]: {:.3f}\n".format(
                    n, n + 1, r)]
                printed += 1

    # peak duplication ####
    did_header = False

    printed = 0
    for n, peak_idx in enumerate(peak_idxs):
        if np.sum(peak_idx == peak_idxs) > 1:
            data['peak_duplicates'][n] = peak_idx
            if not did_header:
                spc = "\n" if did_atleast_one_header else ""
                report += [("{}Found filters with duplicate peak frequencies! "
                            "Showing up to {} filters:\n").format(spc,
                                                                  max_to_print)]
                did_header = did_atleast_one_header = True
            if printed < max_to_print:
                report += ["psi_fs[{}], peak_idx={}\n".format(n, peak_idx)]
                printed += 1

    # Decay: check if any bandpass is a pure sine ############################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("DECAY (check for pure sines)")]
    did_header = did_atleast_one_header = False
    th_ratio_max_to_next_max = (1 / criterion_amplitude)

    for n, p in enumerate(psi_fs):
        psort = np.sort(np.abs(p))  # least to greatest
        ratio = psort[-1] / (psort[-2] + eps)
        if ratio > th_ratio_max_to_next_max:
            if not did_header:
                report += [("Found filter(s) that are pure sines! Threshold for "
                            "ratio of Fourier peak to next-highest value is {} "
                            "- got (more is worse):\n"
                            ).format(th_ratio_max_to_next_max)]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}]: {:.2e}\n".format(n, ratio)]
            data['sine'][n] = ratio

    # Decay: frequency #######################################################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("DECAY (frequency)")]
    did_header = did_atleast_one_header = False

    # compute bandwidths
    bandwidths = [compute_bandwidth(pf, criterion_amplitude)
                  for pf in psi_fs]

    excess_bw = N//2 if strict_analyticity else N
    for n, bw in enumerate(bandwidths):
        if bw == excess_bw:
            if not did_header:
                report += [("Found filter(s) that never sufficiently decay "
                            "in frequency:\n")]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}], bandwidth={}\n".format(n, bw)]

    # handle case where a filter first decays and then rises again
    if unimodal:
        def decayed_then_rose(epf):
            criterion_energy = criterion_amplitude**2
            decay_idxs = np.where(epf < criterion_energy)[0]
            if len(decay_idxs) == 0:
                # never decayed
                return False

            first_decay_idx = decay_idxs[0]
            bound = len(epf)//2  # exclude opposite half
            rise_idxs = np.where(epf[first_decay_idx + 1:bound + 1] >
                                 criterion_energy)[0]
            return bool(len(rise_idxs) > 0)

        did_header = False
        for n, pf in enumerate(psi_fs):
            # center about n=0 to handle left & right separately
            pf = np.roll(pf, -np.argmax(np.abs(pf)))
            epf = np.abs(pf)**2

            dtr_right = decayed_then_rose(epf)
            # frequency-reverse
            epf[1:] = epf[1:][::-1]
            dtr_left = decayed_then_rose(epf)

            # both apply regardless of `strict_analyticity`
            # (since one of them should be impossible if it's `True`)
            if dtr_left or dtr_right:
                if not did_header:
                    report += [("Found filter(s) that decay then rise again in "
                                "frequency:\n")]
                    did_header = did_atleast_one_header = True
                report += ["psi_fs[{}]\n".format(n)]

    # Decay: boundary effects ################################################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("DECAY (boundary effects)")]
    did_header = did_atleast_one_header = False
    th_ratio_max_to_min = (1 / criterion_amplitude)

    psis = [np.fft.ifft(p) for p in psi_fs]
    apsis = [np.abs(p) for p in psis]
    for n, ap in enumerate(apsis):
        ratio = ap.max() / (ap.min() + eps)
        if ratio < th_ratio_max_to_min:
            if not did_header:
                report += [("Found filter(s) with incomplete decay (will incur "
                            "boundary effects), with following ratios of "
                            "amplitude max to edge (less is worse; threshold "
                            "is {}):\n").format(1 / criterion_amplitude)]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}]: {:.1f}\n".format(n, ratio)]
            data['decay'][n] = ratio

    # check lowpass
    if phi_f is not None:
        aphi = np.abs(np.fft.ifft(phi_f))
        ratio = aphi.max() / (aphi.min() + eps)
        if ratio < th_ratio_max_to_min:
            nl = "\n" if did_header else ""
            report += [("{}Lowpass filter has incomplete decay (will incur "
                        "boundary effects), with following ratio of amplitude "
                        "max to edge: {:.1f} > {}\n").format(nl, ratio,
                                                             th_ratio_max_to_min)]
            did_header = did_atleast_one_header = True
            data['decay'][-1] = ratio

    # Phase ##################################################################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("PHASE")]
    did_header = did_atleast_one_header = False
    th_imag_mean = eps

    for n, p in enumerate(psi_fs):
        imag_mean = np.abs(p.imag).mean()
        if imag_mean > th_imag_mean:
            if not did_header:
                report += [("Found filters with non-zero phase, with following "
                            "absolute mean imaginary values:\n")]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}]: {:.1e}\n".format(n, imag_mean)]
            data['imag_mean'][n] = imag_mean

    # Aliasing ###############################################################
    def diff_extend(diff, th, cond='gt', order=1):
        # the idea is to take `diff` without losing samples, if the goal is
        # `where(diff == 0)`; `diff` is forward difference, and two samples
        # participated in producing the zero, where later one's index is dropped
        # E.g. detecting duplicate peak indices:
        # [0, 1, 3, 3, 5] -> diff gives [2], so take [2, 3]
        # but instead of adding an index, replace next sample with zero such that
        # its `where == 0` produces that index
        if order > 1:
            diff_e = diff_extend(diff, th)
            for o in range(order - 1):
                diff_e = diff_e(diff_e, th)
            return diff_e

        diff_e = []
        d_extend = 2*th if cond == 'gt' else th
        prev_true = False
        for d in diff:
            if prev_true:
                diff_e.append(d_extend)
                prev_true = False
            else:
                diff_e.append(d)
            if (cond == 'gt' and np.abs(d) > th or
                cond == 'eq' and np.abs(d) == th):
                prev_true = True
        if prev_true:
            # last sample was zero; extend
            diff_e.append(d_extend)
        return np.array(diff_e)

    if unimodal:
        pop_if_no_header(report, did_atleast_one_header)
        report += [title("ALIASING")]
        did_header = did_atleast_one_header = False
        eps_big = eps * 100  # ease threshold for "zero"

        if len(peak_idxs) < 6:
            warnings.warn("Alias detector requires at least 6 wavelets to "
                          "work properly, per repeated `np.diff`")

        # check whether peak locations follow a linear or exponential
        # distribution, progressively dropping those that do to see if any remain

        # x[n] = A^n + C; x[n] - x[n - 1] = A^n - A^(n-1) = A^n*(1 - A) = A^n*C
        # log(A^n*C) = K + n; diff(diff(K + n)) == 0
        # `abs` for anti-analytic case with descending idxs
        logdiffs = np.diff(np.log(np.abs(np.diff(peak_idxs))), 2)
        # In general it's impossible to determine whether a rounded sequence
        # samples an exponential, since if the exponential rate (A in A^n) is
        # sufficiently small, its rounded values will be linear over some portion.
        # However, it cannot be anything else, and we are okay with linear
        # (unless constant, i.e. duplicate, captured elsewhere) - thus the net
        # case of `exp + lin` is still captured. The only uncertainty is in
        # the threshold; assuming deviation by at most 1 sample, we set it to 1.
        # A test is:
        # `for b in linspace(1.2, 6.5, 500): x = round(b**arange(10) + 50)`
        # with `if any(abs(diff, o).min() == 0 for o in (1, 2, 3)): continue`,
        # Another with: `linspace(.2, 1, 500)` and `round(256*b**arange(10) + 50)`
        # to exclude `x` with repeated or linear values
        # However, while this has no false positives (never misses an exp/lin),
        # it can also count some non-exp/lin as exp/lin, but this is rare.
        # To be safe, per above test, we use the empirical value of 0.9
        logdiffs_extended = diff_extend(logdiffs, .9)
        if len(logdiffs_extended) > len(logdiffs) + 2:
            # this could be `assert` but not worth erroring over this
            warnings.warn("`len(logdiffs_extended) > len(logdiffs) + 2`; will "
                          "use more conservative estimate on peaks distribution")
            logdiffs_extended = logdiffs
        keep = np.where(np.abs(logdiffs_extended) > .9)
        # note due to three `diff`s we artificially exclude 3 samples
        peak_idxs_remainder = peak_idxs[keep]

        # now constant (diff_order==1) and linear (diff_order==2)
        for diff_order in (1, 2):
            idxs_diff2 = np.diff(peak_idxs_remainder, diff_order)
            keep = np.where(np.abs(idxs_diff2) > eps_big)
            peak_idxs_remainder = peak_idxs_remainder[keep]

        # if anything remains, it's neither
        if len(peak_idxs_remainder) > 0:
            report += [("Found Fourier peaks that are spaced neither "
                        "exponentially nor linearly, suggesting possible "
                        "aliasing.\npsi_fs[n], n={}\n"
                        ).format(peak_idxs_remainder)]
            data['alias_peak_idxs'] = peak_idxs_remainder
            did_header = did_atleast_one_header = True

    # Frequency-bandwidth tiling; CQT ########################################
    # note, we checked for linear/exponential spacing in "Aliasing" section
    if unimodal:
        pop_if_no_header(report, did_atleast_one_header)
        report += [title("FREQUENCY-BANDWIDTH TILING")]
        did_header = did_atleast_one_header = False

        def isnt_lower_quarter(pidx):
            return ((is_analytic and pidx > N//8) or
                    (not is_analytic and pidx < (N - N//8)))

        got_peaks_above_first_quarter = any(isnt_lower_quarter(peak_idx)
                                            for peak_idx in peak_idxs)
        if got_peaks_above_first_quarter:
            # idxs must reflect distance from DC
            if is_analytic:
                peak_idxs_dist = peak_idxs
            else:
                peak_idxs_dist = [N - peak_idx for peak_idx in peak_idxs]

            # compute bandwidths, accounting for strict analyticity;
            # can infer full intended bandwidth from just one half
            if strict_analyticity:
                if is_analytic:
                    # right is trimmed
                    bandwidths = [compute_bandwidth(pf, criterion_amplitude,
                                                    return_sided=True)[0]
                                  for pf in psi_fs]
                else:
                    # left is trimmed
                    bandwidths = [compute_bandwidth(pf, criterion_amplitude,
                                                    return_sided=True)[1]
                                  for pf in psi_fs]
            else:
                bandwidths = [compute_bandwidth(pf, criterion_amplitude)
                              for pf in psi_fs]

            Qs_upper_quarters = {n: peak_idx_dist / bw
                                 for n, (peak_idx_dist, bw)
                                 in enumerate(zip(peak_idxs_dist, bandwidths))
                                 # must still use original peak idxs here
                                 if isnt_lower_quarter(peak_idxs[n])}

            Qs_values = list(Qs_upper_quarters.values())
            tolerance = .01  # abs relative difference tolerance 1%
            # pick most favorable reference
            Qs_diffs = np.abs(np.diff(Qs_values))
            Q_ref = Qs_values[np.argmin(Qs_diffs) + 1]

            non_cqts = []
            for n, Q in Qs_upper_quarters.items():
                if abs(Q - Q_ref) / Q_ref > tolerance:
                    non_cqts.append((n, Q))

            if len(non_cqts) > 0:
                non_cqt_strs = ["psi_fs[{}], Q={}".format(n, Q)
                                for n, Q in zip(*zip(*non_cqts))]
                report += [("Found non-CQT wavelets in upper quarters of "
                            "frequencies - i.e., `(center freq) / bandwidth` "
                            "isn't constant: \n{}\n"
                            ).format("\n".join(non_cqt_strs))]
                data['non_cqts'] = non_cqts
                did_header = did_atleast_one_header = True

    # Temporal peak ##########################################################
    if unimodal:
        # check that temporal peak is at t==0 ################################
        pop_if_no_header(report, did_atleast_one_header)
        report += [title("TEMPORAL PEAK")]
        did_header = did_atleast_one_header = False

        for n, ap in enumerate(apsis):
            peak_idx = np.argmax(ap)
            if peak_idx != 0:
                if not did_header:
                    report += [("Found filters with temporal peak not at t=0!, "
                                "with following peak locations:\n")]
                    did_header = did_atleast_one_header = True
                report += ["psi_fs[{}]: {}\n".format(n, peak_idx)]
                data['time_peak_idx'][n] = peak_idx

        # check that there is only one temporal peak #########################
        did_header = False
        for n, ap in enumerate(apsis):
            # count number of inflection points (where sign of derivative changes)
            # exclude very small values
            # center for proper `diff`
            ap = np.fft.ifftshift(ap)
            inflections = np.diff(np.sign(np.diff(ap[ap > 10*eps])))
            n_inflections = sum(np.abs(inflections) > eps)

            if n_inflections > 1:
                if not did_header:
                    report += [("\nFound filters with multiple temporal peaks "
                                "(or incomplete/non-smooth decay)! "
                                "(more precisely, >1 inflection points) with "
                                "following number of inflection points:\n")]
                    did_header = did_atleast_one_header = True
                report += ["psi_fs[{}]: {}\n".format(n, n_inflections)]
                data['n_inflections'] = n_inflections
    else:
        pop_if_no_header(report, did_atleast_one_header)

    # Print report ###########################################################
    report = ''.join(report)
    data['report'] = report
    if verbose:
        if len(report) == 0:  # no-cov
            print("Perfect filterbank!")
        else:
            print(report)
    return data


# reusables / convenience ####################################################
def compute_lp_sum(psi_fs, phi_f=None, J=None, log2_T=None,
                   fold_antianalytic=False):
    lp_sum = 0
    for psi_f in psi_fs:
        lp_sum += np.abs(psi_f)**2
    if phi_f is not None and (
            # else lowest frequency bandpasses are too attenuated
            log2_T is not None and J is not None and log2_T >= J):
        lp_sum += np.abs(phi_f)**2

    if fold_antianalytic:
        lp_sum = fold_lp_sum(lp_sum, analytic_part=True)
    return lp_sum


def fold_lp_sum(lp_sum, analytic_part=True):
    if analytic_part:
        # reflect anti-analytic part onto analytic;
        # goal is energy conservation - if this is ignored and we
        # normalize analytic part perfectly to 2, the non-zero negative
        # freqs will make the filterbank energy-expansive

        # sum onto positives, excluding DC and Nyquist,
        # from negatives, excluding Nyquist
        lp_sum[1:len(lp_sum)//2] += lp_sum[len(lp_sum)//2 + 1:][::-1]
        # zero what we just carried over to not duplicate later by accident
        lp_sum[len(lp_sum)//2 + 1:] = 0
        # with `analytic=True`, this has no effect (all negatives == 0)
        # (note, "analytic" in "analytic_only" includes pseudo-analytic)
    else:
        # above, but in reverse
        lp_sum[len(lp_sum)//2 + 1:] += lp_sum[1:len(lp_sum)//2][::-1]
        lp_sum[1:len(lp_sum)//2] = 0
    return lp_sum


# decimate object ############################################################
class Decimate():
    def __init__(self, backend='numpy', gpu=None, dtype=None,
                 sign_correction='abs', cutoff_mult=1.):
        """Windowed-sinc decimation.

        Parameters
        ----------
        backend : str['numpy', 'torch', 'tensorflow'] / module
            Name of module, or module object, to use as backend.

              - 'torch' defaults to using GPU and single precision.
              - 'tensorflow' is not supported.

        gpu : bool / None
            Whether to use GPU (torch/tensorflow backends only). For 'torch'
            backend, defaults to True.

        dtype : str['float32', 'float64'] / None
            Whether to compute and store filters in single or double precision.

        sign_correction: str / None
            None: no correction

            'abs': `abs(out)`.
                An explored alternative was `out -= out.min()`, but it's not
                favored per
                  - shifting the entire output (dc bias), while the negatives
                    don't result from such a shift
                  - the negatives are in minority and vary with "noisy" factors
                    such as boundary effects and signal regularity, making
                    the process itself noisy and sensitive to outliers
        """
        # input checks
        assert sign_correction in (None, 'abs'), sign_correction
        if not isinstance(dtype, (str, type(None))):
            dtype = str(dtype).split('.')[-1]  # e.g. 'torch.float32'
        assert dtype in (None, 'float32', 'float64'), dtype

        self.dtype = dtype
        self.sign_correction = sign_correction
        self.cutoff_mult = cutoff_mult

        # handle `backend`
        if isinstance(backend, str):
            self.backend_name = backend
            import importlib
            backend = importlib.import_module('wavespin.scattering1d.backend.'
                                              + self.backend_name + "_backend",
                                              'backend').backend
        else:
            self.backend_name = backend.__module__.split('.')[-1].rstrip(
                '_backend')
        self.Bk = backend

        # complete module of backend
        if self.backend_name == 'torch':
            import torch
            self.B = torch
        elif self.backend_name == 'tensorflow':
            raise NotImplementedError("currently only 'numpy' and 'torch' "
                                      "backends are supported.")
            # import tensorflow as tf
            # self.B = tf
        else:
            self.B = np

        # handle `gpu`
        if gpu is None:
            gpu = bool(self.backend_name != 'numpy')
        elif gpu and self.backend_name == 'numpy':  # no-cov
            self._err_backend()
        self.gpu = gpu

        # instantiate reusables
        self.filters = {}
        self.unpads = {}
        self.pads = {}

    def __call__(self, x, factor, axis=-1, x_is_fourier=False):
        """Decimate input (anti-alias filter + subsampling).

        Parameters
        ----------
        x : tensor
            n-dim tensor.

        factor : int
            Subsampling factor, must be power of 2.

        axis : int
            Axis along which to decimate. Negative supported.

        x_is_fourier : bool (default False)
            Whether `x` is already in frequency domain.
            If possible, it's more performant to pass in `x` in time domain
            as it's passed to time domain anyway before padding (unless it
            won't require padding, which is possible).

        Returns
        -------
        o : tensor
            `x` decimated along `axis` axis by `factor` factor.
        """
        assert np.log2(factor).is_integer()
        key = (factor, x.shape[axis])
        if key not in self.filters:
            self.make_filter(key)
        return self.decimate(x, key, axis, x_is_fourier)

    def decimate(self, x, key, axis=-1, x_is_fourier=False):
        xf, filtf, factor, ind_start, ind_end = self._handle_input(
            x, key, axis, x_is_fourier)

        # convolve, subsample, unpad
        of = xf * filtf
        of = self.Bk.subsample_fourier(of, factor, axis=axis)
        o = self.Bk.irfft(of, axis=axis)
        o = self.Bk.unpad(o, ind_start, ind_end, axis=axis)

        # sign correction
        if self.sign_correction == 'abs':
            o = self.B.abs(o)

        return o

    def _handle_input(self, x, key, axis, x_is_fourier):
        # from `key` get filter & related info
        factor, N = key
        filtf = self.filters[key]
        ind_start, ind_end = self.unpads[key]
        pad_left, pad_right = self.pads[key]

        # pad `x` if necessary; handle domain
        if pad_left != 0 or pad_right != 0:
            if x_is_fourier:
                xf = x
                x = self.Bk.ifft(xf, axis=axis)
            xp = self.Bk.pad(x, pad_left, pad_right, pad_mode='zero', axis=axis)
            xf = self.Bk.fft(xp, axis=axis)
        elif not x_is_fourier:
            xf = self.Bk.fft(x, axis=axis)
        else:
            xf = x

        # broadcast filter to input's shape
        broadcast = [None] * x.ndim
        broadcast[axis] = slice(None)
        filtf = filtf[tuple(broadcast)]

        return xf, filtf, factor, ind_start, ind_end

    def make_filter(self, key):
        """Create windowed sinc, centered at n=0 and padded to a power of 2,
        and compute pad and unpad parameters.

        The filters are keyed by `key = (factor, N)`, where `factor` and `N`
        are stored with successive calls to `Decimate`, yielding dynamic
        creation and storage of filters.
        """
        q, N = key
        half_len = 10 * q
        n = int(2 * half_len)
        cutoff = (1. / q) * self.cutoff_mult

        filtf, unpads, pads = self._make_decimate_filter(n + 1, cutoff, q, N)
        self.filters[key] = filtf
        self.unpads[key] = unpads
        self.pads[key] = pads

    # helpers ################################################################
    def _make_decimate_filter(self, numtaps, cutoff, q, N):
        h = self._windowed_sinc(numtaps, cutoff)

        # for FFT conv
        ((pad_left_x, pad_right_x), (pad_left_filt, pad_right_filt)
         ) = self._compute_pad_amount(N, h)
        h = np.pad(h, [pad_left_filt, pad_right_filt])

        # time-center filter about 0 (in DFT sense, n=0)
        h = np.roll(h, -np.argmax(h))
        # take to fourier
        hf = np.fft.fft(h)
        # assert zero phase (imag part zero)
        assert hf.imag.mean() < 1e-15, hf.imag.mean()
        # keep only real part
        hf = hf.real

        # backend, device, dtype
        hf = self._handle_backend_device_dtype(hf)

        # account for additional padding
        ind_start = int(np.ceil(pad_left_x / q))
        ind_end = int(np.ceil((N + pad_left_x) / q))

        return hf, (ind_start, ind_end), (pad_left_x, pad_right_x)

    def _compute_pad_amount(self, N, h):
        # don't concern with whether it decays to zero sooner, assume worst case
        support = len(h)
        # since we zero-pad, can halve (else we'd pad by `support` on each side)
        to_pad = support
        # pow2 for fast FFT conv
        padded_pow2 = int(2**np.ceil(np.log2(N + to_pad)))

        # compute padding for input
        pad_right_x = padded_pow2 - N
        pad_left_x = 0
        # compute padding for filter
        pad_right_filt = padded_pow2 - len(h)
        pad_left_filt = 0

        return (pad_left_x, pad_right_x), (pad_left_filt, pad_right_filt)

    def _windowed_sinc(self, numtaps, cutoff):
        """Sample & normalize windowed sinc, in time domain"""
        win = scipy.signal.get_window("hamming", numtaps, fftbins=False)

        # sample, window, & norm sinc
        alpha = 0.5 * (numtaps - 1)
        m = np.arange(0, numtaps) - alpha
        h = win * cutoff * np.sinc(cutoff * m)
        h /= h.sum()  # L1 norm

        return h

    def _handle_backend_device_dtype(self, hf):
        if self.backend_name == 'numpy':
            if self.dtype == 'float32':
                hf = hf.astype('float32')
            if self.gpu:
                self._err_backend()

        elif self.backend_name == 'torch':
            hf = self.B.from_numpy(hf)
            if self.dtype == 'float32':
                hf = hf.float()
            if self.gpu:
                hf = hf.cuda()

        elif self.backend_name == 'tensorflow':  # no-cov
            raise NotImplementedError

        return hf

    def _err_backend(self):  # no-cov
        raise ValueError("`gpu=True` requires `backend` that's 'torch' "
                         "or 'tensorflow' (got %s)" % str(self.backend_name))


def _get_ranger(verbose):
    if verbose:
        ranger = trange
        if trange is None:  # no-cov
            warnings.warn("Progress bar requires `tqdm` installed.")
    else:  # no-cov
        ranger = range
    return ranger
