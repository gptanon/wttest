# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Introspection tools: coefficients, filterbank."""
import numpy as np
import warnings
import textwrap
from copy import deepcopy

from ...utils.gen_utils import ExtendedUnifiedBackend, print_table
from .postprocessing import drop_batch_dim_jtfs, jtfs_to_numpy
from .misc import energy, l2, l1, make_eps


def coeff_energy(Scx, meta, pair=None, aggregate=True, correction=False,
                 kind='l2'):
    """Computes energy of JTFS coefficients.

    Current implementation permits computing energy directly via
    `sum(abs(coef)**2)`, hence this method isn't necessary.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    pair: str / list/tuple[str] / None
        Name(s) of coefficient pairs whose energy to compute.
        If None, will compute for all.

    aggregate: bool (default True)
        True: return one value per `pair`, the sum of all its coeffs' energies
        False: return `(E_flat, E_slices)`, where:
            - E_flat = energy of every coefficient, flattened into a list
              (but still organized pair-wise)
            - E_slices = energy of every joint slice (if not `'S0', 'S1'`),
              in a pair. That is, sum of coeff energies on per-`(n2, n1_fr)`
              basis.

    correction : bool (default False)
        Whether to apply stride and filterbank norm correction factors:
            - stride: if subsampled by 2, energy will reduce by 2
            - filterbank: since input is assumed real, we convolve only over
              positive frequencies, getting half the energy

        Current JTFS implementation accounts for both so default is `False`
        (in fact `True` isn't implemented with any configuration due to
        forced LP sum renormalization - though it's possible to implement).

        Filterbank energy correction is as follows:

            - S0 -> 1
            - U1 -> 2 (because psi_t is only analytic)
            - phi_t * phi_f -> 2 (because U1)
            - psi_t * phi_f -> 4 (because U1 and another psi_t that's
                                  only analytic)
            - phi_t * psi_f -> 4 (because U1 and psi_f is only for one spin)
            - psi_t * psi_f -> 4 (because U1 and another psi_t that's
                                  only analytic)

        For coefficient correction (e.g. distance computation) we instead
        scale the coefficients by square root of these values.

    kind: str['l1', 'l2']
        Kind of energy to compute. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)`
        (so actually L2^2).

    Returns
    -------
    E: float / tuple[list]
        Depends on `pair`, `aggregate`.

    Rationale
    ---------
    Simply `sum(abs(coef)**2)` won't work because we must account for subsampling.

        - Subsampling by `k` will reduce energy (both L1 and L2) by `k`
          (assuming no aliasing).
        - Must account for variable subsampling factors, both in time
          (if `average=False` for S0, S1) and frequency.
          This includes if only seeking ratio (e.g. up vs down), since
          `(a*x0 + b*x1) / (a*y0 + b*y1) != (x0 + x1) / (y0 + y1)`.
    """
    if pair is None or isinstance(pair, (tuple, list)):
        # compute for all (or multiple) pairs
        pairs = pair
        E_flat, E_slices = {}, {}
        for pair in Scx:
            if pairs is not None and pair not in pairs:
                continue
            E_flat[pair], E_slices[pair] = coeff_energy(
                Scx, meta, pair, aggregate=False, kind=kind)
        if aggregate:
            E = {}
            for pair in E_flat:
                E[pair] = np.sum(E_flat[pair])
            return E
        return E_flat, E_slices

    elif not isinstance(pair, str):
        raise ValueError("`pair` must be string, list/tuple of strings, or None "
                         "(got %s)" % pair)

    # compute compensation factor (see `correction` docs)
    factor = _get_pair_factor(pair, correction)
    fn = lambda c: energy(c, kind=kind)
    norm_fn = lambda total_joint_stride: (2**total_joint_stride
                                          if correction else 1)

    E_flat, E_slices = _iterate_coeffs(Scx, meta, pair, fn, norm_fn, factor)
    Es = []
    for s in E_slices:
        Es.append(np.sum(s))
    E_slices = Es

    if aggregate:
        return np.sum(E_flat)
    return E_flat, E_slices


def coeff_distance(Scx0, Scx1, meta0, meta1=None, pair=None, correction=False,
                   kind='l2'):
    """Computes L2 or L1 relative distance between `Scx0` and `Scx1`.

    Current implementation permits computing distance directly between
    coefficients, as `toolkit.rel_l2(coef0, coef1)`.

    Parameters
    ----------
    Scx0, Scx1: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta0, meta1: dict[dict[np.ndarray]]
        `jtfs.meta()`. If `meta1` is None, will set equal to `meta0`.
        Note that scattering objects responsible for `Scx0` and `Scx1` cannot
        differ in any way that alters coefficient shapes.

    pair: str
        Name of coefficient pair whose distances to compute.

    kind: str['l1', 'l2']
        Kind of distance to compute. L1==`sum(abs(x))`,
        L2==`sqrt(sum(abs(x)**2))`. L1 is not implemented for `correction=False`.

    correction: bool (default False)
        See `help(wavespin.toolkit.coeff_energy)`.

    Returns
    -------
    reldist_flat : list[float]
        Relative distances between individual frequency rows, i.e. per-`n1`.

    reldist_slices : list[float]
        Relative distances between joint slices, i.e. per-`(n2, n1_fr)`.
    """
    if not correction and kind == 'l1':
        raise NotImplementedError

    if meta1 is None:
        meta1 = meta0
    # compute compensation factor (see `correction` docs)
    factor = _get_pair_factor(pair, correction)
    fn = lambda c: c

    def norm_fn(total_joint_stride):
        if not correction:
            return 1
        return (2**(total_joint_stride / 2) if kind == 'l2' else
                2**total_joint_stride)

    c_flat0, c_slices0 = _iterate_coeffs(Scx0, meta0, pair, fn, norm_fn, factor)
    c_flat1, c_slices1 = _iterate_coeffs(Scx1, meta1, pair, fn, norm_fn, factor)

    # make into array and assert shapes are as expected
    c_flat0, c_flat1 = np.asarray(c_flat0), np.asarray(c_flat1)
    c_slices0 = [np.asarray(c) for c in c_slices0]
    c_slices1 = [np.asarray(c) for c in c_slices1]

    assert c_flat0.ndim == c_flat1.ndim == 2, (c_flat0.shape, c_flat1.shape)
    is_joint = bool(pair not in ('S0', 'S1'))
    if is_joint:
        shapes = [np.array(c).shape for cs in (c_slices0, c_slices1) for c in cs]
        # effectively 3D
        assert all(len(s) == 2 for s in shapes), shapes

    d_fn = lambda x: l2(x) if kind == 'l2' else l1(x)
    ref0, ref1 = d_fn(c_flat0), d_fn(c_flat1)
    eps = make_eps(ref0, ref1)
    ref = (ref0 + ref1) / 2 + eps

    def D(x0, x1, axis):
        if isinstance(x0, list):
            return [D(_x0, _x1, axis) for (_x0, _x1) in zip(x0, x1)]

        if kind == 'l2':
            return np.sqrt(np.sum(np.abs(x0 - x1)**2, axis=axis)) / ref
        return np.sum(np.abs(x0 - x1), axis=axis) / ref

    # do not collapse `freq` dimension
    reldist_flat   = D(c_flat0,   c_flat1,   axis=-1)
    reldist_slices = D(c_slices0, c_slices1, axis=(-1, -2) if is_joint else -1)

    # return tuple consistency; we don't do "slices" here
    return reldist_flat, reldist_slices


def coeff_energy_ratios(Scx, meta, down_to_up=True, scaled=False,
                        max_to_eps_ratio=10000):
    """Compute ratios of coefficient slice energies, spin down vs up.
    Statistically robust alternative measure to ratio of total energies.

    Parameters
    ----------
    Scx : dict[list] / dict[tensor]
        `jtfs(x)`.

    meta : dict[dict[np.ndarray]]
        `jtfs.meta()`.

    down_to_up : bool (default True)
        Whether to take `E_dn / E_up` (True) or `E_up / E_dn` (False).
        Note, the actual similarities are opposite, as "down" means convolution
        with down, which is cross-correlation with up.

    scaled : bool (default False)
        `True` scales each ratio in proportion to own energy. Useful for
        accounting for coefficients' total informativeness in addition to
        FDTS discriminativeness (e.g. very low energies are likelier noise).

        If slice A has ratio 5  and energy 100, and
           slice B has ratio 10 and energy 50, then
        with `True`, both are equal, otherwise they're the original 5 and 10.
        If `ratio = a / b`, then `True` does `ratio *= (a + b)`.

        This is same as scaling relative to total energy, within a constant
        scaling factor.

    max_to_eps_ratio : int
        `eps = max(E_pair0, E_pair1) / max_to_eps_ratio`. Epsilon term
        to use in ratio: `E_pair0 / (E_pair1 + eps)`.

    Returns
    -------
    Ratios of coefficient energies.
    """
    # handle args
    assert isinstance(Scx, dict), ("`Scx` must be dict (got %s); " % type(Scx)
                                   + "set `out_type='dict:array'` or 'dict:list'")

    # compute ratios
    l2s = {}
    pairs = ('psi_t * psi_f_dn', 'psi_t * psi_f_up')
    if not down_to_up:
        pairs = pairs[::-1]
    for pair in pairs:
        _, E_slc0 = coeff_energy(Scx, meta, pair=pair, aggregate=False, kind='l2')
        l2s[pair] = np.asarray(E_slc0)

    a, b = l2s.values()
    mx = np.vstack([a, b]).max(axis=0) / max_to_eps_ratio
    eps = np.clip(mx, mx.max() / (max_to_eps_ratio * 1000), None)
    r = a / (b + eps)
    if scaled:
        r *= (a + b)

    return r


def _get_pair_factor(pair, correction):
    if pair == 'S0' or not correction:
        factor = 1
    elif 'psi' in pair:
        factor = 4
    else:
        factor = 2
    return factor


def _iterate_coeffs(Scx, meta, pair, fn=None, norm_fn=None, factor=None):
    coeffs = drop_batch_dim_jtfs(Scx)[pair]
    out_list = bool(isinstance(coeffs, list))  # i.e. dict:list

    # fetch backend
    B = ExtendedUnifiedBackend(coeffs)

    # completely flatten into (*, time)
    if out_list:
        coeffs_flat = []
        for coef in coeffs:
            c = coef['coef']
            assert c.ndim >= 2, c.shape
            coeffs_flat.extend(c)
    else:
        assert coeffs.ndim <= 3, coeffs.shape
        if coeffs.ndim == 3:  # out_3D
            coeffs = B.reshape(coeffs, (-1, coeffs.shape[-1]))
        coeffs_flat = coeffs

    # prepare for iterating
    meta = deepcopy(meta)  # don't change external dict
    if meta['n'][pair].ndim == 3:  # out_3D
        meta['n'     ][pair] = meta['n'][     pair].reshape(-1, 3)
        meta['stride'][pair] = meta['stride'][pair].reshape(-1, 2)

    assert len(coeffs_flat) == len(meta['stride'][pair]), (
        "{} != {} | {}".format(len(coeffs_flat), len(meta['stride'][pair]), pair)
        )

    # define helpers #########################################################
    def get_total_joint_stride(meta_idx):
        n_freqs = 1
        m_start, m_end = meta_idx[0], meta_idx[0] + n_freqs
        stride = meta['stride'][pair][m_start:m_end]
        assert len(stride) != 0, pair

        stride[np.isnan(stride)] = 0
        total_joint_stride = stride.sum()
        meta_idx[0] = m_end  # update idx
        return total_joint_stride

    def n_current():
        i = meta_idx[0]
        m = meta['n'][pair]
        return (m[i] if i <= len(m) - 1 else
                np.array([-3, -3]))  # reached end; ensure equality fails

    def n_is_equal(n0, n1):
        n0, n1 = n0[:2], n1[:2]  # discard U1
        n0[np.isnan(n0)], n1[np.isnan(n1)] = -2, -2  # NaN -> -2
        return bool(np.all(n0 == n1))

    # append energies one by one #############################################
    fn = fn or (lambda c: c)
    norm_fn = norm_fn or (lambda total_joint_stride: 2**total_joint_stride)
    factor = factor or 1

    is_joint = bool(pair not in ('S0', 'S1'))
    E_flat = []
    E_slices = [] if not is_joint else [[]]
    meta_idx = [0]  # make mutable to avoid passing around
    for c in coeffs_flat:
        if hasattr(c, 'numpy'):
            if hasattr(c, 'cpu') and 'torch' in str(type(c)):
                c = c.cpu()
            c = c.numpy()  # TF/torch
        n_prev = n_current()
        assert c.ndim == 1, (c.shape, pair)
        total_joint_stride = get_total_joint_stride(meta_idx)

        E = norm_fn(total_joint_stride) * fn(c) * factor
        E_flat.append(E)

        if not is_joint:
            E_slices.append(E)  # append to list of coeffs
        elif n_is_equal(n_current(), n_prev):
            E_slices[-1].append(E)  # append to slice
        else:
            E_slices[-1].append(E)  # append to slice
            E_slices.append([])

    # in case loop terminates early
    if isinstance(E_slices[-1], list) and len(E_slices[-1]) == 0:
        E_slices.pop()

    # ensure they sum to same
    Es_sum = np.sum([np.sum(s) for s in E_slices])
    adiff = abs(np.sum(E_flat) - Es_sum)
    assert np.allclose(np.sum(E_flat), Es_sum), "MAE=%.3f" % adiff

    return E_flat, E_slices


def est_energy_conservation(x, sc=None, T=None, F=None, J=None, J_fr=None,
                            Q=None, Q_fr=None, max_pad_factor=None,
                            max_pad_factor_fr=None, pad_mode=None,
                            pad_mode_fr=None, average=None, average_fr=None,
                            sampling_filters_fr=None, r_psi=None, analytic=None,
                            out_3D=None, aligned=None, jtfs=False, backend=None,
                            verbose=True, get_out=False):
    """Estimate energy conservation given scattering configurations, especially
    scale of averaging. With default settings, passing only `T`/`F`, computes the
    upper bound.

    Limitations:
      - For time scattering (`jtfs=False`) and non-dyadic length `x`, the
        estimation will be inaccurate per not accounting for energy loss due to
        unpadding.
      - With `jtfs=True`, energies are underestimated per lacking support for
        `out_3D and not average_fr`. That is, convolutions over zero-padded
        regions aren't included with `out_3D=False`. those are regions with
        assumed negligible energy that are nonetheless part of actual
        frequential input. See `out_3D` docs.

    Parameters
    ----------
    x : tensor
        1D input.

    sc : `Scattering1D` / `TimeFrequencyScattering1D` / None
        Scattering object to use. If None, will create per parameters.

    T, F, J, J_fr, Q, Q_fr, max_pad_factor, max_pad_factor_fr, pad_mode,
    pad_mode_fr, average, average_fr, sampling_filters_fr, r_psi, analytic,
    out_3D, aligned:
        Scattering parameters.

    jtfs : bool (default False)
        Whether to estimate per JTFS; if False, does time scattering.
        Must pass also with `sc` to indicate which object it is.
        If `sc` is passed, won't use unaveraged variant where appropriate,
        which won't provide upper bound on energy if `sc(average_fr=True)`.

    backend : None / str
        Backend to use (defaults to torch w/ GPU if available).

    verbose : bool (default True)
        Whether to print results to console.

    get_out : bool (default False)
        Whether to return computed coefficients and scattering objects alongside
        energy ratios.

    Returns
    -------
    ESr : dict[float]
        Energy ratios.

    Scx : tensor / dict[tensor]
        Scattering output (if `get_out==True`).

    sc : `Scattering1D` / `TimeFrequencyScattering1D`
        Scattering object (if `get_out==True`).
    """
    # warn if passing params alongside `sc`
    _kw = dict(T=T, F=F, J=J, J_fr=J_fr, Q=Q, Q_fr=Q_fr,
               max_pad_factor=max_pad_factor, max_pad_factor_fr=max_pad_factor_fr,
               pad_mode=pad_mode, pad_mode_fr=pad_mode_fr,
               average=average, average_fr=average_fr,
               sampling_filters_fr=sampling_filters_fr,
               out_3D=out_3D, aligned=aligned)
    tm_params = ('T', 'J', 'Q', 'max_pad_factor', 'pad_mode', 'average')
    fr_params = ('F', 'J_fr', 'Q_fr', 'max_pad_factor_fr', 'pad_mode_fr',
                 'average_fr', 'sampling_filters_fr', 'out_3D', 'aligned')
    all_params = (*tm_params, *fr_params)
    if sc is not None and any(_kw[arg] is not None for arg in all_params):
        warnings.warn("`sc` object provided - parametric arguments ignored.")
    elif not jtfs and any(_kw[arg] is not None for arg in fr_params):
        warnings.warn("passed JTFS parameters with `jtfs=False` -- ignored.")

    # create scattering object, if not provided
    if sc is not None:
        if jtfs:
            sc_u = sc_a = sc
    else:
        if jtfs:
            from wavespin import TimeFrequencyScattering1D as SC
        else:
            from wavespin import Scattering1D as SC

        # handle args & pack parameters
        N = x.shape[-1]
        if Q is None:
            Q = (8, 3)
        if pad_mode is None:
            pad_mode = 'reflect'
        if r_psi is None:
            r_psi = (.9, .9)
            r_psi_fr = .9 if jtfs else None
        if backend is None:
            try:
                import torch
                backend = 'torch'
            except:
                backend = 'numpy'
        elif backend == 'torch':
            import torch
        kw = dict(shape=N, J=int(np.log2(N)), T=T, max_pad_factor=max_pad_factor,
                  pad_mode=pad_mode, Q=Q, frontend=backend, r_psi=r_psi)
        if not jtfs:
            if average is None:
                average = True
            if analytic is None:
                analytic = False  # library default
            kw.update(**dict(average=average, analytic=analytic, out_type='list'))
        else:
            # handle `J_fr` & `F`
            if J_fr is None:
                if F is None:
                    sc_temp = SC(**kw)
                    n_psi1 = len(sc_temp.psi1_f)
                    J_fr = int(np.log2(n_psi1)) - 1
                    F = 2**J_fr
                else:
                    J_fr = int(np.log2(F))
            elif F is None:
                F = 2**J_fr
            # handle other args
            if pad_mode_fr is None:
                pad_mode_fr = 'conj-reflect-zero'
            if average_fr is None:
                average_fr = False
            if analytic is None:
                analytic = True  # library default
            if aligned is None:
                aligned = True
            if out_3D is None:
                out_3D = False
            if sampling_filters_fr is None:
                sampling_filters_fr = 'resample'
            if Q_fr is None:
                Q_fr = 4

            # pack JTFS args
            kw.update(**dict(max_pad_factor_fr=max_pad_factor_fr, F=F,
                             pad_mode_fr=pad_mode_fr, average_fr=average_fr,
                             analytic=analytic, Q_fr=Q_fr, out_type='dict:list',
                             sampling_filters_fr=sampling_filters_fr,
                             out_3D=out_3D, aligned=aligned, r_psi_fr=r_psi_fr))
            if average is None:
                kw_u = dict(**kw, average=False)
                kw_a = dict(**kw, average=True)
            else:
                kw_u = kw_a = dict(**kw, average=average)

        # create scattering object
        if backend == 'torch':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not jtfs:
            sc = SC(**kw)
            if backend == 'torch':
                sc = sc.to(device)
            meta = sc.meta()
        else:
            sc_u, sc_a = SC(**kw_u), SC(**kw_a)
            if backend == 'torch':
                sc_u, sc_a = sc_u.to(device), sc_a.to(device)

    # scatter
    if not jtfs:
        Scx = sc(x)
        Scx = jtfs_to_numpy(Scx)
    else:
        Scx_u = sc_u(x)
        Scx_a = sc_a(x)
        Scx_u, Scx_a = jtfs_to_numpy(Scx_u), jtfs_to_numpy(Scx_a)

    # compute energies
    # input energy
    Ex = energy(x)
    if not jtfs and average:
        Ex /= 2**sc.log2_T

    # scattering energy & ratios
    ES = {}
    if not jtfs:
        for o in (0, 1, 2):
            ES[f'S{o}'] = np.sum([energy(Scx[int(i)]['coef']) for i in
                                  np.where(meta['order'] == o)[0]])
    else:
        for pair in Scx_u:
            Scx = Scx_u if pair not in ('S0', 'S1') else Scx_a
            ES[pair] = np.sum([energy(c['coef']) for c in Scx[pair]])

    ESr = {k: v/Ex for k, v in ES.items()}
    if not jtfs:
        ESr['total'] = np.sum(list(ES.values())) / Ex
    else:
        E_common = sum(ES[pair] for pair in ('S0', 'psi_t * phi_f',
                                             'psi_t * psi_f_up',
                                             'psi_t * psi_f_dn'))
        E_v1 = E_common + ES['phi_t * phi_f'] + ES['phi_t * psi_f']
        E_v2 = E_common + ES['S1']
        ESr['total_v1'], ESr['total_v2'] = E_v1 / Ex, E_v2 / Ex

    # print energies
    if T is None:
        T = (sc_a if jtfs else sc).T
    _txt = f", F={F}" if jtfs else ""
    print(f"E(Sx)/E(x) | T={T}{_txt}")
    for k, v in ESr.items():
        print("{:.4f} -- {}".format(v, k))

    if jtfs:
        sc = (sc_u, sc_a)
    return (ESr, Scx, sc) if get_out else ESr


def top_spinned(Scx, meta, top_k=5, Q1=None, fs=None, verbose=1):
    """Yields coefficient meta for `top_k` highest energy spinned coefficients.

    Parameters
    ----------
    Scx : dict[tensor]
        Output of JTFS with `out_type='dict:array'` or `out_type='dict:list'`.

    meta : dict
        `jtfs.meta()`.

    top_k : int[>0]
        Number of coefficients to print.

    Q1 : int / None
        `jtfs.Q[0]`.

            - None: will use `wavs` (wavelets) as numerator units for slope
            - int:  will use `octs` (octaves)

    fs : int / None
        Sampling rate.

            - None: will use `samp` (sample) as denominator units for slope
            - int:  will use `sec` (second)

    verbose : bool (default True)
        Whether to print the results or return silently.

    Returns
    -------
    info : list
        Metadata on `top_k` highest spinned coefficients.
        Data is rounded to 4 significant figures.
    """
    assert isinstance(Scx, dict)

    # unpack input
    c_up = Scx['psi_t * psi_f_up']
    c_dn = Scx['psi_t * psi_f_dn']
    if isinstance(c_up[0], dict):
        c_up = [c['coef'] for c in c_up]
        c_dn = [c['coef'] for c in c_dn]
    out_3D = bool(c_up.ndim == 4)

    # compute energies
    # put `n2 * n1_fr` unrolled dim first to iterate for energy computations
    tp_axes = (1, 0, *tuple(range(2, c_up.ndim)))
    e_up = [energy(c) for c in c_up.transpose(tp_axes)]
    e_dn = [energy(c) for c in c_dn.transpose(tp_axes)]
    energies = np.hstack([e_up, e_dn])

    # number of joint slices in each, up and down
    n_each = c_up.shape[1]
    # sort ascending, reverse for descending, get `top_k`
    idxs = np.argsort(energies)[::-1][:top_k]
    # slope scaling and units
    if fs is None and Q1 is None:
        slope_scale = 1
        slope_units = 'wavs/samp'
    elif fs is None:
        slope_scale = 1/Q1
        slope_units = 'octs/samp'
    elif Q1 is None:
        slope_scale = fs
        slope_units = 'wavs/sec'
    else:
        slope_scale = fs / Q1
        slope_units = 'octs/sec'

    # create report ##########################################################
    e_spinned = np.sum(energies)
    info = []
    for idx in idxs:
        # energy ratio
        e_ratio = float("%.4g" % (energies[idx] / e_spinned))
        # spin
        if idx <= n_each:
            pair = 'psi_t * psi_f_up'
            spin = 1
        else:
            pair = 'psi_t * psi_f_dn'
            spin = -1
            idx = idx - n_each
        # n
        _n = meta['n'][pair][idx]
        _n = _n[0, :2] if out_3D else _n
        n = tuple(_n)
        # slope
        _slope = coeff2meta_jtfs(Scx, meta, idx, pair)['slope']
        if out_3D:
            _slope = _slope[0]
        slope = float("%.4g" % (_slope * slope_scale))
        # append
        info.append((n, spin, slope, e_ratio))

    # return, and maybe print report #########################################
    if verbose:
        e_spinned = np.sum(energies)
        txt = ""
        for entry in info:
            # unpack meta
            n, spin, slope, e_ratio, *_ = entry
            sign = '+' if spin == 1 else '-'
            txt += "{}{}: {:.3g}%, {:<4.3g} {}\n".format(
                sign, n, e_ratio * 100, slope, slope_units)
        txt = txt.rstrip('\n')

        n_title = "n2, n1_fr" if out_3D else "n2, n1_fr, n1"
        print(("Top {} spinned coefficients:\n"
           "spin({}): % energy (spinned)\n"
           "{}").format(top_k, n_title, txt))

    return info


def scattering_info(sc, specs=True, show=True):
    """Prints relevant info about the input scattering object.

    Parameters
    ----------
    sc : Scattering1D / TimeFrequencyScattering1D
        Scattering instance.

    specs : bool (default True)
        Whether to also print parameter information (`J`, `Q`, etc).
        Shows finalized values as opposed to input, which may be particularly
        informative for certain parameters (e.g. `pad_mode` for second order).

    show : bool (default True)
        Whether to print to console. If `False`, will instead silently return
        the info string.

    Returns
    -------
    all_txt : str
        Info string, only if `show=False`.

    Clarification
    -------------

        - Quality Factor = `xi` / `sigma` = (center freq) / bandwidth
        - CQT = Constant-Q Transform; non-CQT ~= STFT
        - `'slope' [wav/samp]` is for *spinned*, in wavelets/sample.
          See `help(wavespin.scattering1d.scat_utils.compute_meta_jtfs)`.
        - For other meta, see `compute_meta_scattering` in same file.
    """
    is_jtfs = bool(hasattr(sc, 'scf'))
    all_txt = ""
    sc_name = ("TimeFrequencyScattering1D" if is_jtfs else
               "Scattering1D")

    def fmt(x):
        """Float formatting."""
        return float("{:.3g}".format(x))

    # "Info" #################################################################
    # Quality Factor
    pf1 = sc.psi1_f[0]
    pf2 = sc.psi2_f[0]
    QF1 = pf1['xi'] / pf1['sigma']
    QF2 = pf2['xi'] / pf2['sigma']
    if is_jtfs:
        pf_fr = sc.psi1_f_fr_up
        QF_fr = pf_fr['xi'][0][0] / pf_fr['sigma'][0][0]

    QF = [QF1, QF2]
    if is_jtfs:
        QF.append(QF_fr)

    # CQT vs non-CQT
    n_psi1 = len(sc.psi1_f)
    n_psi2 = len(sc.psi2_f)
    n_psi1_noncqt = sum(not p['is_cqt'] for p in sc.psi1_f)
    n_psi2_noncqt = sum(not p['is_cqt'] for p in sc.psi2_f)
    noncqt_perc1 = 100 * n_psi1_noncqt / n_psi1
    noncqt_perc2 = 100 * n_psi2_noncqt / n_psi2
    if is_jtfs:
        n_psi_fr = len(sc.psi1_f_fr_up[0])
        n_psi_fr_noncqt = n_psi_fr - sum(sc.psi1_f_fr_up['is_cqt'][0])
        noncqt_perc_fr = 100 * n_psi_fr_noncqt / n_psi_fr

    n_psi = [n_psi1, n_psi2]
    noncqt_perc = [noncqt_perc1, noncqt_perc2]
    if is_jtfs:
        n_psi.append(n_psi_fr)
        noncqt_perc.append(noncqt_perc_fr)

    # meta
    def meta_fn(name):
        data = [max(p[name][0] for p in sc.psi1_f),
                max(p[name][0] for p in sc.psi2_f)]
        if is_jtfs:
            data.append(max(sc.psi1_f_fr_up[name][0]))
        return data

    info_meta = {name: meta_fn(name) for name in
                 ('support', 'width', 'scale', 'bw')}

    # formatting
    QF = [fmt(g) for g in QF]
    noncqt_perc = [fmt(g) for g in noncqt_perc]

    # join `n_psi` and `noncqt_perc`
    n_psi_noncqt_perc = [(g0, g1) for (g0, g1) in zip(n_psi, noncqt_perc)]

    # pack
    nv = {
      "Quality Factor": QF,
      "(# of psi, % non-CQT)": n_psi_noncqt_perc,
     **{f"max '{name}'": info_meta[name] for name in info_meta},
    }
    if is_jtfs:
        jmeta = sc.meta()
        spinned_slope = jmeta['slope']['psi_t * psi_f_up']
        max_ss = fmt(max(spinned_slope))
        min_ss = fmt(min(spinned_slope))
        nv.update(**{
          "": "",
          "max 'slope' [wav/samp]": max_ss,
          "min 'slope' [wav/samp]": min_ss,
        })
    # convert for `print_table`
    names, values = list(nv), list(nv.values())

    info = f"Info: {sc_name}\n\n"
    txt = print_table(names, values, show=False)
    txt = textwrap.indent(txt, '    ')
    info += txt
    all_txt += info

    # "Specs" ################################################################
    if specs:
        def fetch_attr(name):
            """Getter helper."""
            if name == 'T':
                out = 2*[sc.T]
                if is_jtfs:
                    out.append(sc.F)
            elif name in ('J', 'Q', 'normalize', 'r_psi'):
                out = [*getattr(sc, name)]
                if is_jtfs:
                    out.append(getattr(sc.scf, f"{name}_fr"))
            else:
                out = 2*[getattr(sc, name)]
                if is_jtfs:
                    out.append(getattr(sc.scf, f"{name}_fr"))
            return out

        # average
        average12 = ('global' if sc.average and sc.T == 'global' else
                     bool(sc.average))
        if is_jtfs:
            average_fr = ('global' if sc.average_fr and sc.F == 'global' else
                          bool(sc.average_fr))
        average = 2*[average12]
        if is_jtfs:
            average.append(average_fr)

        # pad_mode
        pad_mode1 = sc.pad_mode
        pad_mode2 = ('conj-reflect' if (pad_mode1 == 'reflect' and sc.average)
                     else pad_mode1)
        if is_jtfs:
            pad_mode_fr = sc.scf.pad_mode_fr

        pad_mode = [pad_mode1, pad_mode2]
        if is_jtfs:
            pad_mode.append(pad_mode_fr)

        # r_psi
        r_psi = fetch_attr('r_psi')

        # formatting
        r_psi = [fmt(g) for g in r_psi]

        # pack
        nv = {
          'J': fetch_attr('J'),
          'Q': fetch_attr('Q'),
          'T': fetch_attr('T'),
          'r_psi': r_psi,
          'average': average,
          'analytic': fetch_attr('analytic'),
          'pad_mode': pad_mode,
          'max_pad_factor': fetch_attr('max_pad_factor'),
          'oversampling': fetch_attr('oversampling'),
          'normalize': fetch_attr('normalize'),
          'smart_paths': sc.smart_paths,
        }

        if is_jtfs:
            nv.update(**{
              '': '',
              'out_3D': sc.out_3D,
              'aligned': sc.aligned,
              'sampling_filters_fr': sc.sampling_filters_fr,
              'F_kind': f"'{sc.F_kind}'",
            })
        # convert for `print_table`
        names, values = list(nv), list(nv.values())

        specs = f"\nSpecs: {sc_name}\n\n"
        txt = print_table(names, values, show=False)
        txt = textwrap.indent(txt, '    ')
        specs +=  txt
        all_txt += specs

    # Finalize ###############################################################
    if is_jtfs:
        legend_txt = ("\n[a, b, c] <=> [First order, Second order, "
                      "Frequential order]")
    else:
        legend_txt = "\n[a, b] <=> [First order, Second order]"
    all_txt += legend_txt

    if show:
        print(all_txt)
    else:
        return all_txt


def coeff2meta_jtfs(Scx, meta, out_idx, pair=None):
    """Fetches JTFS meta based on output coefficient indices.

    Not fully tested, experimental method.

    Parameters
    ----------
    Scx : tensor / list / dict
        Output of `jtfs(x)`.

    meta : dict
        `jtfs.meta()`.

    out_idx : int / tuple[int]
        Coefficient index.

        If `out_3D=True` and `out_type in ('array', 'list')`, then output
        `out = jtfs(x)` is a 2-tuple, and `out_idx` must also be tuple like
            - `(0, 5)` <=> `out[0][5]`
            - `(1, 5)` <=> `out[1][5]`

    pair : str / None
        For `out_type='dict:array'` or `'dict:list'`.
        `out_idx` then specifies index within this pair.

    Returns
    -------
    m_out : dict
        Meta corresponding to the coefficient.

    How it works
    ------------
    Let `o = sc(x)` and `m = sc.meta()`, with `out_3D=False`. Then,

    'array':
        # n1=5, field='j'
        o[:, 5] <=> m['j'][5]

    'list':
        k = m['key'][5]
        o[5]['coef'] <=> m['j'][k]

    'dict:array':
        # pair='S1'
        o['S1'][:, 5] <=> m['j']['S1'][5]

        # pair='psi_t * psi_f_up'
        k = m['key']['psi_t * psi_f_up'][3]
        o['psi_t * psi_f_up'][:, 3] <=> m['j']['psi_t * psi_f_up'][k]

    'dict:list':
        o['S1'][5]['coef'] <=> m['j']['S1'][5]

        k = m['key']['psi_t * psi_f_up'][3]
        o['psi_t * psi_f_up'][3]['coef'] <=> m['j']['psi_t * psi_f_up'][k]
    """
    if isinstance(Scx, tuple) or isinstance(meta, tuple):
        assert isinstance(Scx, tuple) and isinstance(meta, tuple)
        assert len(Scx) == len(meta) == 2
        meta1, meta2 = meta
        order_idx, coef_idx = out_idx
        offset = len(meta1['j'])
        meta = meta[order_idx]
        out_idx = coef_idx
    else:
        offset = None

    o, m = _process_meta_conversion_inputs(Scx, meta, pair)

    k = m['key'][out_idx]
    if offset is not None:
        k = slice(k.start - offset, k.stop - offset, k.step)
    m_out = {field: m[field][k] for field in m if field != 'key'}
    return m_out


def meta2coeff_jtfs(Scx, meta, meta_goal, pair=None):
    """Fetches JTFS coefficients whose meta matches a given specification.

    Not fully tested, experimental method.

    Parameters
    ----------
    Scx : tensor / list / dict
        Output of `jtfs(x)`.

    meta : dict
        `jtfs.meta()`.

    meta_goal : dict
        One or more meta key-value pairs. Examples:

            - {'j': [2, 0, 3]}  # j2=2, j1_fr=0, j1=3
            - {'j': [2, 0, 3], 'is_cqt': False, 'spin': 1}
            - {'xi': [None, None, .25], 'j': [1, 2, None]}

        `None` to ignore axis, so `'j': [None, None, 1]` will fetch all `j1=1`.

        `'xi'` and `'sigma'` will be fetched as closest instead of exact match,
        so `0.25` will return `0.248` but not `0.254`.

        Does not support multiple values for same key, e.g.
        `{'j': [[2, 0, 3], [2, 0, 1]]}` - instead the method can be called
        iteratively.

    pair : str / None
        For `out_type='dict:array'` or `'dict:list'`.
        `meta_goal` then specifies meta within this pair.

    Returns
    -------
    c_outs : list
        Coefficients, as indexings of `Scx`. So if `out_type='list'`, then
        coefficients are the usual dictionaries with a `'coef'` key, otherwise
        they're tensors.
    """
    o, m = _process_meta_conversion_inputs(Scx, meta, pair)

    # find all indices that satisfy `meta_goal`
    m_idxs = []
    for field, value in meta_goal.items():
        mv = m[field]
        # replace with what can be compared directly
        value = value.copy()
        mv = mv.copy()
        for i, v in enumerate(value):
            if v is not None and np.isnan(v):
                value[i] = 999
        mv[np.isnan(mv)] = 999

        bools = []
        for ax, v in enumerate(value):
            if v is None:
                continue
            if field in ('xi', 'sigma'):
                # find closest match
                diffs = np.abs(mv[:, ax] - v)
                closest_idx = np.argmin(diffs)
                diffs_min = diffs[closest_idx]
                # avoid using ==, fetch close enough
                bools.append(diffs < diffs_min*1.001)
            else:
                bools.append(mv[:, ax] == v)
        idxs = np.where(np.prod(bools, axis=0))[0]
        m_idxs.extend(idxs)

    if len(m_idxs) == 0:
        raise ValueError("Meta not found: {}".format(meta_goal))
    m_idxs = np.unique(m_idxs)

    out_idxs_and_meta = []
    for i, k in enumerate(m['key']):
        for mi in m_idxs:
            if mi in range(k.start, k.stop):
                coef_meta = {field: m[field][mi] for field in m}
                out_idxs_and_meta.append((i, coef_meta))

    c_outs = [(o[out_idx] if isinstance(o, dict) else o[:, out_idx], mt)
              for (out_idx, mt) in out_idxs_and_meta]
    return c_outs


def _process_meta_conversion_inputs(Scx, meta, pair):
    o, m = Scx, meta
    assert (pair is not None if isinstance(o, dict) else
            pair is None)
    if isinstance(o, dict):
        o = o[pair]
        m = {field: m[field][pair] for field in m}

    out_3D = bool(m['j'].ndim == 3)
    if out_3D:
        for field in m:
            if field != 'key':
                reshape = ((-1, m[field].shape[-1]) if m[field].ndim == 3 else
                           (-1,))
                m[field] = m[field].reshape(*reshape)
    return o, m
