# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import numpy as np
import math
import warnings
from copy import deepcopy

from .filter_bank import calibrate_scattering_filters, gauss_1d, morlet_1d
from .refining import smart_paths_exclude, primitive_paths_exclude
from ..utils.measures import (compute_spatial_support,
                              compute_minimum_required_length)


def compute_border_indices(log2_T, J, i0, i1):
    """
    Computes border indices at all scales which correspond to the original
    signal boundaries after padding.

    At the finest resolution,
    original_signal = padded_signal[..., i0:i1].
    This function finds the integers i0, i1 for all temporal subsamplings
    by 2**J, being conservative on the indices.

    Maximal subsampling is by `2**log2_T` if `average=True`, else by
    `2**max(log2_T, J)`. We compute indices up to latter to be sure.

    Parameters
    ----------
    log2_T : int
        Maximal subsampling by low-pass filtering is `2**log2_T`.
    J : int / tuple[int]
        Maximal subsampling by band-pass filtering is `2**J`.
    i0 : int
        start index of the original signal at the finest resolution
    i1 : int
        end index (excluded) of the original signal at the finest resolution

    Returns
    -------
    ind_start, ind_end: dictionaries with keys in [0, ..., log2_T] such that the
        original signal is in padded_signal[ind_start[j]:ind_end[j]]
        after subsampling by 2**j

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/utils.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    if isinstance(J, tuple):
        J = max(J)
    ind_start = {0: i0}
    ind_end = {0: i1}
    for j in range(1, max(log2_T, J) + 1):
        ind_start[j] = (ind_start[j - 1] // 2) + (ind_start[j - 1] % 2)
        ind_end[j] = (ind_end[j - 1] // 2) + (ind_end[j - 1] % 2)
    return ind_start, ind_end


def compute_padding(J_pad, N):
    """
    Computes the padding to be added on the left and on the right
    of the signal.

    It should hold that 2**J_pad >= N

    Parameters
    ----------
    J_pad : int
        2**J_pad is the support of the padded signal
    N : int
        original signal support size

    Returns
    -------
    pad_left: amount to pad on the left ("beginning" of the support)
    pad_right: amount to pad on the right ("end" of the support)

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/utils.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    N_pad = 2**J_pad
    if N_pad < N:
        raise ValueError('Padding support should be larger than the original '
                         'signal size!')
    to_add = 2**J_pad - N
    pad_right = to_add // 2
    pad_left = to_add - pad_right
    return pad_left, pad_right


def compute_minimum_support_to_pad(N, J, Q, T, criterion_amplitude=1e-3,
                                   normalize='l1', r_psi=math.sqrt(0.5),
                                   sigma0=.13, P_max=5, eps=1e-7,
                                   pad_mode='reflect'):
    """
    Computes the support to pad given the input size and the parameters of the
    scattering transform.

    Parameters
    ----------
    N : int
        temporal size of the input signal
    J : int
        scale of the scattering
    Q : int >= 1
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
          - If `Q1==0`, will exclude `psi1_f` from computation.
          - If `Q2==0`, will exclude `psi2_f` from computation.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    normalize : string / tuple[string]
        Normalization convention for the filters (in the temporal domain).
        Supports 'l1', 'l2', 'l1-energy', 'l2-energy', but only 'l1' or 'l2' is
        used. See `help(Scattering1D)`.
    criterion_amplitude: float `>0` and `<1`
        Represents the numerical error which is allowed to be lost after
        convolution and padding.
        The larger `criterion_amplitude`, the smaller the padding size is.
        Defaults to `1e-3`
    r_psi : float
        See `help(wavespin.Scattering1D())`.
    sigma0 : float
        See `help(wavespin.Scattering1D())`.
    P_max : int
        See `help(wavespin.Scattering1D())`.
    eps : float
        See `help(wavespin.Scattering1D())`.
    pad_mode : str
        Name of padding used. If 'zero', will halve `min_to_pad`, else no effect.

    Returns
    -------
    min_to_pad: int
        Minimal value to pad the signal to avoid boundary effects and insufficient
        filter decay.
    """
    # compute params for calibrating, & calibrate
    Q1, Q2 = Q if isinstance(Q, tuple) else (Q, 1)
    Q_temp = (max(Q1, 1), max(Q2, 1))  # don't pass in zero
    N_init = N

    # `None` means `xi_min` is limitless. Since this method is used to compute
    # padding, then we can't know what it is, so we compute worst case.
    # If `max_pad_factor=None`, then the realized filterbank's (what's built)
    # `xi_min` is also limitless. Else, it'll be greater, depending on
    # `max_pad_factor`.
    J_pad = None
    sigma_low, xi1, sigma1, _, xi2, sigma2, _ = calibrate_scattering_filters(
        J, Q_temp, T, r_psi=r_psi, sigma0=sigma0, J_pad=J_pad)

    # split `normalize` into orders
    if isinstance(normalize, tuple):
        normalize1, normalize2 = normalize
    else:
        normalize1 = normalize2 = normalize

    # compute psi1_f with greatest time support, if requested
    if Q1 >= 1:
        psi1_f_fn = lambda N: morlet_1d(
            N, xi1[-1], sigma1[-1], normalize=normalize1, P_max=P_max, eps=eps)
    # compute psi2_f with greatest time support, if requested
    if Q2 >= 1:
        psi2_f_fn = lambda N: morlet_1d(
            N, xi2[-1], sigma2[-1], normalize=normalize2, P_max=P_max, eps=eps)
    # compute lowpass
    phi_f_fn = lambda N: gauss_1d(N, sigma_low, normalize=normalize1,
                                  P_max=P_max, eps=eps)

    # compute for all cases as psi's time support might exceed phi's
    ca = dict(criterion_amplitude=criterion_amplitude)
    N_min_phi = compute_minimum_required_length(phi_f_fn, N_init=N_init, **ca)
    phi_support = compute_spatial_support(phi_f_fn(N_min_phi), **ca)

    if Q1 >= 1:
        N_min_psi1 = compute_minimum_required_length(psi1_f_fn, N_init=N_init,
                                                     **ca)
        psi1_support = compute_spatial_support(psi1_f_fn(N_min_psi1), **ca)
    else:
        psi1_support = -1  # placeholder
    if Q2 >= 1:
        N_min_psi2 = compute_minimum_required_length(psi2_f_fn, N_init=N_init,
                                                     **ca)
        psi2_support = compute_spatial_support(psi2_f_fn(N_min_psi2), **ca)
    else:
        psi2_support = -1

    # set min to pad based on each
    pads = (phi_support, psi1_support, psi2_support)

    # can pad half as much
    if pad_mode == 'zero':
        pads = [p//2 for p in pads]
    pad_phi, pad_psi1, pad_psi2 = pads
    # set main quantity as the max of all
    min_to_pad = max(pads)

    # return results
    return min_to_pad, pad_phi, pad_psi1, pad_psi2


# arg handling ###############################################################
def _check_runtime_args_common(x):
    if x.ndim < 1:
        raise ValueError("x should be at least 1D. Got %s" % str(x))


def _check_runtime_args_scat1d(out_type, average):
    if out_type == 'array' and not average:
        raise ValueError("out_type=='array' and `not average` are mutually "
                         "incompatible. Please set out_type='list'.")

    if out_type not in ('array', 'list'):
        raise RuntimeError("`out_type` must be one of: 'array', 'list'. "
                           "Got %s" % out_type)


def _check_runtime_args_jtfs(average, average_fr, out_type, out_3D):
    if 'array' in out_type and not average:
        raise ValueError("Options `average=False` and `'array' in out_type` "
                         "are mutually incompatible. "
                         "Please set out_type='list' or 'dict:list'")

    if out_3D and not average_fr:
        raise ValueError("`out_3D=True` requires `average_fr=True`.")

    supported = ('array', 'list', 'dict:array', 'dict:list')
    if out_type not in supported:
        raise RuntimeError("`out_type` must be one of: {} (got {})".format(
            ', '.join(supported), out_type))


def _handle_input_and_backend(self, x):
    if self.frontend_name == 'torch':
        import torch
        backend_obj = torch
    elif self.frontend_name == 'tensorflow':
        import tensorflow as tf
        backend_obj = tf
    else:
        backend_obj = np

    # ensure shape's as expected
    batch_shape = x.shape[:-1]
    signal_shape = x.shape[-1:]
    if self.frontend_name != 'tensorflow':
        x = x.reshape((-1, 1) + signal_shape)
    else:
        x = tf.reshape(x, tf.concat(((-1, 1), signal_shape), 0))

    # handle torch
    if self.frontend_name == 'torch':
        self.load_filters()

        # convert to tensor if it isn't already
        is_jtfs = bool(hasattr(self, 'scf'))
        p_ref = (self.psi1_f_stacked if
                 (self.vectorized_early_U_1 and not is_jtfs) else
                 self.psi1_f[0][0])
        device = p_ref.device.type

        if type(x).__module__.split('.')[0] == 'numpy':
            x = torch.from_numpy(x).to(device=device)
        if x.device.type != device:
            x = x.to(device)

    return x, batch_shape, backend_obj


def _restore_batch_shape(Scx, batch_shape, frontend_name, out_type, backend_obj,
                         out_3D=None, out_structure=None, is_jtfs=False):
    # handle JTFS ############################################################
    if is_jtfs:
        if len(batch_shape) == 1:
            return Scx  # good to go
        elif out_structure is not None:
            if isinstance(Scx, tuple):
                Scx = list(Scx)
                for i in range(len(Scx)):
                    Scx[i] = Scx[i].reshape(*batch_shape, *Scx[i].shape[1:])
                Scx = tuple(Scx)
            else:
                Scx = Scx.reshape(*batch_shape, *Scx.shape[1:])
            return Scx
        else:
            args = (batch_shape, frontend_name, out_type.lstrip('dict:'),
                    backend_obj, out_3D, out_structure, is_jtfs)
            if out_type.startswith('dict:'):
                for pair in Scx:
                    Scx[pair] = _restore_batch_shape(Scx[pair], *args)
                return Scx
            elif out_3D and isinstance(Scx, tuple):
                Scx = list(Scx)
                for i in range(2):
                    Scx[i] = _restore_batch_shape(Scx[i], *args)
                Scx = tuple(Scx)
                return Scx

    # reshape ################################################################
    if frontend_name != 'tensorflow':
        if out_type.endswith('array'):
            scattering_shape = (Scx.shape[1:] if is_jtfs else
                                Scx.shape[-2:])
            new_shape = batch_shape + scattering_shape
            Scx = Scx.reshape(new_shape)
        elif out_type.endswith('list'):
            for c in Scx:
                scattering_shape = (c['coef'].shape[1:] if is_jtfs else
                                    c['coef'].shape[-1:])
                new_shape = batch_shape + scattering_shape
                c['coef'] = c['coef'].reshape(new_shape)
    else:
        tf = backend_obj
        if out_type.endswith('array'):
            scattering_shape = (Scx.shape[1:] if is_jtfs else
                                Scx.shape[-2:])
            new_shape = tf.concat((batch_shape, scattering_shape), 0)
            Scx = tf.reshape(Scx, new_shape)
        elif out_type.endswith('list'):
            for c in Scx:
                scattering_shape = (c['coef'].shape[1:] if is_jtfs else
                                    c['coef'].shape[-1:])
                new_shape = tf.concat((batch_shape, scattering_shape), 0)
                c['coef'] = tf.reshape(c['coef'], new_shape)

    return Scx


def _handle_args_jtfs(out_type, kwargs):
    from .frontend.base_frontend import ScatteringBase1D

    # subclass's `out_type`
    subcls_out_type = out_type.lstrip('dict:')
    # need for joint wavelets
    max_order_tm = 2
    # time scattering object shouldn't process any smart paths
    smart_paths_tm = 0

    # split tm & JTFS
    kwargs = deepcopy(kwargs)  # don't affect original
    for_jtfs = ('paths_exclude',)  # redirected kwargs
    kwargs_tm, kwargs_fr = {}, {}
    for name in ScatteringBase1D.SUPPORTED_KWARGS:
        if name in kwargs and name not in for_jtfs:
            kwargs_tm[name] = kwargs.pop(name)
    kwargs_fr = kwargs

    return max_order_tm, subcls_out_type, smart_paths_tm, kwargs_tm, kwargs_fr


# misc #######################################################################
def _handle_paths_exclude(paths_exclude, j_all, n_psis, supported, names=None):
    """Handles `paths_exclude` argument of Scattering1D or
    TimeFrequencyScattering1D.

      - Ensures `paths_exclude` is dict and doesn't have unsupported keys
      - Ensures the provided n and j aren't out of bounds
      - Handles negative indexing
      - Handles `key: int` (expected `key: list[int]`)
      - "Converts" from j to n (fills all 'n' that have the specified 'j')
      - Doesn't handle `'n2, n1'`
    """
    # check basic structure
    if paths_exclude is None:
        paths_exclude = {nm: [] for nm in supported}
        return paths_exclude
    elif not isinstance(paths_exclude, dict):
        raise ValueError("`paths_exclude` must be dict, got %s" % type(
            paths_exclude))

    # fill what's missing as we can't change size of dict during iteration
    for nm in supported:
        if nm not in paths_exclude:
            paths_exclude[nm] = []

    # in case we handle some paths separately (n2, n1_fr in JTFS)
    if names is None:
        names = list(paths_exclude)

    for p_name in names:
        # # ensure all keys are functional
        assert p_name in supported, (p_name, supported)
        # ensure list
        if isinstance(paths_exclude[p_name], int):
            paths_exclude[p_name] = [paths_exclude[p_name]]
        else:
            try:
                paths_exclude[p_name] = list(paths_exclude[p_name])
            except:
                raise ValueError(("`paths_exclude` values must be list[int] "
                                  "or int, got paths_exclude['{}'] type: {}"
                                  ).format(p_name,
                                           type(paths_exclude[p_name])))

        # n2, n1_fr ######################################################
        if p_name[0] == 'n':
            for i, n in enumerate(paths_exclude[p_name]):
                # handle negative
                if n < 0:
                    paths_exclude[p_name][i] = n_psis + n

                # warn if 'n2' already excluded
                if p_name == 'n2':
                    n = paths_exclude[p_name][i]
                    n_j2_0 = [n2 for n2 in range(n_psis) if j_all[n2] == 0]
                    if n in n_j2_0:
                        warnings.warn(
                            ("`paths_exclude['n2']` includes `{}`, which "
                             "is already excluded (alongside {}) per "
                             "having j2==0."
                             ).format(n, ', '.join(map(str, n_j2_0))))
        # j2, j1_fr ######################################################
        elif p_name[0] == 'j':
            for i, j in enumerate(paths_exclude[p_name]):
                # handle negative
                if j < 0:
                    j = max(j_all) + j
                # forbid out of bounds
                if j > max(j_all):
                    raise ValueError(("`paths_exclude` exceeded maximum {}: "
                                      "{} > {}\nTo specify max j, use `-1`"
                                      ).format(p_name, j, max(j_all)))
                # warn if 'j2' already excluded
                elif p_name == 'j2' and j == 0:
                    warnings.warn(("`paths_exclude['j2']` includes `0`, "
                                   "which is already excluded."))

                # convert to n ###########################################
                # fetch all n that have the specified j
                n_j_all = [n for n in range(n_psis) if j_all[n] == j]

                # append if not already present
                n_name = 'n2' if p_name == 'j2' else 'n1_fr'
                for n_j in n_j_all:
                    if n_j not in paths_exclude[n_name]:
                        paths_exclude[n_name].append(n_j)

    return paths_exclude


def _handle_smart_paths(smart_paths, paths_exclude, psi1_f, psi2_f):
    if isinstance(smart_paths, (tuple, float)):
        if isinstance(smart_paths, tuple):
            kw = dict(e_loss=smart_paths[0], level=smart_paths[1])
        else:
            kw = dict(e_loss=smart_paths)
        assert 0 < kw['e_loss'] < 1, smart_paths
        paths_exclude_base = smart_paths_exclude(psi1_f, psi2_f, **kw)
    elif smart_paths == 'primitive':
        paths_exclude_base = primitive_paths_exclude(psi1_f, psi2_f)
    elif not smart_paths:
        paths_exclude_base = {'n2, n1': []}
    else:
        raise ValueError("`smart_paths` must be float, str['primitive'], dict, "
                         "or `False`, got %s" % str(smart_paths))
    paths_exclude.update(paths_exclude_base)

# metas ######################################################################
def compute_meta_scattering(psi1_f, psi2_f, phi_f, log2_T, paths_include_n2n1,
                            max_order=2):
    """Get metadata of the Wavelet Time Scattering transform.

    Specifies the content of each scattering coefficient - which order,
    frequencies, filters were used, and so on. See below for more info.

    See Scattering1D docs for description of parameters.

    Returns
    -------
    meta : dictionary
        A dictionary with the following keys:

        - `'order`' : tensor
            A tensor of length `C`, the total number of scattering
            coefficients, specifying the scattering order.
        - `'xi'` : tensor
            A tensor of size `(C, max_order)`, specifying the center
            frequency of the filter used at each order (padded with NaNs).
        - `'sigma'` : tensor
            A tensor of size `(C, max_order)`, specifying the frequency
            bandwidth of the filter used at each order (padded with NaNs).
        - `'j'` : tensor
            A tensor of size `(C, max_order)`, specifying the dyadic scale
            of the filter used at each order (padded with NaNs).
        - `'is_cqt'` : tensor
            A tensor of size `(C, max_order)`, specifying whether the filter
            was constructed per Constant Q Transform (padded with NaNs).
        - `'n'` : tensor
            A tensor of size `(C, max_order)`, specifying the indices of
            the filters used at each order (padded with NaNs).
        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/utils.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # instantiate
    meta = {}
    for field in ('order', 'xi', 'sigma', 'j', 'is_cqt', 'n', 'key'):
        meta[field] = [[], [], []]

    # Zeroth order
    meta['order'][0].append(0)
    meta['xi'][0].append((phi_f['xi'],))
    meta['sigma'][0].append((phi_f['sigma'],))
    meta['j'][0].append((log2_T,))
    meta['is_cqt'][0].append(())
    meta['n'][0].append(())
    meta['key'][0].append(())

    # First order
    for n1, p1 in enumerate(psi1_f):
        xi1, sigma1, j1, is_cqt1 = [p1[field] for field in
                                    ('xi', 'sigma', 'j', 'is_cqt')]
        meta['order'][1].append(1)
        meta['xi'][1].append((xi1,))
        meta['sigma'][1].append((sigma1,))
        meta['j'][1].append((j1,))
        meta['is_cqt'][1].append((is_cqt1,))
        meta['n'][1].append((n1,))
        meta['key'][1].append((n1,))

    # Second order
    if max_order >= 2:
        for n2, p2 in enumerate(psi2_f):
            xi2, sigma2, j2, is_cqt2 = [p2[field] for field in
                                        ('xi', 'sigma', 'j', 'is_cqt')]

            for n1, p1 in enumerate(psi1_f):
                xi1, sigma1, j1, is_cqt1 = [p1[field] for field in
                                            ('xi', 'sigma', 'j', 'is_cqt')]
                if n1 in paths_include_n2n1[n2]:
                    meta['order'][2].append(2)
                    meta['xi'][2].append((xi2, xi1))
                    meta['sigma'][2].append((sigma2, sigma1))
                    meta['j'][2].append((j2, j1))
                    meta['is_cqt'][2].append((is_cqt2, is_cqt1))
                    meta['n'][2].append((n2, n1))
                    meta['key'][2].append((n2, n1))

    # join orders
    for field, value in meta.items():
        meta[field] = value[0] + value[1] + value[2]

    # left-pad with nans
    pad_fields = ['xi', 'sigma', 'j', 'is_cqt', 'n']
    pad_len = max_order

    for field in pad_fields:
        meta[field] = [(math.nan,) * (pad_len - len(x)) + x
                       for x in meta[field]]

    # to array
    array_fields = ['order', 'xi', 'sigma', 'j', 'is_cqt', 'n']

    for field in array_fields:
        meta[field] = np.array(meta[field])

    return meta


def compute_meta_jtfs(scf, psi1_f, psi2_f, phi_f, log2_T, sigma0,
                      average, average_global, average_global_phi,
                      oversampling, out_type, out_exclude, paths_exclude,
                      api_pair_order):
    """Get metadata of the Joint Time-Frequency Scattering transform.

    Specifies the content of each scattering coefficient - which order,
    frequencies, filters were used, and so on. See below for more info.

    Parameters
    ----------
    scf : `scattering1d.frontend.base_frontend._FrequencyScatteringBase1D`
        Frequential scattering object, storing pertinent attributes and filters.

    psi1_f, psi2_f, phi_f : list, list, list
        See `help(wavespin.scattering1d.TimeFrequencyScattering1D)`.
        Meta for time scattering is extracted directly from filters.

    log2_T, sigma0: int, float
        See `help(wavespin.scattering1d.TimeFrequencyScattering1D)`.

    average : bool
        Affects `S0`'s meta, and temporal stride meta.

    average_global : bool
        Affects `S0`'s meta, and temporal stride meta.

    average_global_phi : bool
        Affects joint temporal stride meta.

    oversampling : int
        See `help(wavespin.scattering1d.TimeFrequencyScattering1D)`.
        Affects temporal stride meta.

    out_type : str
         - `'dict:list'` or `'dict:array'`: meta is packed
           into respective pairs (e.g. `meta['n']['psi_t * phi_f'][1]`)
         - `'list'` or `'array'`: meta is flattened (e.g. `meta['n'][15]`).

    out_exclude : list/tuple[str]
        Names of coefficient pairs to exclude from meta.

    paths_exclude : dict / None
        See `help(wavespin.scattering1d.TimeFrequencyScattering1D)`.
        Paths to exclude from meta.

    api_pair_order : set[str]
        To follow the API ordering, alongside coefficients.

    Returns
    -------
    meta : dictionary
        Each value is a tensor, `C` is the total number of scattering coeffs,
        and each tensor is padded with NaNs where appropraite. Key `'key'` is
        an exception, which is a list.

        - `'order`' : length `C`
            Scattering order.
        - `'xi'` : shape `(C, 3)`
            Center frequency of the filter used.
        - `'sigma'` : shape `(C, 3)`
            Bandwidth of the filter used as a continuous-time parameter.
        - `'j'` : shape `(C, 3)`
            Maximum permitted subsampling factor of the filter used.
        - `'is_cqt'` : shape `(C, 3)`
            `True` if the filter was constructed per Constant Q Transform.
        - `'n'` : shape `(C, 3)`
            Indices of the filters used.
            Lowpass filters in `phi_*` pairs are denoted via `-1`.
        - `'spin'` : length `C`
            Spin of each frequential scattering filter used.
            +1=up, -1=down, 0=none.
        - `'slope'` : length `C`
            Slope of each frequential scattering filter used, in wavelets/sample.
            To convert to
               - octaves/sample:  `slope /= Q1`, `Q1 = Q[0]`.
               - wavelets/second: `slope *= fs`, `fs` is sampling rate in `Hz`.
               - octaves/second:  `slope *= (fs / Q1)`.
            NOTE: not to be interpreted as pure slope or rotation, instead as
            a product of different relative scalings along time and log-frequency.
        - `'stride'` : shape `(C, 2)`
            A tensor of size `(C, 2)`, specifying the total temporal and
            frequential convolutional stride (i.e. subsampling) of resulting
            coefficient (including lowpass filtering).
        - `'key'` : list
            `slice`s for converting between coefficient and meta indices.
            See "Coefficients <-> meta" below.

        In case of `out_3D=True`, for joint pairs, will reshape each field into
        `(n_coeffs, C, meta_len)`, where `n_coeffs` is the number of joint slices
        in the pair, and `meta_len` is the existing `shape[-1]` (1, 2, or 3).

        Tensors sized `(..., 3)` index dim 1 as `(n2, n1_fr, n1)`, so e.g.
        `meta['n'][5, 0]` fetches `n2` for coefficient indexed with 5, so the
        second-order filter used to produce that coefficient is `psi2_f[n2]`.

    Coefficients <-> meta
    ---------------------
    Meta is built such that indexing meta equates indexing coefficients for
    'array' `out_type`, and same for 'list' minus the ability to index joint
    coeffs with `n1` (also 'dict:' for each). Hence conversion is straightforward,
    but takes some data reformatting to achieve; this is facilitated by the
    `'key'` meta.

    See `wavespin.toolkit.coeff2meta_jtfs()` and
    `wavespin.toolkit.meta2coeff_jtfs()`.

    Computation and Structure
    -------------------------
    Computation replicates logic in `timefrequency_scattering1d()`. Meta values
    depend on:
        - out_3D (True only possible with `average and average_fr`)
        - aligned
        - sampling_psi_fr
        - sampling_phi_fr
        - average
        - average_global
        - average_global_phi
        - average_fr
        - average_fr_global
        - average_fr_global_phi
        - oversampling
        - oversampling_fr
        - max_pad_factor_fr (mainly via `unrestricted_pad_fr`)
        - max_noncqt_fr
        - out_exclude
        - paths_exclude
        - paths_include_build (not user-set)
    and some of their interactions. Listed are only "unobvious" parameters;
    anything that controls the filterbanks will change meta (`J`, `Q`, etc).
    """
    def _get_compute_params(n2, n1_fr):
        """Reproduce the exact logic in `timefrequency_scattering1d.py`."""
        # basics
        scale_diff = scf.scale_diffs[n2]
        J_pad_fr = scf.J_pad_frs[scale_diff]
        N_fr_padded = 2**J_pad_fr

        # n1_fr_subsample, lowpass_subsample_fr ##############################
        global_averaged_fr = (scf.average_fr_global if n1_fr != -1 else
                              scf.average_fr_global_phi)
        if n2 == -1 and n1_fr == -1:
            lowpass_subsample_fr = 0
            if scf.average_fr_global_phi:
                n1_fr_subsample = scf.log2_F
                log2_F_phi = scf.log2_F
                log2_F_phi_diff = 0
            else:
                log2_F_phi = scf.log2_F_phis['phi'][scale_diff]
                log2_F_phi_diff = scf.log2_F_phi_diffs['phi'][scale_diff]
                n1_fr_subsample = max(scf.n1_fr_subsamples['phi'][scale_diff] -
                                      scf.oversampling_fr, 0)

        elif n1_fr == -1:
            lowpass_subsample_fr = 0
            if scf.average_fr_global_phi:
                total_conv_stride_over_U1_phi = min(J_pad_fr, scf.log2_F)
                n1_fr_subsample = total_conv_stride_over_U1_phi
                log2_F_phi = scf.log2_F
                log2_F_phi_diff = 0
            else:
                n1_fr_subsample = max(scf.n1_fr_subsamples['phi'][scale_diff] -
                                      scf.oversampling_fr, 0)
                log2_F_phi = scf.log2_F_phis['phi'][scale_diff]
                log2_F_phi_diff = scf.log2_F_phi_diffs['phi'][scale_diff]

        else:
            total_conv_stride_over_U1 = (
                scf.total_conv_stride_over_U1s[scale_diff][n1_fr])
            n1_fr_subsample = max(scf.n1_fr_subsamples['spinned'
                                                       ][scale_diff][n1_fr] -
                                  scf.oversampling_fr, 0)
            log2_F_phi = scf.log2_F_phis['spinned'][scale_diff][n1_fr]
            log2_F_phi_diff = scf.log2_F_phi_diffs['spinned'][scale_diff][n1_fr]
            if global_averaged_fr:
                lowpass_subsample_fr = (total_conv_stride_over_U1 -
                                        n1_fr_subsample)
            elif scf.average_fr:
                lowpass_subsample_fr = max(total_conv_stride_over_U1 -
                                           n1_fr_subsample -
                                           scf.oversampling_fr, 0)
            else:
                lowpass_subsample_fr = 0

        # total stride, unpadding ############################################
        total_conv_stride_over_U1_realized = (n1_fr_subsample +
                                              lowpass_subsample_fr)

        if scf.out_3D:
            stride_ref = scf.total_conv_stride_over_U1s[0][0]
            stride_ref = max(stride_ref - scf.oversampling_fr, 0)
            ind_start_fr = scf.ind_start_fr_max[stride_ref]
            ind_end_fr   = scf.ind_end_fr_max[  stride_ref]
        else:
            _stride = total_conv_stride_over_U1_realized
            ind_start_fr = scf.ind_start_fr[n2][_stride]
            ind_end_fr   = scf.ind_end_fr[  n2][_stride]

        return (N_fr_padded, total_conv_stride_over_U1_realized,
                n1_fr_subsample, scale_diff, log2_F_phi_diff, log2_F_phi,
                ind_start_fr, ind_end_fr, global_averaged_fr)

    def _get_fr_params(n1_fr, scale_diff, log2_F_phi_diff, log2_F_phi):
        if n1_fr != -1:
            # spinned
            psi_id = scf.psi_ids[scale_diff]
            p = [scf.psi1_f_fr_up[field][psi_id][n1_fr]
                 for field in ('xi', 'sigma', 'j', 'is_cqt')]
        else:
            # phi_f
            if not scf.average_fr_global:
                F_phi = scf.F / 2**log2_F_phi_diff
                p = (0., sigma0 / F_phi, log2_F_phi, nan)
            else:
                p = (0., sigma0 / 2**log2_F_phi, log2_F_phi, nan)

        xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr = p
        return xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr

    def _exclude_excess_scale(n2, n1_fr):
        scale_diff = scf.scale_diffs[n2]
        psi_id = scf.psi_ids[scale_diff]
        j1_frs = scf.psi1_f_fr_up['j'][psi_id]
        return bool(n1_fr > len(j1_frs) - 1)

    def _get_slope(n2, n1_fr):
        if n2 == -1 and n1_fr == -1:
            slope = nan
        elif n2 == -1:
            slope = 0
        elif n1_fr == -1:
            slope = inf
        else:
            # fetch peaks from actual wavelets used
            psi_id = scf.psi_ids[scf.scale_diffs[n2]]
            peak1_fr = scf.psi1_f_fr_up['peak_idx'][psi_id][n1_fr]

            # compute w.r.t. original, i.e. unsubsampled, untrimmed quantities
            # (signal, scalogram, wavelet), i.e. max oversampling
            # for time this means sampling rate
            N2 = len(psi2_f[n2][0])
            # for freq this means length, meaning scale to the wavelet
            N1_fr = len(scf.psi1_f_fr_up[psi_id][0])

            # note peak index is independent of subsampling factor
            peak2 = psi2_f[n2]['peak_idx'][0]

            # compute & normalize such that slope is measured in [samples]
            slope = (peak2 / N2) / (peak1_fr / N1_fr)

        return slope

    def _skip_path(n2, n1_fr):
        # n2 >= 0 since build won't have `-1` as a key and won't skip phi_t path
        build_skip_path = bool(n2 >= 0 and n2 not in scf.paths_include_build)
        if build_skip_path:
            # can't do other checks and don't need to
            return True

        excess_scale = bool(scf.sampling_psi_fr == 'exclude' and
                            _exclude_excess_scale(n2, n1_fr))
        user_skip_path = bool(n2 in paths_exclude.get('n2', {}) or
                              n1_fr in paths_exclude.get('n1_fr', {}))
        return excess_scale or user_skip_path

    def _fill_n1_info(pair, n2, n1_fr, spin):
        if _skip_path(n2, n1_fr):
            return

        # track S1 from padding to `_joint_lowpass()`
        (N_fr_padded, total_conv_stride_over_U1_realized, n1_fr_subsample,
         scale_diff, log2_F_phi_diff, log2_F_phi, ind_start_fr, ind_end_fr,
         global_averaged_fr) = _get_compute_params(n2, n1_fr)

        # fetch xi, sigma for n2, n1_fr
        if n2 != -1:
            xi2, sigma2, j2, is_cqt2 = (xi2s[n2], sigma2s[n2], j2s[n2],
                                        is_cqt2s[n2])
        else:
            xi2, sigma2, j2, is_cqt2 = 0., sigma_low, log2_T, nan
        xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr = _get_fr_params(
            n1_fr, scale_diff, log2_F_phi_diff, log2_F_phi)

        # get temporal stride info
        global_averaged = (average_global if n2 != -1 else
                           average_global_phi)
        if global_averaged:
            total_conv_stride_tm = log2_T
        else:
            k1_plus_k2 = max(min(j2, log2_T) - oversampling, 0)
            if average:
                k2_tm_J = max(log2_T - k1_plus_k2 - oversampling, 0)
                total_conv_stride_tm = k1_plus_k2 + k2_tm_J
            else:
                total_conv_stride_tm = k1_plus_k2
        stride = (total_conv_stride_over_U1_realized, total_conv_stride_tm)

        # convert to filter format but preserve original for remaining meta steps
        n1_fr_n = n1_fr if (n1_fr != -1) else inf
        n2_n    = n2    if (n2    != -1) else inf

        # find slope
        slope = _get_slope(n2, n1_fr)

        # global average pooling, all S1 collapsed into single point
        if global_averaged_fr:
            meta['order' ][pair].append(2)
            meta['xi'    ][pair].append((xi2,     xi1_fr,     nan))
            meta['sigma' ][pair].append((sigma2,  sigma1_fr,  nan))
            meta['j'     ][pair].append((j2,      j1_fr,      nan))
            meta['is_cqt'][pair].append((is_cqt2, is_cqt1_fr, nan))
            meta['n'     ][pair].append((n2_n,    n1_fr_n,    nan))
            meta['spin'  ][pair].append(spin)
            meta['stride'][pair].append(stride)
            meta['slope' ][pair].append(slope)
            if pair not in out_exclude:
                meta['key'][pair].append(slice(rkey[:], rkey[:] + 1))
                rkey[:] += 1
            return

        fr_max = scf.N_frs[n2] if (n2 != -1) else len(xi1s)
        n_n1s = 0
        # simulate subsampling
        n1_step = 2 ** total_conv_stride_over_U1_realized

        for n1 in range(0, N_fr_padded, n1_step):
            # simulate unpadding
            if n1 / n1_step < ind_start_fr:
                continue
            elif n1 / n1_step >= ind_end_fr:
                break

            if n1 >= fr_max:  # equivalently `j1 > j2`
                # these are padded rows, no associated filters
                xi1, sigma1, j1, is_cqt1 = nan, nan, nan, nan
            else:
                xi1, sigma1, j1, is_cqt1 = (xi1s[n1], sigma1s[n1], j1s[n1],
                                            is_cqt1s[n1])
            meta['order' ][pair].append(2)
            meta['xi'    ][pair].append((xi2,     xi1_fr,     xi1))
            meta['sigma' ][pair].append((sigma2,  sigma1_fr,  sigma1))
            meta['j'     ][pair].append((j2,      j1_fr,      j1))
            meta['is_cqt'][pair].append((is_cqt2, is_cqt1_fr, is_cqt1))
            meta['n'     ][pair].append((n2_n,    n1_fr_n,    n1))
            meta['spin'  ][pair].append(spin)
            meta['stride'][pair].append(stride)
            meta['slope' ][pair].append(slope)
            n_n1s += 1

        if pair not in out_exclude:
            start, end = rkey[:], rkey[:] + n_n1s
            if 'array' in out_type:
                meta['key'][pair].extend([slice(i, i + 1)
                                          for i in range(start, end)])
            else:
                meta['key'][pair].append(slice(start, end))
            rkey[:] += n_n1s

    # handle `out_exclude`
    if out_exclude is None:
        out_exclude = ()

    # extract meta
    log2_F = scf.log2_F
    sigma_low = phi_f['sigma']

    j1_frs = scf.psi1_f_fr_up['j']
    fields = ('xi', 'sigma', 'j', 'is_cqt')
    meta1, meta2 = [{field: [p[field] for p in psi_f] for field in fields}
                    for psi_f in (psi1_f, psi2_f)]
    xi1s, sigma1s, j1s, is_cqt1s = meta1.values()
    xi2s, sigma2s, j2s, is_cqt2s = meta2.values()

    # fetch phi meta; must access `phi_f_fr` as `j1s_fr` requires sampling phi
    meta_phi = {}
    for field in ('xi', 'sigma', 'j'):
        meta_phi[field] = {}
        for k in scf.phi_f_fr[field]:
            meta_phi[field][k] = scf.phi_f_fr[field][k]
    xi1s_fr_phi, sigma1_fr_phi, j1s_fr_phi = list(meta_phi.values())

    meta = {}
    inf = -1  # placeholder for infinity
    nan = math.nan
    # this order must match, for `'dict' not in out_type`, what's used in
    # `timefrequency_scattering1d.py`
    coef_names = (
        'S0',                # (time)  zeroth order
        'S1',                # (time)  first order
        'phi_t * phi_f',     # (joint) joint lowpass
        'phi_t * psi_f',     # (joint) time lowpass
        'psi_t * phi_f',     # (joint) freq lowpass
        'psi_t * psi_f_up',  # (joint) spin up
        'psi_t * psi_f_dn',  # (joint) spin down
    )
    for field in ('order', 'xi', 'sigma', 'j', 'is_cqt', 'n', 'spin', 'stride',
                  'slope', 'key'):
        meta[field] = {name: [] for name in coef_names}

    # for 'key' meta to increment throughout pairs
    class RunningKey():
        def __init__(self): self.count = 0
        def __getitem__(self, _): return self.count
        def __setitem__(self, _, value): self.count = value

        def maybe_reset(self):
            if 'dict' in out_type:
                self.count = 0

    rkey = RunningKey()

    # Zeroth-order ###########################################################
    if average_global:
        k0 = log2_T
    elif average:
        k0 = max(log2_T - oversampling, 0)
    meta['order' ]['S0'].append(0)
    meta['xi'    ]['S0'].append((nan, nan, 0.        if average else nan))
    meta['sigma' ]['S0'].append((nan, nan, sigma_low if average else nan))
    meta['j'     ]['S0'].append((nan, nan, log2_T    if average else nan))
    meta['is_cqt']['S0'].append((nan, nan, nan))
    meta['n'     ]['S0'].append((nan, nan, inf       if average else nan))
    meta['spin'  ]['S0'].append(nan)
    meta['stride']['S0'].append((nan, k0 if average else nan))
    meta['slope' ]['S0'].append(nan)
    if 'S0' not in out_exclude:
        meta['key']['S0'].append(slice(rkey[:], rkey[:] + 1))
        rkey[:] += 1

    # First-order ############################################################
    def stride_S1(j1):
        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)
        k1_log2_T = max(log2_T - k1 - oversampling, 0)

        if average_global:
            total_conv_stride_tm = log2_T
        elif average:
            total_conv_stride_tm = k1 + k1_log2_T
        else:
            total_conv_stride_tm = k1
        return total_conv_stride_tm

    rkey[:] = 0 if out_type.startswith('dict') else rkey[:]
    for (n1, (xi1, sigma1, j1, is_cqt1)
         ) in enumerate(zip(xi1s, sigma1s, j1s, is_cqt1s)):
        meta['order' ]['S1'].append(1)
        meta['xi'    ]['S1'].append((nan, nan, xi1))
        meta['sigma' ]['S1'].append((nan, nan, sigma1))
        meta['j'     ]['S1'].append((nan, nan, j1))
        meta['is_cqt']['S1'].append((nan, nan, is_cqt1))
        meta['n'     ]['S1'].append((nan, nan, n1))
        meta['spin'  ]['S1'].append(nan)
        meta['stride']['S1'].append((nan, stride_S1(j1)))
        meta['slope']['S1'].append(nan)
        if 'S1' not in out_exclude:
            meta['key']['S1'].append(slice(rkey[:], rkey[:] + 1))
            rkey[:] += 1

    S1_len = len(meta['n']['S1'])
    assert S1_len >= scf.N_frs_max, (S1_len, scf.N_frs_max)

    # Joint scattering #######################################################
    # `phi_t * phi_f` coeffs
    rkey.maybe_reset()
    _fill_n1_info('phi_t * phi_f', n2=-1, n1_fr=-1, spin=0)

    # `phi_t * psi_f` coeffs
    rkey.maybe_reset()
    for n1_fr in range(len(j1_frs[0])):
        _fill_n1_info('phi_t * psi_f', n2=-1, n1_fr=n1_fr, spin=0)

    # `psi_t * phi_f` coeffs
    rkey.maybe_reset()
    for n2 in range(len(j2s)):
        if n2 in scf.paths_include_build:
            _fill_n1_info('psi_t * phi_f', n2, n1_fr=-1, spin=0)

    # `psi_t * psi_f` coeffs
    for spin in (1, -1):
        pair = ('psi_t * psi_f_up' if spin == 1 else
                'psi_t * psi_f_dn')
        rkey.maybe_reset()
        for n2 in range(len(j2s)):
            if n2 in scf.paths_include_build:
                psi_id = scf.psi_ids[scf.scale_diffs[n2]]
                for n1_fr, j1_fr in enumerate(j1_frs[psi_id]):
                    _fill_n1_info(pair, n2, n1_fr, spin=spin)

    # convert to array
    array_fields = ['order', 'xi', 'sigma', 'j', 'is_cqt', 'n', 'spin', 'slope',
                    'stride']
    for field in array_fields:
        for pair, v in meta[field].items():
            meta[field][pair] = np.array(v)

    # handle `out_3D` ########################################################
    if scf.out_3D:
      # reorder for 3D
      for field in array_fields:
        # meta_len
        if field in ('spin', 'slope', 'order'):
            meta_len = 1
        elif field == 'stride':
            meta_len = 2
        else:
            meta_len = 3

        for pair in meta[field]:
          # number of n2s
          if pair.startswith('phi_t'):
              n_n2s = 1
          else:
              n_n2s = sum((n2 in scf.paths_include_build and
                           n2 not in paths_exclude['n2'])
                          for n2 in range(len(j2s)))

          # number of n1_frs; n_slices
          n_slices = None
          if pair in ('S0', 'S1'):
              # simply expand dim for consistency, no 3D structure
              meta[field][pair] = meta[field][pair].reshape(-1, 1, meta_len)
              continue

          elif 'psi_f' in pair:
              if pair.startswith('phi_t'):
                  n_slices = sum(not _skip_path(n2=-1, n1_fr=n1_fr)
                                 for n1_fr in range(len(j1_frs[0])))
              else:
                  n_slices = sum(not _skip_path(n2=n2, n1_fr=n1_fr)
                                 for n2 in range(len(j2s))
                                 for n1_fr in range(len(j1_frs[0])))

          elif 'phi_f' in pair:
              n_n1_frs = 1

          # n_slices
          if n_slices is None:
              n_slices = n_n2s * n_n1_frs

          # reshape meta
          shape = (n_slices, -1, meta_len) if meta_len != 1 else (n_slices, -1)
          meta[field][pair] = meta[field][pair].reshape(shape)

    # handle `out_exclude ####################################################
    # first check that we followed the API pair order
    for field in meta:
        assert tuple(meta[field]) == api_pair_order, (
            tuple(meta['field']), api_pair_order)

    if out_exclude:
        # drop excluded pairs
        for pair in out_exclude:
            for field in meta:
                del meta[field][pair]

    # sanity checks ##########################################################
    # ensure time / freq stride doesn't exceed log2_T / log2_F in averaged cases,
    # and realized J / J_fr in unaveraged
    max_j1, max_j2 = max(j1s), max(j2s)
    max_joint_tm_j = max_j2  # joint coeff bandpass stride controlled by j2 alone
    max_j_fr = max(j1_frs[0])
    smax_t_nophi = log2_T if average else max_joint_tm_j
    if scf.average_fr:
        if not scf.out_3D and not scf.aligned:
            # see "Compute logic: stride, padding" in `core`
            smax_f_nophi = max(scf.log2_F, max_j_fr)
        else:
            smax_f_nophi = scf.log2_F
    else:
        smax_f_nophi = max_j_fr
    for pair in meta['stride']:
        if pair == 'S0' and not average:
            continue
        if pair != 'S1':
            stride_max_t = (smax_t_nophi if ('phi_t' not in pair) else
                            log2_T)
            stride_max_f = (smax_f_nophi if ('phi_f' not in pair) else
                            log2_F)
        else:
            stride_max_t = log2_T if average else max_j1

        for i, s in enumerate(meta['stride'][pair][..., 1].ravel()):
            assert s <= stride_max_t, ("meta['stride'][{}][{}] > stride_max_t "
                                       "({} > {})").format(pair, i, s,
                                                           stride_max_t)
        if pair in ('S0', 'S1'):
            continue
        for i, s in enumerate(meta['stride'][pair][..., 0].ravel()):
            assert s <= stride_max_f, ("meta['stride'][{}][{}] > stride_max_f "
                                       "({} > {})").format(pair, i, s,
                                                           stride_max_f)

    # handle `out_type #######################################################
    if not out_type.startswith('dict'):
        # join pairs
        if not scf.out_3D:
            meta_flat = {}
            for field in meta:
                vals = list(meta[field].values())
                if field != 'key':
                    vals = np.concatenate(vals, axis=0)
                else:
                    vals = [v for vpair in vals for v in vpair]
                meta_flat[field] = vals
        else:
            meta_flat0, meta_flat1 = {}, {}
            for field in meta:
                vals0 = [v for pair in meta[field] for v in meta[field][pair]
                         if pair in ('S0', 'S1')]
                vals1 = [v for pair in meta[field] for v in meta[field][pair]
                         if pair not in ('S0', 'S1')]
                if field != 'key':
                    vals0, vals1 = [np.array(v) for v in (vals0, vals1)]
                    vals0 = vals0.squeeze(1)
                meta_flat0[field], meta_flat1[field] = vals0, vals1

            meta_flat = (meta_flat0, meta_flat1)
        meta = meta_flat

    return meta
