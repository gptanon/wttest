# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Coefficient postprocessing tools."""
import numpy as np
import warnings
from copy import deepcopy

from ...utils.gen_utils import ExtendedUnifiedBackend
from .misc import tensor_padded


def normalize(X, mean_axis=(1, 2), std_axis=(1, 2), C=None, mu=1, C_mult=None):
    """Log-normalize + (optionally) standardize coefficients for learning
    algorithm suitability.

    Is a modification of Eq. 10 of https://arxiv.org/pdf/2007.10926.pdf
    For exact match (minus temporal global averaging), set
    `mean_axis=std_axis=(0, 2)`.

    Parameters
    ----------
    X : tensor
        Nonnegative tensor with dimensions `(samples, features, spatial)`.
        If there's more than one `features` or `spatial` dimensions, flatten
        before passing.
        (Obtain tensor via e.g. `pack_coeffs_jtfs(Scx)`, or `out_type='array'`.)

    std_axis : tuple[int] / int / None
        If not None, will unit-variance along specified axes, after
        log & mu norms.

    mean_axis : tuple[int] / int / None
        If not None, will zero-mean along specified axes, after
        log & mu norms.

    C : float / None
        `log(1 + X * C / median)`.
        Greater will bring more disparate values closer. Too great will equalize
        too much, too low will have minimal effect.

        Defaults to `5 / sparse_mean(abs(X / mu))`, which should yield moderate
        contraction for a variety of signals. This was computed on a mixture
        of random processes, with outliers, and may not generalize to all signals.

            - `sparse_mean` takes mean over non-negligible points, aiding
              consistency between representations. A scalogram with an extra
              octave, for example, may capture nothing in the new octave, while
              a simple mean would lower the output, attenuating existing values.

    mu : float / None
        In case precomputed; See "Online computation".

        `mu=None` will compute `mu` for per-channel normalization, while
        `mu=1` essentially disables `mu` and preserves channels' relative scaling;
        see "Relative scaling".

    C_mult : float / None
        Multiplies `C`. Useful if the default `C` compute scheme is appropriate
        but needs adjusting. Defaults to `5` if `C` is None, else to `1`.

    Returns
    -------
    Xnorm : tensor
        Normalized `X`.

    Relative scaling
    ----------------
    Scaling `features` independently changes the relative norms bewteen them.

      - If a signal rarely has high frequencies and low are dominant, for example,
        then post-normalization this nuance is lost and highs and lows are brought
        to a common norm - which may be undesired.
      - SNR is lowered, as low signal contents that are dominated by noise
        or float inaccuracies are amplified.
      - Convolutions over `features` dims are invalidated (as it's akin to
        standardizing individual time steps in 1D convolution); e.g. if
        normalizing on per-`n1` basis, then we can no longer do 2D convs
        over the joint `(n1, time)` pairs.
      - To keep convs valid, all spatial dims that are convolved over must be
        standardized by the same factor - i.e. same `mean` and `std`.
        For 2D convs along `(n1, t)`, this means `*_axis = (-1, -2)`.

    Despite the first two points, this "channel normalization" has been used with
    success for 1D convs in various settings.

    To preserve relative scaling, set `mu=1`.

    Online computation
    ------------------
    Any computation with `axis` that includes `0` requires simultaneous access
    to all samples. This poses a problem in two settings:

        1. Insufficient RAM. The solution is to write an *equivalent* computation
           that aggregates statistics one sample at a time. E.g. for `mu`:

           ::

               Xsum = []
               for x in dataset:
                   Xsum.append(B.sum(x, axis=-1, keepdims=True))
               mu = B.median(B.vstack(Xsum), axis=0, keepdims=True)

        2. Streaming / new samples. In this case we must reuse parameters computed
           over e.g. entire train set.

    Computations over all axes *except* `0` are done on per-sample basis, which
    means not having to rely on other samples - but also an inability to do so
    (i.e. to precompute and reuse params).
    """
    # validate args & set defaults ###########################################
    if X.ndim != 3:  # no-cov
        raise ValueError("input must be 3D, `(samples, features, spatial)` - "
                         "got %s" % str(X.shape))
    B = ExtendedUnifiedBackend(X)

    if B.backend_name == 'tensorflow' and mu is None:  # no-cov
        raise ValueError("mu=None with TensorFlow backend isn't supported, as "
                         "TF's `median` doesn't support axis args")

    # check input values
    if B.min(X) < 0:
        warnings.warn("`X` must be non-negative; will take modulus.")
        X = B.abs(X)
    # convert axes to positive
    axes = [mean_axis, std_axis]
    for i, ax in enumerate(axes):
        if ax is None:
            continue
        ax = ax if isinstance(ax, (list, tuple)) else [ax]
        ax = list(ax)
        for j, a in enumerate(ax):
            if a < 0:
                ax[j] = X.ndim + a
        axes[i] = tuple(ax)
    mean_axis, std_axis = axes

    # check input dims
    dim_ones = tuple(d for d in range(X.ndim) if X.shape[d] == 1)
    if dim_ones != ():
        def check_dims(g, name):
            g = g if isinstance(g, (tuple, list)) else (g,)
            if all(dim in dim_ones for dim in g):
                raise ValueError("input dims cannot be `1` along same dims as "
                                 "`{}` (gives NaNs); got X.shape == {}, "
                                 "{} = {}".format(name, X.shape, name, mean_axis))

        check_dims(mean_axis, 'mean_axis')
        check_dims(std_axis,  'std_axis')
        # check mu
        if mu is None and 0 in dim_ones and 2 in dim_ones:
            raise ValueError("input dims cannot be `1` along dims 0 and 2 "
                             "if `mu` is None (gives NaNs); "
                             "got X.shape == {}".format(X.shape))

    # main transform #########################################################
    if mu is None:
        # spatial sum (integral)
        Xsum = B.sum(X, axis=-1, keepdims=True)
        # sample median
        mu = B.median(Xsum, axis=0, keepdims=True)

    def sparse_mean(x, div=100, iters=4):
        """Mean of non-negligible points"""
        m = B.mean(x)
        for _ in range(iters - 1):
            m = B.mean(x[x > m / div])
        return m

    # rescale
    Xnorm = X / mu
    # contraction factor
    if C_mult is None:
        C_mult = 5 if C is None else 1
    if C is None:
        C = 1 / sparse_mean(B.abs(Xnorm), iters=4)
    C *= C_mult
    # log
    Xnorm = B.log(1 + Xnorm * C)

    # standardization ########################################################
    if mean_axis is not None:
        Xnorm -= B.mean(Xnorm, axis=mean_axis, keepdims=True)

    if std_axis is not None:
        Xnorm /= B.std(Xnorm, axis=std_axis, keepdims=True)

    return Xnorm


def pack_coeffs_jtfs(Scx, meta, structure=1, sample_idx=None,
                     separate_lowpass=None, sampling_psi_fr=None, out_3D=None,
                     reverse_n1=False, debug=False, recursive=False):
    """Packs efficiently JTFS coefficients into one of valid 4D structures.

    Parameters
    ----------
    Scx : tensor / list / dict
        JTFS output. Must have `out_type` `'dict:array'` or `'dict:list'`,
        and `average=True` or `oversampling=99`.

        Batch dimension must be an integer or not exist (see `sample_idx`).

    meta : dict
        JTFS meta.

    structure : int / None
        Structure to pack `Scx` into (see "Structures" below), integer 1 to 4.
        Will pack into a structure even if not suitable for convolution (as
        determined by JTFS parameters); see "Structures" if convs are relevant.

          - If can pack into one structure, can pack into any other (`1` to `5`).
          - `6` to `9` aren't implemented since they're what's already returned
            as output.
          - `structure=5` with `out_3D=True` and `aligned=True` is the only fully
            valid one for convolutions. This method is only needed for 3D or 4D
            convolution; 1D convs can be done on any JTFS with `average=True`,
            and 2D on any `out_3D=True`.

    sample_idx : int / None
        Index of sample in batched input to pack. If `None` (default), will
        pack all samples.
        Returns 5D if `None` *and* there's more than one sample.

    separate_lowpass : None / bool
        If True, will pack spinned (`psi_t * psi_f_up`, `psi_t * psi_f_dn`)
        and lowpass (`phi_t * phi_f`, `phi_t * psi_f`, `psi_t * phi_f`) pairs
        separately. Recommended for convolutions (see "Structures & Uniformity").

        Defaults to False if `structure != 5`. `structure = 5` requires True.

    sampling_psi_fr : str / None
        Used for sanity check for padding along `n1_fr`.
        Must match what was passed to `TimeFrequencyScattering1D`.
        If None, will assume library default.

    out_3D : bool / None
        Used for sanity check for padding along `n1`
        (enforces same number of `n1`s per `n2`).

    reverse_n1 : bool (default False)
        If True, will reverse ordering of `n1`. By default, low n1 <=> high freq
        (as directly output by `timefrequency_scattering1d`).

    debug : bool (defualt False)
        If True, coefficient values will be replaced by meta `n` values for
        debugging purposes, where the last dim is size 4 and contains
        `(n1_fr, n2, n1, time)` assuming `structure == 1`.

    recursive : bool (default False)
        Internal argument for handling `batch_size > 1`, do not use.

    Returns
    -------
    out: tensor / tuple[tensor]
        Packed `Scx`, depending on `structure` and `separate_lowpass`:

        ::

            - 1: `out` if False else
                 `(out, out_phi_f, out_phi_t)`
            - 2: same as 1
            - 3: `(out_up, out_dn, out_phi_f)` if False else
                 `(out_up, out_dn, out_phi_f, out_phi_t)`
            - 4: `(out_up, out_dn)` if False else
                 `(out_up, out_dn, out_phi_t)`
            - 5: `(out_up, out_dn, out_phi_f, out_phi_t, out_phi)`

        `out_phi_t` is `phi_t * psi_f` and `phi_t * phi_f` concatenated.
        `out_phi_f` is `psi_t * phi_f` for all configs except
        `3, True`, where it is concatenated with `phi_t * phi_f`.

        Low index <=> high frequency for all dims (unless `reverse_n1=True`).
        Low index <=> low time, for time dimension.
        For further info, see "Structures", "Parameter effects", and "Notes".

    Structures
    ----------
    Assuming `aligned=True`, then for `average, average_fr`, the following form
    valid convolution structures:

    ::

        1. `True, True*`:  3D/4D*, `(n1_fr, n2, n1, time)`
        2. `True, True*`:  2D/4D*, `(n2, n1_fr, n1, time)`
        3. `True, True*`:  4D,     `(n2, n1_fr//2,     n1, time)`*2,
                                   `(n2, 1, n1, time)`
        4. `True, True*`:  2D/4D*, `(n2, n1_fr//2 + 1, n1, time)`*2
        5. `True, True*`:  4D,     `(n2, n1_fr//2,     n1, time)`*2,
                                   `(n2, 1, n1, time)`,
                                   `(1, n1_fr, n1, time)`,
                                   `(1, 1, n1, time)`
        6. `True, True*`:  2D/3D*, `(n2 * n1_fr, n1, time)`
        7. `True, False`:  1D/2D*, `(n2 * n1_fr * n1, time)`
        8. `False, True`:  list of variable length 1D tensors
        9. `False, False`: list of variable length 1D tensors

    **Indexing/units**:

      - n1: frequency [Hz], first-order temporal variation
      - n2: frequency [Hz], second-order temporal variation
        (frequency of amplitude modulation)
      - n1_fr: quefrency [cycles/octave], first-order frequential variation
        (frequency of frequency modulation bands, roughly. More precisely,
        correlates with frequential bands (independent components/modes) of
        varying widths, decay factors, and recurrences, per temporal slice)
      - time: time [sec]
      - The actual units are discrete, "Hz" and "sec" are an example.
        To convert, multiply by sampling rate `fs`.
      - The `n`'s are indexings of the output array, also indexings of wavelets
        once accounting for stride and order reversal (n1_reverse).

          - E.g. `n1=2` may index `psi1_f[2*log2_F]` - or, generally,
            `psi1_f[2*total_conv_stride_over_U1_realized]` (see `core`).
          - With `aligned=False`, `n1` striding varies on per-`n2` basis.
            `n1` is the only "uncertain" index in this regard, and only `n1` and
            `t` are subject to stride; `n2` always means `psi2_f[n2]`, and
            `n1_fr` always means `psi1_f_fr_up[n1_fr]` (or down).
          - Hence, the frequency in "n2: frequency [Hz]" is obtained via
            `psi2_f[n2]['xi']`.
          - Higher n <=> higher center frequency. That is, coeffs are packed in
            order of decreasing frequency, just as in computation.
            Exceptions: 1) structure `1` or `2`, where spin down's `n1_fr` axis
            is reversed, and 2) if `n1_reverse=True`.

    **Convolution-validity**:

      - Structure 3 is 3D/4D-valid only if one deems valid the disjoint
        representation with separate convs over spinned and lowpassed
        (thus convs over lowpassed-only coeffs are deemed valid) - or if one
        opts to exclude the lowpassed pairs.
      - Structure 4 is 3D/4D-valid only if one deems valid convolving over both
        lowpassed and spinned coefficients.
      - Structure 5 is completely valid.
      - For convolutions, first dim is assumed to be channels (unless doing
        4D convs).
      - `True*` indicates a "soft requirement"; as long as `aligned=True`,
        `False` can be fully compensated with padding.
        Since 5 isn't implemented with `False`, it can be obtained from `False`
        by reshaping one of 1-4.
      - `2D/4D*` means 3D/4D convolutions aren't strictly valid for convolving
        over trailing (last) dimensions (see below), but 1D/2D are.
        `3D` means 1D, 2D, 3D are all valid.

    Structure interpretations for convolution
    -----------------------------------------
    Interpretations for convolution (and equivalently, spatial coherence)
    are as follows:

      1. The true JTFS structure. `(n2, n1, time)` are uniform and thus
         valid dimensions for 3D convolution (if all `phi` pairs are excluded,
         which isn't default behavior; see "Uniformity").
      2. It's a dim-permuted 1, but last three dimensions are no longer uniform
         and don't necessarily form a valid convolution pair.
         This is the preferred structure for conceptualizing or debugging as
         it's how the computation graph unfolds (and so does information
         density, as `N_fr` varies along `n2`).
      3. It's 2, but split into uniform pairs - `out_up, out_dn, out_phi`
         suited for convolving over last three dims. These still include
         `phi_t * psi_f` and `phi_t * phi_f` pairs, so for strict uniformity
         these slices should drop (e.g. `out_up[1:]`).
      4. It's 3, but only `out_up, out_dn`, and each includes `psi_t * phi_f`.
         If this "soft uniformity" is acceptable then `phi_t * psi_f` pairs
         should be kept.
      5. Completely valid convolutional structure.
         Every pair is packed separately. The only role of `pack_coeffs_jtfs`
         here is to reshape the pairs into 4D tensors, and pad.
      6. `n2` and `n1_fr` are flattened into one dimension. The resulting
         3D structure is suitable for 2D convolutions along `(n1, time)`.
      7. `n2`, `n1_fr`, and `n1` are flattened into one dimension. The resulting
         2D structure is suitable for 1D convolutions along `time`.
      8. `time` is variable; structue not suitable for convolution.
      9. `time` and `n1` are variable; structure not suitable for convolution.

    Structures not suited for convolutions may be suited for other transforms,
    e.g. Dense or Graph Neural Networks (or graph convolutions).

    Helpful visuals:

      https://wavespon.readthedocs.io/en/latest/extended/general_method_docs.html

    Uniformity
    ----------
    Coefficients are "uniform" if their generating wavelets are spaced uniformly
    (that is, equally incremented/spaced apart) in log space. The lowpass filter
    is equivalently an infinite scale wavelet, thus it breaks uniformity
    (it'd take infinite number of wavelets to be one increment away from lowpass).
    Opposite spins require stepping over the lowpass and are hence disqualified.

    Above is strictly true in continuous time. In a discrete setting, however,
    the largest possible non-dc scale is far from infinite. A 2D lowpass wavelet
    is somewhat interpretable as a subsequent scaling and rotation of the
    largest scale bandpass, as the bandpass itself is such a scaling and rotation
    of its preceding bandpass (emphasis on "somewhat", as this is wrong in
    important ways).

    Nonetheless, a lowpass is an averaging rather than modulation extracting
    filter: its physical units differ, and it has zero FDTS sensitivity - and
    this is a stronger objection for convolution. Further, when convolving over
    modulus of wavelet transform (as frequential scattering does), the dc bin
    is most often dominant, and by a lot - thus without proper renormalization
    it will drown out the bandpass coefficients in concatenation.

    The safest configuration for convolution thus excludes all lowpass pairs:
    `phi_t * phi_f`, `phi_t * psi_f`, and `psi_t * phi_f`; these can be convolved
    over separately. The bandpass and lowpass concatenations aren't recommended
    as anything but experimental.

    Parameter effects
    -----------------
    `average` and `average_fr` are described in "Structures". Additionally:

      - `aligned`:

        - `True`: enables the true JTFS structure (every structure in 1-7 is
          as described).
        - `False`: yields variable stride along `n1`, disqualifying it from
          3D convs along `(n2, n1, time)`. However, assuming semi-uniformity
          is acceptable, then each `n2` slice in `(n2, n1_fr, n1, time)`, i.e.
          `(n1_fr, n1, time)`, has the same stride, and forms valid conv pair
          (so use 3 or 4). Other structures require similar accounting.
          Rules out structure 1 for 3D/4D convs.

      - `out_3D`:

        - `True`: enforces same freq conv stride on *per-`n2`* basis, enabling
          3D convs even if `aligned=False`.

      - `sampling_psi_fr`:

        - `'resample'`: enables the true JTFS structure.
        - `'exclude'`: enables the true JTFS structure (it's simply a subset of
          `'resample'`). However, this involves large amounts of zero-padding to
          fill the missing convolutions and enable 4D concatenation.
        - `'recalibrate'`: breaks the true JTFS structure. `n1_fr` frequencies
          and widths now vary with `n2`, which isn't spatially coherent in 4D.
          It also renders `aligned=True` a pseudo-alignment.
          Like with `aligned=False`, alignment and coherence is preserved on
          per-`n2` basis, retaining the true structure in a piecewise manner.
          Rules out structure 1 for 3D/4D convs.

      - `average`:

        - It's possible to support `False` the same way `average_fr=False` is
          supported, but this isn't implemented.

    Notes
    -----
      1. Method requires `out_exclude=None` if `not separate_lowpass` - else,
         the following are allowed to be excluded: `'phi_t * psi_f'`,
         `'phi_t * phi_f'`, and if `structure != 4`, `'psi_t * phi_f'`.

      2. The built-in energy renormalization includes doubling the energy
         of `phi_t * psi_f` pairs to compensate for computing only once (for
         just one spin since it's identical to other spin), while here it may
         be packed twice (structure=`1` or `2`, or structure=`3` or `4` and
         `not separate_lowpass`); to compensate, its energy is halved before
         packing.

      3. Energy duplication isn't avoided for all configs:

          - `3, separate_lowpass`: packs the `phi_t * phi_f` pair twice -
            with `phi_t * psi_f`, and with `psi_t * phi_f`.
            `out_phi_f` always concats with `phi_t * phi_f` for `3` since
            `phi_f` is never concat with spinned, so it can't concat with
            `phi_t` pairs as usual.
          - `4, not separate_lowpass`: packs `phi_t * phi_f` and `psi_t * phi_f`
            pairs twice, once for each spin.
          - `4, separate_lowpass`: packs `psi_t * phi_f` pairs twice, once for
            each spin.
          - Note both `3` and `4` pack `phi_t * psi_f` pairs twice if
            `not separate_lowpass`, but the energy is halved anyway and hence
            not duped.

         This is intentional, as the idea is to treat each packing as an
         independent unit.
    """
    B = ExtendedUnifiedBackend(Scx)

    def combined_to_tensor(combined_all, recursive):
        def process_dims(o):
            if recursive:
                assert o.ndim == 5, o.shape
            else:
                assert o.ndim == 4, o.shape
                o = o[None]
            return o

        def not_none(x):
            return (x is not None if not recursive else
                    all(_x is not None for _x in x))

        # fetch combined params
        if structure in (1, 2):
            combined, combined_phi_t, combined_phi_f, combined_phi = combined_all
        else:
            (combined_up, combined_dn, combined_phi_t, combined_phi_f,
             combined_phi) = combined_all

        # compute pad params
        cbs = [(cb[0] if recursive else cb) for cb in combined_all
               if not_none(cb)]
        n_n1s_max = max(len(cb[n2_idx][n1_fr_idx])
                        for cb in cbs
                        for n2_idx in range(len(cb))
                        for n1_fr_idx in range(len(cb[n2_idx])))
        pad_value = 0 if not debug else -2
        # left pad along `n1` if `reverse_n1`
        left_pad_axis = (-2 if reverse_n1 else None)
        general = False  # use routine optimized for JTFS
        kw = dict(pad_value=pad_value, left_pad_axis=left_pad_axis,
                  general=general)

        # `phi`s #############################################################
        out_phi_t, out_phi_f, out_phi = None, None, None
        # ensure `phi`s and spinned pad to the same number of `n1`s
        ref_shape = ((None, None, n_n1s_max, None) if not recursive else
                     (None, None, None, n_n1s_max, None))
        # this will pad along `n1`
        if not_none(combined_phi_t):
            out_phi_t = tensor_padded(combined_phi_t, ref_shape=ref_shape, **kw)
            out_phi_t = process_dims(out_phi_t)
        if not_none(combined_phi_f):
            out_phi_f = tensor_padded(combined_phi_f, ref_shape=ref_shape, **kw)
            out_phi_f = process_dims(out_phi_f)
        if not_none(combined_phi):
            out_phi = tensor_padded(combined_phi, ref_shape=ref_shape, **kw)
            out_phi = process_dims(out_phi)

        # spinned ############################################################
        # don't need `ref_shape` here since by implementation max `n1`s
        # should be found in spinned (`phi`s are trimmed to ensure this)
        if structure in (1, 2):
            out = tensor_padded(combined, **kw)
            out = process_dims(out)

            if structure == 1:
                tp_shape = (0, 2, 1, 3, 4)
                out = B.transpose(out, tp_shape)
                if separate_lowpass:
                    if out_phi_t is not None:
                        out_phi_t = B.transpose(out_phi_t, tp_shape)
                    if out_phi_f is not None:
                        out_phi_f = B.transpose(out_phi_f, tp_shape)

            out = (out if not separate_lowpass else
                   (out, out_phi_f, out_phi_t))

        elif structure in (3, 4, 5):
            out_up = tensor_padded(combined_up, **kw)
            out_dn = tensor_padded(combined_dn, **kw)
            out_up = process_dims(out_up)
            out_dn = process_dims(out_dn)

            if structure == 3:
                out = ((out_up, out_dn, out_phi_f) if not separate_lowpass else
                       (out_up, out_dn, out_phi_f, out_phi_t))
            elif structure == 4:
                if not separate_lowpass:
                    out = (out_up, out_dn)
                else:
                    out = (out_up, out_dn, out_phi_t)
            elif structure == 5:
                out = (out_up, out_dn, out_phi_f, out_phi_t, out_phi)

        # sanity checks ##########################################################
        phis = dict(out_phi_t=out_phi_t, out_phi_f=out_phi_f, out_phi=out_phi)
        # take spinned as ref, which has every dim populated
        ref = out[0] if isinstance(out, tuple) else out

        for name, op in phis.items():
            if op is not None:
                errmsg = (name, op.shape, ref.shape)
                # `t`s must match
                assert op.shape[-1] == ref.shape[-1], errmsg
                # number of `n1`s must match
                assert op.shape[-2] == ref.shape[-2], errmsg
                # number of samples must match
                assert op.shape[0]  == ref.shape[0],  errmsg

                # due to transpose
                fr_dim = -3 if structure != 1 else -4
                if name in ('out_phi_f', 'out_phi'):
                    assert op.shape[fr_dim] == 1, op.shape
                    if name == 'out_phi':
                        # only for structure=5, which has `n2` at `shape[-4]`
                        assert op.shape[-4] == 1, op.shape
                    continue

                # phi_t only #################################################
                # compute `ref_fr_len`
                if structure in (1, 2, 5):
                    ref_fr_len = ref.shape[fr_dim]
                elif structure == 3:
                    # separate spins have half of total `n1_fr`s, but
                    # we also pack `phi_t` only once
                    ref_fr_len = ref.shape[fr_dim] * 1
                elif structure == 4:
                    # above + having `psi_t * phi_f`
                    # (i.e. fr_len_4 = fr_len_3 + 1)
                    ref_fr_len = (ref.shape[fr_dim] - 1) * 1
                if structure != 5 and separate_lowpass:
                    # due to `phi_t * phi_f` being present only in `out_phi_t`
                    ref_fr_len += 1

                # assert
                assert op.shape[fr_dim] == ref_fr_len, (
                    "{} != {} | {} | {}, {}".format(op.shape[fr_dim], ref_fr_len,
                                                    name, op.shape, ref.shape))
        if structure in (3, 4, 5):
            assert out_up.shape == out_dn.shape, (out_up.shape, out_dn.shape)

        if not recursive:
            # drop batch dim
            if isinstance(out, tuple):
                # the `None` is in case of `out_exclude`
                out = tuple((o[0] if o is not None else o) for o in out)
            else:
                out = out[0]
        return out

    # pack full batch recursively ############################################
    if not isinstance(Scx, dict):  # no-cov
        raise ValueError("must use `out_type` 'dict:array' or 'dict:list' "
                         "for `pack_coeffs_jtfs` (got `type(Scx) == %s`)" % (
                             type(Scx)))

    # infer batch size
    ref_pair = list(Scx)[0]
    if isinstance(Scx[ref_pair], list):
        n_samples = Scx[ref_pair][0]['coef'].shape[0]
    else:  # tensor
        n_samples = Scx[ref_pair].shape[0]
    n_samples = int(n_samples)

    # handle recursion, if applicable
    if n_samples > 1 and sample_idx is None:
        combined_phi_t_s, combined_phi_f_s, combined_phi_s = [], [], []
        if structure in (1, 2):
            combined_s = []
        elif structure in (3, 4, 5):
            combined_up_s, combined_dn_s = [], []

        for sample_idx in range(n_samples):
            combined_all = pack_coeffs_jtfs(Scx, meta, structure, sample_idx,
                                            separate_lowpass, sampling_psi_fr,
                                            debug, recursive=True)

            combined_phi_t_s.append(combined_all[-3])
            combined_phi_f_s.append(combined_all[-2])
            combined_phi_s.append(combined_all[-1])
            if structure in (1, 2):
                combined_s.append(combined_all[0])
            elif structure in (3, 4, 5):
                combined_up_s.append(combined_all[0])
                combined_dn_s.append(combined_all[1])

        phis = (combined_phi_t_s, combined_phi_f_s, combined_phi_s)
        if structure in (1, 2):
            combined_all_s = (combined_s, *phis)
        elif structure in (3, 4, 5):
            combined_all_s = (combined_up_s, combined_dn_s, *phis)
        out = combined_to_tensor(combined_all_s, recursive=True)
        return out

    ##########################################################################

    # validate `structure` / set default
    structures_available = {1, 2, 3, 4, 5}
    if structure is None:  # no-cov
        structure = structures_available[0]
    elif structure not in structures_available:  # no-cov
        raise ValueError(
            "invalid `structure={}`; Available are: {}".format(
                structure, ','.join(map(str, structures_available))))

    if separate_lowpass is None:
        separate_lowpass = False if structure != 5 else True
    elif separate_lowpass and structure == 5:  # no-cov
        raise ValueError("`structure=5` requires `separate_lowpass=False`.")

    # unpack coeffs for further processing
    Scx_unpacked = {}
    list_coeffs = isinstance(list(Scx.values())[0], list)
    if sample_idx is None and not recursive and n_samples == 1:
        sample_idx = 0

    Scx = drop_batch_dim_jtfs(Scx, sample_idx)
    t_ref = None
    for pair in Scx:
        is_joint = bool(pair not in ('S0', 'S1'))
        if not is_joint:
            continue
        Scx_unpacked[pair] = []
        for coef in Scx[pair]:
            if list_coeffs and (isinstance(coef, dict) and 'coef' in coef):
                coef = coef['coef']
            if t_ref is None:
                t_ref = coef.shape[-1]
            assert coef.shape[-1] == t_ref, (coef.shape, t_ref,
                                             "(if using average=False, set "
                                             "oversampling=99)")

            if coef.ndim == 2:
                Scx_unpacked[pair].extend(coef)
            elif coef.ndim == 1:
                Scx_unpacked[pair].append(coef)
            else:
                raise ValueError("expected `coef.ndim` of 1 or 2, got "
                                 "shape = %s" % str(coef.shape))

    # check that all necessary pairs are present
    pairs = ('psi_t * psi_f_up', 'psi_t * psi_f_dn', 'psi_t * phi_f',
             'phi_t * psi_f', 'phi_t * phi_f')
    # structure 4 requires `psi_t * phi_f`
    okay_to_exclude_if_sep_lp = (pairs[-3:] if structure != 4 else
                                 pairs[-2:])
    Scx_pairs = list(Scx)
    for p in pairs:
      if p not in Scx_pairs:
        if (not separate_lowpass or
            (separate_lowpass and p not in okay_to_exclude_if_sep_lp)):  # no-cov
          raise ValueError(("configuration requires pair '%s', which is "
                            "missing") % p)

    # for later; controls phi_t pair energy norm
    phi_t_packed_twice = bool((structure in (1, 2)) or
                              (structure in (3, 4) and not separate_lowpass))

    # pack into dictionary indexed by `n1_fr`, `n2` ##########################
    packed = {}
    ns = meta['n']
    n_n1_frs_max = 0
    for pair in pairs:
        if pair not in Scx_pairs:
            continue
        packed[pair] = []
        nsp = ns[pair].astype(int).reshape(-1, 3)

        idx = 0
        n2s_all = nsp[:, 0]
        n2s = np.unique(n2s_all)

        for n2 in n2s:
            n1_frs_all = nsp[n2s_all == n2, 1]
            packed[pair].append([])
            n1_frs = np.unique(n1_frs_all)
            n_n1_frs_max = max(n_n1_frs_max, len(n1_frs))

            for n1_fr in n1_frs:
                packed[pair][-1].append([])
                n1s_done = 0

                if out_3D:
                    # same number of `n1`s for all frequential slices *per-`n2`*
                    n_n1s = len(n1_frs_all)
                    n_n1s_in_n1_fr = n_n1s // len(n1_frs)
                    assert (n_n1s / len(n1_frs)
                            ).is_integer(), (n_n1s, len(n1_frs))
                else:
                    n_n1s_in_n1_fr = len(nsp[n2s_all == n2, 2
                                             ][n1_frs_all == n1_fr])
                if not debug:
                    while idx < len(nsp) and n1s_done < n_n1s_in_n1_fr:
                        try:
                            coef = Scx_unpacked[pair][idx]
                        except Exception as e:
                            print(pair, idx)
                            raise e
                        if pair == 'phi_t * psi_f' and phi_t_packed_twice:
                            # see "Notes" in docs
                            coef = coef / B.sqrt(2., dtype=coef.dtype)
                        packed[pair][-1][-1].append(coef)
                        idx += 1
                        n1s_done += 1
                else:
                    # pack meta instead of coeffs
                    n1s = nsp[n2s_all == n2, 2][n1_frs_all == n1_fr]
                    coef = [[n2, n1_fr, n1, 0] for n1 in n1s]
                    # ensure coef.shape[-1] == t
                    while Scx_unpacked[pair][0].shape[-1] > len(coef[0]):
                        for i in range(len(coef)):
                            coef[i].append(0)
                    coef = np.array(coef)

                    packed[pair][-1][-1].extend(coef)
                    assert len(coef) == n_n1s_in_n1_fr
                    idx += len(coef)
                    n1s_done += len(coef)

    # pad along `n1_fr`
    if sampling_psi_fr is None:
        sampling_psi_fr = 'exclude'
    pad_value = 0 if not debug else -2
    for pair in packed:
        if 'psi_f' not in pair:
            continue
        for n2_idx in range(len(packed[pair])):
            if len(packed[pair][n2_idx]) < n_n1_frs_max:
                assert sampling_psi_fr == 'exclude'  # should not occur otherwise
            else:
                continue

            # make a copy to avoid modifying `packed`
            ref = list(tensor_padded(packed[pair][n2_idx][0]))
            # assumes last dim is same (`average=False`)
            # and is 2D, `(n1, t)` (should always be true)
            for i in range(len(ref)):
                if debug:
                    # n2 will be same, everything else variable
                    ref[i][1:] = ref[i][1:] * 0 + pad_value
                else:
                    ref[i] = ref[i] * 0

            while len(packed[pair][n2_idx]) < n_n1_frs_max:
                packed[pair][n2_idx].append(list(ref))

    # pack into list ready to convert to 4D tensor ###########################
    # current indexing: `(n2, n1_fr, n1, time)`
    # c = combined
    c_up    = packed['psi_t * psi_f_up']
    c_dn    = packed['psi_t * psi_f_dn']
    c_phi_t = packed['phi_t * psi_f'] if 'phi_t * psi_f' in Scx_pairs else None
    c_phi_f = packed['psi_t * phi_f'] if 'psi_t * phi_f' in Scx_pairs else None
    c_phi   = packed['phi_t * phi_f'] if 'phi_t * phi_f' in Scx_pairs else None

    can_make_c_phi_t = bool(c_phi_t is not None and c_phi is not None)

    # `deepcopy` below is to ensure same structure packed repeatedly in different
    # places isn't modified in both places when it's modified in one place.
    # `None` set to variables means they won't be tensored and returned.

    if structure in (1, 2):
        # structure=2 is just structure=1 transposed, so pack them same
        # and transpose later.
        # instantiate total combined
        combined = c_up
        c_up = None

        # append phi_f ####
        if not separate_lowpass:
            for n2 in range(len(c_phi_f)):
                for n1_fr in range(len(c_phi_f[n2])):
                    c = c_phi_f[n2][n1_fr]
                    combined[n2].append(c)
            c_phi_f = None
            # assert that appending phi_f only increased dim1 by 1
            l0, l1 = len(combined[0]), len(c_dn[0])
            assert l0 == l1 + 1, (l0, l1)

        # append down ####
        # assert that so far dim0 hasn't changed
        assert len(combined) == len(c_dn), (len(combined), len(c_dn))

        # dn: reverse `psi_f` ordering
        for n2 in range(len(c_dn)):
            c_dn[n2] = c_dn[n2][::-1]

        for n2 in range(len(combined)):
            combined[n2].extend(c_dn[n2])
        c_dn = None

        # pack phi_t ####
        if not separate_lowpass or can_make_c_phi_t:
            c_phi_t = deepcopy(c_phi_t)
            c_phi_t[0].append(c_phi[0][0])
            # phi_t: reverse `psi_f` ordering
            c_phi_t[0].extend(packed['phi_t * psi_f'][0][::-1])
            c_phi = None

        # append phi_t ####
        if not separate_lowpass:
            combined.append(c_phi_t[0])
            c_phi_t = None

    elif structure == 3:
        # pack spinned ####
        if not separate_lowpass:
            c_up.append(c_phi_t[0])
            c_dn.append(deepcopy(c_phi_t[0]))
            c_phi_t = None

        # pack phi_t ####
        if separate_lowpass and can_make_c_phi_t:
            c_phi_t[0].append(deepcopy(c_phi[0][0]))

        # pack phi_f ####
        # structure=3 won't pack `phi_f` with `psi_f`, so can't pack
        # `phi_t * phi_f` along `phi_t * psi_f` (unless `separate_lowpass=True`
        # where `phi_t` isn't packed with `psi_f`), must pack with `psi_t * phi_f`
        # instead
        if not separate_lowpass or (c_phi_f is not None and c_phi is not None):
            c_phi_f.append(c_phi[0])
            c_phi = None

    elif structure == 4:
        # pack phi_f ####
        for n2 in range(len(c_phi_f)):
            # structure=4 joins `psi_t * phi_f` with spinned
            c = c_phi_f[n2][0]
            c_up[n2].append(c)
            c_dn[n2].append(c)
        c_phi_f = None
        # assert up == dn along dim1
        l0, l1 = len(c_up[0]), len(c_dn[0])
        assert l0 == l1, (l0, l1)

        # pack phi_t ####
        if separate_lowpass and can_make_c_phi_t:
            # pack `phi_t * phi_f` with `phi_t * psi_f`
            c_phi_t[0].append(c_phi[0][0])
        elif not separate_lowpass:
            # pack `phi_t * phi_f` with `phi_t * psi_f`, packed with each spin
            # phi_t, append `n1_fr` slices via `n2`
            c_up.append(deepcopy(c_phi_t[0]))
            c_dn.append(c_phi_t[0])
            # phi, append one `n2, n1_fr` slice
            c_up[-1].append(deepcopy(c_phi[0][0]))
            c_dn[-1].append(c_phi[0][0])

            c_phi_t, c_phi = None, None
        # assert up == dn along dim0, and dim1
        assert len(c_up) == len(c_dn), (len(c_up), len(c_dn))
        l0, l1 = len(c_up[0]), len(c_dn[0])
        assert l0 == l1, (l0, l1)

    elif structure == 5:
        pass  # all packed

    # reverse ordering of `n1` ###############################################
    if reverse_n1:
        # pack all into `cbs`
        if c_up is not None:
            cbs = [c_up, c_dn]
        else:
            cbs = [combined]
        if c_phi_t is not None:
            cbs.append(c_phi_t)
        if c_phi_f is not None:
            cbs.append(c_phi_f)
        if c_phi is not None:
            cbs.append(c_phi)

        # reverse `n1`
        cbs_new = []
        for i, cb in enumerate(cbs):
            cbs_new.append([])
            for n2_idx in range(len(cb)):
                cbs_new[i].append([])
                for n1_fr_idx in range(len(cb[n2_idx])):
                    cbs_new[i][n2_idx].append(cb[n2_idx][n1_fr_idx][::-1])

        # unpack all from `cbs`
        if c_up is not None:
            c_up = cbs_new.pop(0)
            c_dn = cbs_new.pop(0)
        else:
            combined = cbs_new.pop(0)
        if c_phi_t is not None:
            c_phi_t = cbs_new.pop(0)
        if c_phi_f is not None:
            c_phi_f = cbs_new.pop(0)
        if c_phi is not None:
            c_phi = cbs_new.pop(0)
        assert len(cbs_new) == 0, len(cbs_new)

    # finalize ###############################################################
    phis = (c_phi_t, c_phi_f, c_phi)
    combined_all = ((combined, *phis) if c_up is None else
                    (c_up, c_dn, *phis))
    if recursive:
        return combined_all
    return combined_to_tensor(combined_all, recursive=False)


# convenience reusables / helpers ############################################
def drop_batch_dim_jtfs(Scx, sample_idx=0):
    """Index into dim0 with `sample_idx` for every JTFS coefficient, and
    drop that dimension.

    Doesn't modify input:

        - dict/list: new list/dict (with copied meta if applicable)
        - array: new object but shared storage with original array (so original
          variable reference points to unindexed array).
    """
    fn = lambda x: x[sample_idx]
    return _iterate_apply(Scx, fn)


def jtfs_to_numpy(Scx):
    """Convert PyTorch/TensorFlow tensors to numpy arrays, with meta copied,
    and without affecting original data structures.
    """
    B = ExtendedUnifiedBackend(Scx)
    return _iterate_apply(Scx, B.numpy)


def _iterate_apply(Scx, fn):
    def get_meta(s):
        return {k: v for k, v in s.items() if not hasattr(v, 'ndim')}

    if isinstance(Scx, dict):
        out = {}  # don't modify source dict
        for pair in Scx:
            if isinstance(Scx[pair], list):
                out[pair] = []
                for i, s in enumerate(Scx[pair]):
                    out[pair].append(get_meta(s))
                    out[pair][i]['coef'] = fn(s['coef'])
            else:
                out[pair] = fn(Scx[pair])
    elif isinstance(Scx, list):
        out = []  # don't modify source list
        for s in Scx:
            o = get_meta(s)
            o['coef'] = fn(s['coef'])
            out.append(o)
    elif isinstance(Scx, tuple):  # out_type=='array' && out_3D==True
        out = (fn(Scx[0]), fn(Scx[1]))
    elif hasattr(Scx, 'ndim'):
        out = fn(Scx)
    else:  # no-cov
        raise ValueError(("unrecognized input type: {}; must be as returned by "
                          "`jtfs(x)`.").format(type(Scx)))
    return out
