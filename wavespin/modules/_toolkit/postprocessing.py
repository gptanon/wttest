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

from ...utils.gen_utils import ExtendedUnifiedBackend


def normalize(X, mean_axis=(1, 2), std_axis=(1, 2), C=None, mu=1, C_mult=None,
              log=True):
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

    log : bool (default True)
        Whether to apply the log.
        `False` makes this a simple `(x - mean) / std` convenience function,
        and makes `C`, `mu`, `C_mult` have no effect.

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

    Mathematical observation on `C`
    -------------------------------
    Larger `C` drives `log(1 + C*x)` closer to `log(x)` in shape.

    Details: `C -> inf` -> `log(1 + C*x) -> log(x) - const.` when mean-normalized.
    Note,

        log((1 + C*x)/C) = log(1 + C*x) - log(C)
        <=>
        log(1/C + x) + log(C) = log(1 + C*x)

    With zero-mean, the `+ log(C)` drops, and as `C` grows, we get `log(~0 + x)`.
    Note, we don't approach `log(x)`, but rather `log(x) - mean(log(x))`,
    since the mean of `log(1/C + x) + log(C)` isn't `log(C)`.

    Demo (very small chance of failing; it's of course not a true equality):

    ::

        import numpy as np

        x = np.abs(np.random.randn(100))
        a = np.log(x)
        b = np.log1p(9999999999999*x)
        assert np.allclose(a - a.mean(), b - b.mean())
        # thus, `a` and `b` match within an offset

        # `C` need not be huge to get a good approximation, but it still
        # needs to be much larger than what we'll use in practice:
        b = np.log1p(99*x)
        assert np.linalg.norm( (a-a.mean()) - (b-b.mean())
                              ) / np.linalg.norm( (a-a.mean()) ) < .1

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
        def check_dims(ax, name):
            ax = ax if isinstance(ax, (tuple, list)) else (ax,)
            if all(dim in dim_ones for dim in ax):
                raise ValueError("input dims cannot be `1` along same dims as "
                                 "`{}` (gives NaNs); got X.shape == {}, "
                                 "{} = {}".format(name, X.shape, name, ax))

        check_dims(mean_axis, 'mean_axis')
        check_dims(std_axis,  'std_axis')
        # check mu
        if mu is None and 0 in dim_ones and 2 in dim_ones:
            raise ValueError("input dims cannot be `1` along dims 0 and 2 "
                             "if `mu` is None (gives NaNs); "
                             "got X.shape == {}".format(X.shape))

    # main transform #########################################################
    if log:
        if mu is None:
            # spatial sum (integral)
            Xsum = B.sum(X, axis=-1, keepdims=True)
            # sample median
            mu = B.median(Xsum, axis=0, keepdims=True)

        # rescale
        Xnorm = X / mu
        # contraction factor
        if C_mult is None:
            C_mult = 5 if C is None else 1
        if C is None:
            C = 1 / sparse_mean(B.abs(Xnorm), iters=4, B=B)
        C *= C_mult
        # log
        Xnorm = B.log1p(Xnorm * C)
    else:
        Xnorm = X

    # standardization ########################################################
    if mean_axis is not None:
        Xnorm -= B.mean(Xnorm, axis=mean_axis, keepdims=True)

    if std_axis is not None:
        Xnorm /= B.std(Xnorm, axis=std_axis, keepdims=True)

    return Xnorm


def sparse_mean(x, div=100, iters=4, B=None):
    """Mean of non-negligible points.
    `normalize` helper, but also useful standalone.
    """
    if B is None:
        B = np
    m = B.mean(x)
    for _ in range(iters - 1):
        m = B.mean(x[x > m / div])
    return m


def pack_coeffs_jtfs(Scx, meta, structure=1, separate_lowpass=None,
                     as_numpy=False, sampling_psi_fr=None, out_3D=None,
                     did_energy_correction=True, reverse_n1=False, debug=False):
    """Packs efficiently JTFS coefficients into one of valid 4D structures.

    If using with non-numpy, see `as_numpy` parameter.

    Parameters
    ----------
    Scx : tensor / list / dict
        JTFS output. Must have `out_type` `'dict:array'` or `'dict:list'`,
        and `average=True` or `oversampling=99`.

        Batch dimension must be an integer, and must be present.

    meta : dict
        JTFS meta.

    structure : int / None
        Structure to pack `Scx` into (see "Structures" below), integer 1 to 5.
        Will pack into a structure even if not suitable for convolution (as
        determined by JTFS parameters); see "Structures" if convs are relevant.

          - If can pack into one structure, can pack into any other (`1` to `5`).
          - `6` to `9` aren't implemented since they're what's already returned
            as output.
          - `structure=5` with `out_3D=True` and `aligned=True` is the only fully
            valid one for convolutions. This method is only needed for 3D or 4D
            convolution; 1D convs can be done on any JTFS with `average=True`,
            and 2D on any `out_3D=True`.

    separate_lowpass : None / bool
        If True, will pack spinned (`psi_t * psi_f_up`, `psi_t * psi_f_dn`)
        and lowpass (`phi_t * phi_f`, `phi_t * psi_f`, `psi_t * phi_f`) pairs
        separately. Recommended for convolutions (see "Structures & Uniformity").

        Defaults to False if `structure != 5`. `structure = 5` requires True.

    as_numpy : bool (default False)
        TL;DR with non-numpy input, `True` may be faster, but output's numpy.
        GPU + `out_type='dict:array'` *will* be faster, but output's numpy.

        For non-numpy backend, especially with GPU, it may be faster to use
        `wavespin.toolkit.jtfs_to_numpy(, nometa=True)` before this function,
        even if output is to remain on GPU, and especially if the GPU backend
        isn't PyTorch. It's certainly true if output is to be kept on CPU as numpy
        (e.g. for storing on disk). `as_numpy=True` makes this call.

    sampling_psi_fr : str / None
        Used for sanity check for padding along `n1_fr`.
        Must match what was passed to `TimeFrequencyScattering1D`.
        If None, will assume library default.

    out_3D : bool / None
        Used for sanity check for padding along `n1`
        (enforces same number of `n1`s per `n2`).

    did_energy_correction : bool (default True)
        Should equal `jtfs.do_energy_correction`. Defaults to `True`.
        Determines whether, if `phi_t * psi_f` pairs are packed twice, their
        energy is halved.

    reverse_n1 : bool (default False)
        If True, will reverse ordering of `n1`. By default, low n1 <=> high freq
        (as directly output by `timefrequency_scattering1d`).

    debug : bool (default False)
        If True, coefficient values will be replaced by meta `n` values for
        debugging purposes, where the last dim is size 4 and contains
        `(n1_fr, n2, n1, time)` assuming `structure == 1`.

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
        `3`, where it is concatenated with `phi_t * phi_f`.

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
                           # TODO should be `4D*`?
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
         packing. See `did_energy_correction`.

      3. Energy duplication isn't avoided for all configs:

          - `3, separate_lowpass`: packs the `phi_t * phi_f` pair twice -
            with `phi_t * psi_f`, and with `psi_t * phi_f`.
            `out_phi_f` always concats with `phi_t * phi_f` for `3` since
            `phi_f` is never concat with spinned, so it can't concat with
            `phi_t` pairs as usual.
            (In retrospect, it need not be packed twice, but perhaps I did it
            for self-consistency.)

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
    if as_numpy:
        Scx = jtfs_to_numpy(Scx, nometa=True)
    B = ExtendedUnifiedBackend(Scx)

    def combined_to_tensor(combined_all):
        # fetch combined params
        if structure in (1, 2):
            combined, combined_phi_t, combined_phi_f, combined_phi = combined_all
        else:
            (combined_up, combined_dn, combined_phi_t, combined_phi_f,
             combined_phi) = combined_all

        # note, currently everywhere batch dim is -2
        # to undo it, we'd do `transpose(-2, *list(range(0, ndim-2)), -1)`.
        batch_swap_dims = (3, 0, 1, 2, 4)

        # `phi`s #############################################################
        out_phi_t, out_phi_f, out_phi = None, None, None
        if combined_phi_t is not None:
            out_phi_t = B.as_tensor(combined_phi_t)
        if combined_phi_f is not None:
            out_phi_f = B.as_tensor(combined_phi_f)
        if combined_phi is not None:
            out_phi = B.as_tensor(combined_phi)

        # spinned ############################################################
        if structure in (1, 2):
            out = B.as_tensor(combined)

            if structure == 1:
                # here we'd first `.transpose(3, 0, 1, 2, 4)` (`batch_swap_dims`),
                # then use `tp_dims = (0, 2, 1, 3, 4)`. The combined operation is
                # latter upon former.
                tp_dims = (3, 1, 0, 2, 4)
            else:
                tp_dims = batch_swap_dims

            out = B.transpose(out, tp_dims)
            if separate_lowpass:
                if out_phi_t is not None:
                    out_phi_t = B.transpose(out_phi_t, tp_dims)
                if out_phi_f is not None:
                    out_phi_f = B.transpose(out_phi_f, tp_dims)

            out = (out if not separate_lowpass else
                   (out, out_phi_f, out_phi_t))

        elif structure in (3, 4, 5):
            out_up = B.transpose(B.as_tensor(combined_up), batch_swap_dims)
            out_dn = B.transpose(B.as_tensor(combined_dn), batch_swap_dims)

            if out_phi_t is not None:
                out_phi_t = B.transpose(out_phi_t, batch_swap_dims)
            if out_phi_f is not None:
                out_phi_f = B.transpose(out_phi_f, batch_swap_dims)
            if out_phi is not None:
                out_phi = B.transpose(out_phi, batch_swap_dims)

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

        # sanity checks ######################################################
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
        return out

    # pack full batch recursively ############################################
    if not isinstance(Scx, dict):  # no-cov
        raise ValueError("must use `out_type` 'dict:array' or 'dict:list' "
                         "for `pack_coeffs_jtfs` (got `type(Scx) == %s`)" % (
                             type(Scx)))

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
    elif not separate_lowpass and structure == 5:  # no-cov
        raise ValueError("`structure=5` requires `separate_lowpass=True`.")

    # unpack coeffs for further processing -----------------------------------
    Scx_unpacked = {}
    # infer `out_type`; fetch reference time length
    c_ref = list(Scx.values())[0]
    list_coeffs = bool(isinstance(c_ref, list))
    if list_coeffs:
        t_ref = c_ref[0]['coef'].shape[-1]
    else:
        t_ref = c_ref.shape[-1]

    # unpack joint pairs
    for pair in Scx:
        if pair in ('S0', 'S1'):  # joint only
            continue
        Scx_pair = Scx[pair]

        # make coeffs Python-iterable
        if list_coeffs:
            coeffs = [c['coef'] for c in Scx_pair]
        else:
            coeffs = Scx_pair.swapaxes(0, 1)
        # ensure batch dim present (not foul-proof method)
        ndim = coeffs[0].ndim
        assert ndim != 1, (
            'must have batch dimension, got coef.shape=%s' % coeffs[0].shape)

        unpacked_pair = []
        for coef in coeffs:
            assert coef.shape[-1] == t_ref, (
                coef.shape, t_ref,
                "(if using average=False, set oversampling=99)")

            if coef.ndim == 3:
                # [batch_size, n1, t] -> [n1, batch_size, t]
                unpacked_pair.extend(coef.swapaxes(0, 1))
            elif coef.ndim == 2:
                unpacked_pair.append(coef)
            else:
                raise ValueError("expected `coef.ndim` of 1 or 2, got "
                                 "shape = %s" % str(coef.shape))
        Scx_unpacked[pair] = unpacked_pair

    # validation, reusable params --------------------------------------------
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

    # for later: controls phi_t pair energy norm
    phi_t_packed_twice = bool((structure in (1, 2)) or
                              (structure in (3, 4) and not separate_lowpass))

    # for later: fetch sqrt2, n_t
    coef_ref = Scx_unpacked[list(Scx_unpacked)[0]][0]
    n_t = coef_ref.shape[-1]
    if not debug:
        sqrt2 = B.sqrt(2., dtype=coef_ref.dtype)
    else:
        czeros = [0] * max(n_t - 3, 1)

    # Below, a `recipe = dict()` option was explored to avoid all the loop
    # recomputations, but its speedup relative to total compute, i.e. including
    # `jtfs(x)`, was found to be insignificant.

    # pack into dictionary indexed by `n1_fr`, `n2` ##########################
    packed = {}
    ns = meta['n']
    n_n1_frs_max = 0
    n_n1s_max = 0
    for pair in pairs:
        # `pair` loop --------------------------------------------------------
        if pair not in Scx_pairs:
            continue
        Scx_unpacked_pair = Scx_unpacked[pair]
        do_energy_halving = (
            pair == 'phi_t * psi_f' and phi_t_packed_twice and
            did_energy_correction)

        with np.errstate(invalid='ignore'):
            nsp = ns[pair].astype(int).reshape(-1, 3)
        idx = 0
        n2s_all = nsp[:, 0]
        n2s = np.unique(n2s_all)

        packed_pair = []
        for n2 in n2s:
            # `n2` loop ------------------------------------------------------
            n2s_all_eq_n2 = (n2s_all == n2)
            n1_frs_all = nsp[n2s_all_eq_n2, 1]
            n1_frs = np.unique(n1_frs_all)
            n_n1_frs = len(n1_frs)
            n_n1_frs_max = max(n_n1_frs_max, n_n1_frs)

            if out_3D:
                # same number of `n1`s for all frequential slices *per-`n2`*
                n_n1s = len(n1_frs_all)
                n_n1s_in_n1_fr = n_n1s // n_n1_frs
                assert (n_n1s / n_n1_frs
                        ).is_integer(), (n_n1s, n_n1_frs)

            packed_n2 = []
            for n1_fr in n1_frs:
                # `n1_fr` loop -----------------------------------------------
                if not out_3D:
                    n_n1s_in_n1_fr = np.sum(n1_frs_all == n1_fr)
                n_n1s_max = max(n_n1s_max, n_n1s_in_n1_fr)

                # `n1` loops -------------------------------------------------
                packed_n1_fr = []
                if not debug:
                    for n1_idx in range(n_n1s_in_n1_fr):
                        coef = Scx_unpacked_pair[idx]
                        if do_energy_halving:
                            # see "Notes" in docs
                            coef = coef / sqrt2
                        packed_n1_fr.append(coef)
                        idx += 1
                else:
                    # pack meta instead of coeffs
                    n1s = nsp[n2s_all_eq_n2, 2][n1_frs_all == n1_fr]
                    # ensure `coeff.shape[-1] == n_t` (or more ...
                    # don't recall if needed)
                    coef = np.array([[[n2, n1_fr, n1] + czeros for n1 in n1s]])

                    packed_n1_fr.extend(coef.swapaxes(0, 1))
                    assert coef.shape[1] == n_n1s_in_n1_fr
                    idx += coef.shape[1]

                # pack into n2
                packed_n2.append(packed_n1_fr)
            # pack into pair
            packed_pair.append(packed_n2)
        # pack into all
        packed[pair] = packed_pair

    # pad along `n1_fr`, `n1` ------------------------------------------------
    if sampling_psi_fr is None:
        sampling_psi_fr = 'exclude'
    pad_value = 0 if not debug else -2
    for pair in packed:
        for n2_idx in range(len(packed[pair])):
            packed_n2 = packed[pair][n2_idx]

            # pad along `n1` -------------------------------------------------
            for n1_fr_idx in range(len(packed_n2)):
                packed_n1_fr = packed_n2[n1_fr_idx]
                if len(packed_n1_fr) < n_n1s_max:
                    ref = B.as_tensor(packed_n1_fr[0])
                    if debug:
                        ref = ref.copy()
                        # n2 will be same, everything else variable
                        ref[:, 1:] = ref[:, 1:] * 0 + pad_value
                    else:
                        ref = ref * 0
                    n_current = len(packed[pair][n2_idx][n1_fr_idx])
                    to_append = [ref] * (n_n1s_max - n_current)
                    packed[pair][n2_idx][n1_fr_idx].extend(to_append)

            # pad along `n1_fr` ----------------------------------------------
            if 'psi_f' not in pair:
                continue

            n_current = len(packed[pair][n2_idx])
            if n_current < n_n1_frs_max:
                assert sampling_psi_fr == 'exclude'  # should not occur otherwise
            else:
                continue

            # make a copy to avoid modifying `packed`
            ref = list(B.as_tensor(packed[pair][n2_idx][0]))
            # assumes last dim is same (`average=True`)
            # and is 2D, `(n1, t)` (should always be true)
            if debug:
                ref = ref.copy()
            for i in range(len(ref)):
                if debug:
                    # n2 will be same, everything else variable
                    ref[i][:, 1:] = ref[i][:, 1:] * 0 + pad_value
                else:
                    ref[i] = ref[i] * 0

            to_append = [list(ref)] * (n_n1_frs_max - n_current)
            packed[pair][n2_idx].extend(to_append)

    # pack into list ready to convert to 4D tensor ###########################
    # current indexing: `(n2, n1_fr, n1, time)`
    # c_* == combined_*

    c_up    = packed['psi_t * psi_f_up']
    c_dn    = packed['psi_t * psi_f_dn']
    c_phi_t = packed['phi_t * psi_f'] if 'phi_t * psi_f' in Scx_pairs else None
    c_phi_f = packed['psi_t * phi_f'] if 'psi_t * phi_f' in Scx_pairs else None
    c_phi   = packed['phi_t * phi_f'] if 'phi_t * phi_f' in Scx_pairs else None

    can_make_c_phi_t = bool(c_phi_t is not None and c_phi is not None)

    # `deepcopy` below was formerly used to ensure same structure packed
    # repeatedly in different places isn't modified in both places when it's
    # modified in one place. The logic was rewritten to attain equivalent
    # behavior, with old code shown alongside for clarity.
    # `None` set to variables means they won't be tensored and returned.

    if structure in (1, 2):
        # structure=2 is just structure=1 transposed, so pack them same
        # and transpose later.
        # instantiate total combined
        combined = c_up
        c_up = None

        # append phi_f ####
        if not separate_lowpass:
            for n2_idx in range(len(c_phi_f)):
                for n1_fr in range(len(c_phi_f[n2_idx])):
                    c = c_phi_f[n2_idx][n1_fr]
                    combined[n2_idx].append(c)
            c_phi_f = None
            # assert that appending phi_f only increased dim1 by 1
            l0, l1 = len(combined[0]), len(c_dn[0])
            assert l0 == l1 + 1, (l0, l1)

        # append down ####
        # assert that so far dim0 hasn't changed
        assert len(combined) == len(c_dn), (len(combined), len(c_dn))

        # dn: reverse `psi_f` ordering
        for n2_idx in range(len(c_dn)):
            c_dn[n2_idx] = c_dn[n2_idx][::-1]

        for n2_idx in range(len(combined)):
            combined[n2_idx].extend(c_dn[n2_idx])
        c_dn = None

        # pack phi_t ####
        if not separate_lowpass or can_make_c_phi_t:
            # doing this ahead of time avoids mutating it via `c_phi_t` append
            packed_slc = packed['phi_t * psi_f'][0][::-1]
            c_phi_t[0].append(c_phi[0][0])
            # phi_t: reverse `psi_f` ordering
            c_phi_t[0].extend(packed_slc)
            c_phi = None
            # old logic:
            #     c_phi_t = deepcopy(c_phi_t)
            #     c_phi_t[0].append(c_phi[0][0])
            #     # phi_t: reverse `psi_f` ordering
            #     c_phi_t[0].extend(packed['phi_t * psi_f'][0][::-1])

        # append phi_t ####
        if not separate_lowpass:
            combined.append(c_phi_t[0])
            c_phi_t = None

    elif structure == 3:
        # pack spinned ####
        if not separate_lowpass:
            c_up.append(c_phi_t[0])
            c_dn.append(c_phi_t[0])
            c_phi_t = None
            # old logic:
            #     c_up.append(c_phi_t[0])
            #     c_dn.append(dc(c_phi_t[0]))
            #     c_phi_t = None

        # pack phi_t ####
        if separate_lowpass and can_make_c_phi_t:
            c_phi_t[0].append(c_phi[0][0])
            # old logic:
            #     c_phi_t[0].append(dc(c_phi[0][0]))

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
        for n2_idx in range(len(c_phi_f)):
            # structure=4 joins `psi_t * phi_f` with spinned
            c = c_phi_f[n2_idx][0]
            c_up[n2_idx].append(c)
            c_dn[n2_idx].append(c)
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
            c_phi_t[0].append(c_phi[0][0])
            c_up.append(c_phi_t[0])
            c_dn.append(c_phi_t[0])
            c_phi_t, c_phi = None, None
            # old logic:
            #     c_up.append(dc(c_phi_t[0]))
            #     c_dn.append(c_phi_t[0])
            #     # phi, append one `n2, n1_fr` slice
            #     c_up[-1].append(dc(c_phi[0][0]))
            #     c_dn[-1].append(c_phi[0][0])
            #     c_phi_t, c_phi = None, None

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
    return combined_to_tensor(combined_all)


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


def jtfs_to_numpy(Scx, nometa=False):
    """Convert PyTorch/TensorFlow tensors to numpy arrays, with meta copied
    (unless `nometa=True`), and without affecting original data structures.
    """
    B = ExtendedUnifiedBackend(Scx)
    return _iterate_apply(Scx, B.numpy, nometa=nometa)


def _iterate_apply(Scx, fn, nometa=False):
    def get_meta(s):
        if nometa:
            return {}
        return {k: v for k, v in s.items() if k != 'coef'}  # TODO was ndim check

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
