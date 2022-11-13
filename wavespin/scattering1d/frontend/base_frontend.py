# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ...frontend.base_frontend import ScatteringBase
import math
import numbers
import warnings
from types import FunctionType
from copy import deepcopy

import numpy as np

from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering1d import timefrequency_scattering1d
from ..filter_bank import (scattering_filter_factory, fold_filter_fourier,
                           N_and_pad_2_J_pad, n2_n1_cond)
from ..refining import energy_norm_filterbank_tm
from ..filter_bank_jtfs import _FrequencyScatteringBase1D
from ..scat_utils import (
    compute_border_indices, compute_padding, compute_minimum_support_to_pad,
    compute_meta_scattering, compute_meta_jtfs,
    _handle_paths_exclude, _handle_smart_paths, _handle_input_and_backend,
    _check_runtime_args_common, _check_runtime_args_scat1d,
    _check_runtime_args_jtfs, _restore_batch_shape)
from ...utils.gen_utils import fill_default_args
from ...toolkit import pack_coeffs_jtfs, scattering_info
from ... import CFG


class ScatteringBase1D(ScatteringBase):
    SUPPORTED_KWARGS = {
        'normalize', 'r_psi', 'max_pad_factor',
        'analytic', 'paths_exclude', 'precision',
    }
    DEFAULT_KWARGS = dict(
        normalize='l1-energy', r_psi=math.sqrt(.5), max_pad_factor=1,
        analytic=True, paths_exclude=None, precision=None,
    )
    DYNAMIC_PARAMETERS = {
        'oversampling', 'out_type', 'paths_exclude', 'pad_mode',
    }
    def __init__(self, shape, J=None, Q=8, T=None, average=True, oversampling=0,
                 out_type='array', pad_mode='reflect', smart_paths=.01,
                 max_order=2, vectorized=True, backend=None, **kwargs):
        super(ScatteringBase1D, self).__init__()
        self.shape = shape
        self.J = J if isinstance(J, tuple) else (J, J)
        self.Q = Q if isinstance(Q, tuple) else (Q, 1)
        self.T = T
        self.average = average
        self.oversampling = oversampling
        self.out_type = out_type
        self.pad_mode = pad_mode
        self.smart_paths = smart_paths
        self.max_order = max_order
        self.vectorized = vectorized
        self.backend = backend
        self.kwargs = kwargs

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """
        self.sigma0 = CFG['S1D']['sigma0']
        self.P_max = CFG['S1D']['P_max']
        self.eps = CFG['S1D']['eps']
        self.criterion_amplitude = CFG['S1D']['criterion_amplitude']

        # handle `kwargs` ####################################################
        if len(self.kwargs) > 0:
            I = fill_default_args(self.kwargs, self.default_kwargs,
                                  copy_original=True)
        else:
            I = self.default_kwargs

        # store for reference
        self.kwargs_filled = deepcopy(I)

        # set args
        for name in ScatteringBase1D.SUPPORTED_KWARGS:
            setattr(self, name, I.pop(name))

        # handle formatting: if supplied one value, set it for both orders
        for name in ('r_psi', 'normalize'):
            attr = getattr(self, name)
            if not isinstance(attr, tuple):
                setattr(self, name, (attr, attr))

        # invalid arg check
        if len(I) != 0:
            raise ValueError("unknown kwargs:\n{}Supported are:\n{}".format(
                I, ScatteringBase1D.SUPPORTED_KWARGS))
        ######################################################################

        # handle `vectorized`
        self.vectorized_early_U_1 = bool(self.vectorized and
                                         self.vectorized != 2)

        # handle `precision`
        if self.precision is None:
            if 'numpy' in str(self.backend).lower():
                self.precision = 'double'
            else:
                self.precision = 'single'
        elif self.precision not in ('single', 'double'):
            raise ValueError("`precision` must be 'single', 'double', or None, "
                             "got %s" % str(self.precision))

        # handle `shape`
        if isinstance(self.shape, numbers.Integral):
            self.N = self.shape
        elif isinstance(self.shape, tuple):
            self.N = self.shape[0]
            if len(self.shape) > 1:
                raise ValueError("If shape is specified as a tuple, it must "
                                 "have exactly one element")
        else:
            raise ValueError("shape must be an integer or a 1-tuple")
        # dyadic scale of N, also min possible padding
        self.N_scale = math.ceil(math.log2(self.N))

        # handle `J`
        if None in self.J:
            default_J = self.N_scale - 2
            self.J = list(self.J)
            self.J[0] = default_J if self.J[0] is None else self.J[0]
            self.J[1] = default_J if self.J[1] is None else self.J[1]
            self.J = tuple(self.J)

        # check `pad_mode`, set `pad_fn`
        if isinstance(self.pad_mode, FunctionType):
            def pad_fn(x):
                return self.pad_mode(x, self.pad_left, self.pad_right)
            self.pad_mode = 'custom'
        elif self.pad_mode not in ('reflect', 'zero'):
            raise ValueError(("unsupported `pad_mode` '{}';\nmust be a "
                              "function, or string, one of: 'zero', 'reflect'."
                              ).format(str(self.pad_mode)))
        else:
            def pad_fn(x):
                return self.backend.pad(x, self.pad_left, self.pad_right,
                                        self.pad_mode)
        self.pad_fn = pad_fn

        # check `normalize`
        supported = ('l1', 'l2', 'l1-energy', 'l2-energy')
        if any(n not in supported for n in self.normalize):
            raise ValueError(("unsupported `normalize`; must be one of: {}\n"
                              "got {}").format(supported, self.normalize))

        # ensure 2**max(J) <= nextpow2(N)
        Np2up = 2**self.N_scale
        if 2**max(self.J) > Np2up:
            raise ValueError(("2**J cannot exceed input length (rounded up to "
                              "pow2) (got {} > {})".format(
                                  2**max(self.J), Np2up)))

        # validate `max_pad_factor`
        # 1/2**J < 1/Np2up so impossible to create wavelet without padding
        if max(self.J) == self.N_scale and self.max_pad_factor == 0:
            raise ValueError("`max_pad_factor` can't be 0 if "
                             "max(J) == log2(nextpow2(N)). Got J=%s, N=%s" % (
                                 str(self.J), self.N))

        # check T or set default
        if self.T is None:
            self.T = 2**max(self.J)
        elif self.T == 'global':
            self.T = Np2up
        elif self.T > Np2up:
            raise ValueError(("The temporal support T of the low-pass filter "
                              "cannot exceed (nextpow2 of) input length. "
                              "Got {} > {})").format(self.T, self.N))
        # log2_T, global averaging
        self.log2_T = math.floor(math.log2(self.T))
        self.average_global_phi = bool(self.T == Np2up)
        self.average_global = bool(self.average_global_phi and self.average)

        # Compute the minimum support to pad (ideally)
        min_to_pad, pad_phi, pad_psi1, pad_psi2 = compute_minimum_support_to_pad(
            self.N, self.J, self.Q, self.T, r_psi=self.r_psi,
            sigma0=self.sigma0, P_max=self.P_max, eps=self.eps,
            criterion_amplitude=self.criterion_amplitude,
            normalize=self.normalize, pad_mode=self.pad_mode)
        if self.average_global:
            min_to_pad = max(pad_psi1, pad_psi2)  # ignore phi's padding

        J_pad_ideal = N_and_pad_2_J_pad(self.N, min_to_pad)
        if self.max_pad_factor is None:
            self.J_pad = J_pad_ideal
        else:
            self.J_pad = min(J_pad_ideal, self.N_scale + self.max_pad_factor)
            if J_pad_ideal - self.J_pad > 1:
                extent_txt = ('Severe boundary'
                              if J_pad_ideal - self.J_pad > 2 else
                              'Boundary')
                warnings.warn(f"{extent_txt} effects and filter distortion "
                              "expected per insufficient temporal padding; "
                              "recommended higher `max_pad_factor` or lower "
                              "`J` or `T`.")

        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.N)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.log2_T, self.J, self.pad_left, 2**self.J_pad - self.pad_right)

    def create_filters(self):
        # Create the filters
        self.phi_f, self.psi1_f, self.psi2_f = scattering_filter_factory(
            self.N, self.J_pad, self.J, self.Q, self.T,
            normalize=self.normalize, analytic=self.analytic,
            criterion_amplitude=self.criterion_amplitude,
            r_psi=self.r_psi, sigma0=self.sigma0, P_max=self.P_max,
            eps=self.eps, precision=self.precision)

        # energy norm
        # must do after `analytic` since analyticity affects norm
        if any('energy' in n for n in self.normalize):
            energy_norm_filterbank_tm(self.psi1_f, self.psi2_f, phi_f=None,
                                      J=self.J, log2_T=self.log2_T,
                                      normalize=self.normalize)

        # `paths_exclude`, `smart_paths`
        self._maybe_modified_paths_exclude = False
        self.handle_paths_exclude()
        # `paths_include_n2n1`
        self.build_paths_include_n2n1()

        # record whether configuration yields second order filters
        meta = ScatteringBase1D.meta(self)
        self._no_second_order_filters = (self.max_order < 2 or
                                         bool(np.isnan(meta['n'][-1][1])))

        # handle `vectorized`
        if self.vectorized_early_U_1:
            self.psi1_f_stacked = np.array([p[0] for p in self.psi1_f])[None]
            for n1 in range(len(self.psi1_f)):
                # replace original arrays so there's not two total copies
                self.psi1_f[n1][0] = []
                self.psi1_f[n1][0] = self.psi1_f_stacked[0, n1]
        else:
            self.psi1_f_stacked = None

    def handle_paths_exclude(self):
        supported = {'n2', 'j2', 'n2, n1'}
        if self.paths_exclude is None:
            self.paths_exclude = {nm: [] for nm in supported}
        # fill missing keys
        for pair in supported:
            self.paths_exclude[pair] = self.paths_exclude.get(pair, [])

        # smart_paths
        _handle_smart_paths(self.smart_paths, self.paths_exclude,
                            self.psi1_f, self.psi2_f)

        # user paths
        j_all = [p['j'] for p in self.psi2_f]
        n_psis = len(self.psi2_f)
        _handle_paths_exclude(
            self.paths_exclude, j_all, n_psis, supported, names=('n2', 'j2'))

    def build_paths_include_n2n1(self):
        self._paths_include_n2n1 = {}
        for n2, p2f in enumerate(self.psi2_f):
            j2 = p2f['j']
            self._paths_include_n2n1[n2] = []
            for n1, p1f in enumerate(self.psi1_f):
                j1 = p1f['j']
                if n2_n1_cond(n1, n2, j1, j2, self.paths_exclude):
                    self._paths_include_n2n1[n2].append(n1)

    def scattering(self, x):
        # input checks #######################################################
        _check_runtime_args_common(x)
        _check_runtime_args_scat1d(self.out_type, self.average)
        x, batch_shape, backend_obj = _handle_input_and_backend(self, x)

        # scatter, postprocess, return #######################################
        Scx = scattering1d(
            x, self.pad_fn, self.backend,
            paths_include_n2n1=self.paths_include_n2n1,
            **{arg: getattr(self, arg) for arg in (
                'log2_T', 'psi1_f', 'psi2_f', 'phi_f',
                'max_order', 'average', 'ind_start', 'ind_end',
                'oversampling', 'out_type', 'average_global', 'vectorized',
                'vectorized_early_U_1', 'psi1_f_stacked',
                )}
        )
        Scx = _restore_batch_shape(Scx, batch_shape, self.frontend_name,
                                   self.out_type, backend_obj)
        return Scx

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_scattering()`.
        """
        return compute_meta_scattering(self.psi1_f, self.psi2_f, self.phi_f,
                                       self.log2_T, self.paths_include_n2n1,
                                       max_order=self.max_order)

    def info(self, specs=True, show=True):
        """Prints relevant info. See `help(wavespin.toolkit.scattering_info)`."""
        return scattering_info(self, specs, show)

    # properties #############################################################
    @property
    def default_kwargs(self):
        return deepcopy(ScatteringBase1D.DEFAULT_KWARGS)

    @property
    def paths_include_n2n1(self):
        if self._maybe_modified_paths_exclude:
            self.build_paths_include_n2n1()
            self._maybe_modified_paths_exclude = False
        return self._paths_include_n2n1

    @property
    def paths_exclude(self):
        self._maybe_modified_paths_exclude = True
        return self._paths_exclude

    @paths_exclude.setter
    def paths_exclude(self, value):
        self._maybe_modified_paths_exclude = True
        self._paths_exclude = value

    # docs ###################################################################
    _doc_class = \
        r"""
        The 1D scattering transform

        The scattering transform computes a cascade of wavelet transforms
        alternated with a complex modulus non-linearity. The scattering
        transform of a 1D signal :math:`x(t)` may be written as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_J^{{(0)}} x(t) = x \star \phi_J(t)$,

            $S_J^{{(1)}} x(t, \lambda) = |x \star \psi_\lambda^{{(1)}}|
            \star \phi_J$, and

            $S_J^{{(2)}} x(t, \lambda, \mu) = |\,| x \star \psi_\lambda^{{(1)}}|
            \star \psi_\mu^{{(2)}} | \star \phi_J$.

        :math:`\star` denotes convolution in time. The filters
        $\psi_\lambda^{{(1)}}(t)$ and $\psi_\mu^{{(2)}}(t)$ are analytic
        wavelets with center frequencies $\lambda$ and $\mu$, while
        $\phi_J(t)$ is a real lowpass filter centered at the zero frequency.

        The `Scattering1D` class implements the 1D scattering transform for a
        given set of filters whose parameters are specified at initialization.
        While the wavelets are fixed, other parameters may be changed after
        the object is created, such as whether to compute all of
        :math:`S_J^{{(0)}} x`, $S_J^{{(1)}} x$, and $S_J^{{(2)}} x$ or just
        $S_J^{{(0)}} x$ and $S_J^{{(1)}} x$.
        {frontend_paragraph}
        Given an input `{array}` `x` of shape `(B, N)`, where `B` is the
        number of signals to transform (the batch size) and `N` is the length
        of the signal, we compute its scattering transform by passing it to
        the `scattering` method (or its alias, `_call__`{alias_name}), also
        see its docs.

        Example
        -------
        ::

            # Set the parameters of the scattering transform
            J = 6
            N = 2 ** 13
            Q = 8

            # Generate a sample signal
            x = np.random.randn(N)

            # Define a Scattering1D object
            sc = Scattering1D(N, Q, J)

            # Calculate the scattering transform
            Scx = sc(x)

            # Print relevant network info
            sc.info()

        Above, the length of the signal is :math:`N = 2^{{13}} = 8192`, while
        the maximum scale of the scattering transform is set to
        :math:`2^J = 2^6 = 64`. The time-frequency resolution of the first-order
        wavelets :math:`\psi_\lambda^{{(1)}}(t)` is controlled by `Q = 8`, which
        sets the number of wavelets per octave to `8` while preserving
        redundancy, which increases frequency resolution with higher `Q`.

        The second-order wavelets :math:`\psi_\mu^{{(2)}}(t)` have one wavelet
        per octave by default, but can be set like `Q = (8, 2)`. Internally,
        `J_fr` and `Q_fr`, the frequential variants of `J` and `Q`, are
        defaulted, but can be specified as well.

        {parameters}

        {attributes}

        References
        ----------
        This is a modification of
        https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
        frontend/base_frontend.py
        Kymatio, (C) 2018-present. The Kymatio developers.
        """

    _doc_params = \
        r"""
        Parameters
        ----------
        shape : int
           The length of the input signals.

        J : int / tuple[int]
            Controls the maximum dyadic scale of the scattering transform, hence
            also the total number of wavelets. The maximum width of any filter is
            `2**J`, not to be confused with maximum support.

            Tuple sets `J1` and `J2` separately, for first-order and second-order
            scattering, respectively.

            Defaults to `ceil(log2(shape)) - 2`. It's recommended to not exceed
            this default, as the largest scale wavelets will derive most
            information from padding rather than signal, or be distorted if
            there's not enough padding.

            Extended description:

                - In scattering literature, `J` also equals the number of octaves,
                  which makes the total number of wavelets `J*Q`; this is rarely
                  achieved, but it can be used as a ballpark.
                - Specifically, greater `J` corresponds to more *CQT* octaves.
                  This is desired and is what's assumed by scattering theory;
                  the non-CQT portion is ~STFT. The CQT portion enjoys time-warp
                  stability and other properties; it's where `xi / sigma = const`.
                - Since higher `Q` <=> higher width, a great `Q` relative to `J`
                  strays further (lower) of `J*Q`, as max width is achieved sooner
                  (at higher center frequency).

        Q : int >= 1 / tuple[int]
            Controls the number of wavelets per octave, and their frequency
            resolution. If tuple, sets `Q = (Q1, Q2)`, where `Q1` and `Q2` control
            behaviors in first and second order respectively.

                - Q1: For audio signals, a value of `>= 12` is recommended in
                  order to separate partials.
                - Q2: Recommended `1` for most applications. Higher values are
                  recommended only for `Q1 <= 10`.

            Defaults to `1`.

            **Extended description:**

                - Greater Q <=> greater frequency resolution <=> greater scale.
                - Q, together with `r_psi`, sets the quality factor of wavelets,
                  i.e. `(center freq) / bandwidth`. Literature sometimes
                  incorrectly states this same Q to be the quality factor.
                - Higher Q1 is better for multi-component signals (many AM/FM),
                  at expense of few-component signals with rapid AM/FM variations.
                - Q2 is the Q1 of amplitude modulations produced by Q1's
                  scalogram. Higher is better if we expect the AMs themselves
                  to be multi-component. Not to be confused with the *signal*
                  having multi-component AM; higher Q2 only benefits if the
                  *transform* extracts multi-component AM. Otherwise, Q2=1 is
                  ideal for single-component AM, and likelyhood of more components
                  strongly diminishes with greater Q1.

        T : int / str['global']
            Temporal width of the low-pass filter, controlling amount of imposed
            time-shift invariance and maximum subsampling.
            'global' for global average pooling (simple arithmetic mean),
            which is faster and eases on padding (ignores `phi_f`'s requirement).

        average : bool (default True)
            Determines whether the output is averaged in time or not. The
            averaged output corresponds to the standard scattering transform,
            while the un-averaged output skips lowpass filtering, and for
            first-order coefficients, equals the scalogram, i.e. modulus of
            the Continuous Wavelet Transform.

        oversampling : int >= 0 (default 0)
            Controls the oversampling factor relative to the default as a
            power of two. Since the convolving by wavelets (or lowpass filters)
            and taking the modulus reduces the high-frequency content of the
            signal, we can subsample to save space and improve performance.
            This may, in rare cases, lose non-neglibible precision; increasing
            `oversampling` will help, but greater than `1` should never be needed
            and should be reserved for other purposes (e.g. visualization).

            If `average_global=True`, doesn't change output size and has no
            effect on lowpassing, instead only reducing intermediate subsampling
            (conv with wavelets).

            Can be changed after instantiation. See `DYNAMIC_PARAMETERS` doc.

        out_type : str
            Output format:

                - 'array': `{array}` of shape `(B, C, N_sub)`, where
                  `N_sub` is coefficient temporal length after subsampling,
                  and `C` is the number of scattering coefficients.
                - 'list': list of dictionaries, of length `C`.
                  Each dictionary contains a coefficient, keyed by `'coef'`, and
                  the meta of filter(s) that produced it, e.g. `'j'` (scale)
                  and `'n'` (index, as in `psi1_f[n]`).
                  Each coefficient is shaped `(B, 1, N_sub)`.

            Defaults to 'array'.

            Can be changed after instantiation. See `DYNAMIC_PARAMETERS` doc.

        pad_mode : str / function
            Name of padding scheme to use, one of (`x = [1, 2, 3]`):

                - zero:    [0, 0, 0, 1, 2, 3, 0, 0]
                - reflect: [2, 3, 2, 1, 2, 3, 2, 1]

            Or, pad function with signature `pad_fn(x, pad_left, pad_right)`.
            This sets `self.pad_mode='custom'` (the name of padding is used
            for some internal logic).
            Defaults to 'reflect'.

            Can be safely changed after instantiation IF the original `pad_mode`
            wasn't `'zero'`. `'zero'` pads less than any other padding, so ch
            changing from `'zero'` to anything else risks incurring boundary
            effects.

        smart_paths : float / tuple[float, int] / str['primitive']
            Threshold controlling maximum energy loss guaranteed by the
            Smart Scattering Paths algorithm. Higher = more coefficients excluded,
            but more information lost.

            The guarantee isn't absolute but empirical. Tuple, like `(.01, 1)`,
            controls the degree of confidence:

                - 0: liberal. Reasonably safe.
                  Medium-sized survey consisting of audio and seizure iEEG
                  datasets didn't exceed this.
                - 1: conservative. Very safe.
                  Roughly interprets as 1 in 10 million chance of violating the
                  bound for a general signal. The odds of getting such a signal
                  as a WGN realization are much lower.
                - 2: adversarial-practical. Extremely safe.
                  Derived by maximizing the energy loss via gradient descent,
                  with very loose constraints to avoid practically impossible
                  edge cases. Not practically useful.
                - 3: adversarial. 2 but completely unconstrained.
                  Practically useless, but has uses for debugging.

            If we're allowed to fit the entire dataset, we can get "level 4" with
            *more* paths excluded via `wavespin.toolkit.fit_smart_paths`.

            'primitive' excludes coefficients if `j2 <= j1`, the criterion
            used by some libraries (e.g. Kymatio). This is a poor criterion,
            provided only for reference.

            `0` to disable. Will still exclude `j2==0` paths.

            Defaults to (.01, 1).
            See `help(wavespin.smart_paths_exclude)` for an extended description.
            For full control, set to `0` and pass in desired output of
            `smart_paths_exclude()` to `paths_exclude`.

        max_order : int
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.

        vectorized : bool (default True) / int[0, 1, 2]
            `True` batches operations (does more at once), which is much faster
            but uses more peak memory (highest amount used at any given time).

            `False` uses for-loops.

            `2` should be tried in case `True` fails. It's `True` without a
            memory-heavy step, which still preserves most of the speedup.

        kwargs : dict
            Keyword arguments controlling advanced configurations, passed via
            `**kwargs`.
            See `help(Scattering1D.SUPPORTED_KWARGS)`.
            These args are documented below.

        normalize : str / tuple[str]
            Tuple sets first-order and second-order separately, but only the
            first element sets `normalize` for `phi_f`. Supported:

                - 'l1': bandpass normalization; all filters' amplitude envelopes
                  sum to 1 in time domain (for Morlets makes them peak at 1
                  in frequency domain). `sum(abs(psi)) == 1`.
                - 'l2': energy normalization; all filters' energies are 1
                  in time domain; not suitable for scattering.
                  `sum(abs(psi)**2) == 1`.
                - 'l1-energy', 'l2-energy': additionally renormalizes the
                  entire filterbank such that its LP-sum (overlap of
                  frequency-domain energies) is `<=1` (`<=2` for time scattering
                  per using only analytic filters, without anti-analytic).

                  - This improves "even-ness" of input's representation, i.e.
                    no frequency is tiled too great or little (amplified or
                    attenuated).
                  - `l2-energy` is self-defeating, as the `energy` part
                    reverts to `l1`.
                  - `phi` is excluded from norm computations, against the
                    standard. This is because lowpass functions separately from
                    bandpass in coefficient computations, and any `log2_T`
                    that differs from `J` will attenuate or amplify lowest
                    frequencies in an undesired manner. In rare cases, this
                    *is* desired, and can be achieved by calling relevant
                    functions manually.

        r_psi : float / tuple[float]
            Controls the redundancy of the filters (the larger r_psi, the larger
            the overlap between adjacent wavelets), and stability against
            time-warp deformations (larger r_psi improves it).
            Defaults to sqrt(0.5). Must be >0 and <1.

        max_pad_factor : int (default 2) / None
            Will pad by at most `2**max_pad_factor` relative to `nextpow2(shape)`.

            E.g. if input length is 150, then maximum padding with
            `max_pad_factor=2` is `256 * (2**2) = 1024`.

            The maximum realizable value is `4`: a Morlet wavelet of scale `scale`
            requires `2**(scale + 4)` samples to convolve without boundary
            effects, and with fully decayed wavelets - i.e. x16 the scale,
            and the largest permissible `J` or `log2_T` is `log2(N)`.

            `None` means limitless. A limitation with `analytic=True` is,
            `compute_minimum_support_to_pad` does not account for
            `analytic=True`.

        analytic : bool (default True)
            `True` enforces strict analyticity by zeroing wavelets' negative
            frequencies. Required for accurate AM/FM extraction.

            `True` worsens time decay for very high and very low frequency
            wavelets. Except for rare cases, `True` is strongly preferred, as
            `False` can extract AM and FM where none exist, or fail to extract,
            and the loss of time localization is mostly drowned out by lowpass
            averaging.

        paths_exclude : dict[str: list[int]] / dict[str: int] / None
            Will exclude coefficients with these paths from computation and
            output.
            Supported keys: 'n2', 'j2'. E.g.:

                - {'n2': [2, 3, 5], 'j2': 1}
                - {'n2': [0, -1], 'j2': [-3, 0]}

            Negatives wrap around like indexing, e.g. `j2=-1` means `max(j2s)`.
            `dict[str: int]` will convert to `dict[str: list[int]]`.
            `n2=3` means will exclude `self.psi2_f[3]`.
            `j2=3` excludes all `p2f = self.psi2_f` with `p2f['j'] == 3`.

            For speed or memory purposes, it's recommended to tune `smart_paths`
            instead.

            Can be changed after instantiation. See `DYNAMIC_PARAMETERS` doc.

        precision : str / None
            One of:

                - `'single'`: float32, complex64
                - `'double'`: float64, complex128

            Controls numeric precision at which filters are built and computations
            are done. Will automatically cast input `x` to this precision.

            Defaults to `'double'` for NumPy, otherwise to `'single'`.
        """

    _doc_attrs = \
        r"""
        Attributes
        ----------
        N : int
            Alias for `shape`.

        J_pad : int
            `2**J_pad` is the padded length of the signal and filters.

        pad_left : int
            The amount of padding to the left of the signal.

        pad_right : int
            The amount of padding to the right of the signal.

        phi_f : dictionary
            A dictionary containing the lowpass filter at all resolutions.
            See `wavespin.scattering1d.filter_bank.scattering_filter_factory`.

        psi1_f : dictionary
            A dictionary containing all the first-order wavelet filters, each
            represented as a dictionary containing that filter at all resolutions.
            See `wavespin.scattering1d.filter_bank.scattering_filter_factory`.

        psi2_f : dictionary
            A dictionary containing all the second-order wavelet filters, each
            represented as a dictionary containing that filter at all resolutions.
            See `wavespin.scattering1d.filter_bank.scattering_filter_factory`.

        pad_mode : str
            One of supported padding modes: 'reflect', 'zero' - or 'custom'
            if a function was passed.

        pad_fn : function
            A backend padding function, or user function (as passed
            to `pad_mode`), with signature `pad_fn(x, pad_left, pad_right)`.

        average_global_phi : bool
            True if `T == nextpow2(shape)`, i.e. `T` is maximum possible
            and equivalent to global averaging, in which case lowpassing is
            replaced by simple arithmetic mean.

            In case of `average==False`, controls scattering logic for
            `phi_t` pairs in JTFS.

        average_global : bool
            True if `average_global_phi and average_fr`. Same as
            `average_global_phi` if `average_fr==True`.

            In case of `average==False`, controls scattering logic for
            `psi_t` pairs in JTFS.

        vectorized_early_U_1 : bool
            == `vectorized and vectorized != 2`.

            Controls whether a memory-intensive speedup is done in `scattering1d`,
            thereby whether `psi1_f_stacked` is constructed.

        psi1_f_stacked : tensor / None
            Concatenation of all first-order filters, shaped
            `(len(psi1_f), 2**J_pad)`. Done if `vectorized` is True.

        _maybe_modified_paths_exclude : bool
            Checked by `paths_include_n2n1` to update itself if `paths_exclude`
            has changed, then set to `False`.
            Set to `True` by setter and getter of `paths_exclude` each time the
            parameter is accessed.

        sigma0 : float
            Controls the definition of 'scale' and 'width' for all filters.
            Controls alias error in subsampling after lowpass filtering, and
            the number of CQT wavelets.

            Specifically, it's the continuous-time width parameter of the lowpass
            filter when `width = 1`. Any other width has `sigma = sigma0 / width`
            in frequency. For `T`, this means `sigma0 / T`, and the largest scale
            wavelet has `sigma = sigma0 / 2**J`.

            This quantity's default is set such that subsampling a signal
            lowpassed by the lowpass filter with `sigma = sigma0 / T`, then
            subsampled by `T`, is approximately unaliased, i.e. alias is under
            the default `criterion_amplitude`.  See
            `wavespin.measures.compute_bandwidth`.

            Configurable via `wavespin.CFG`.
            Defaults to 0.13.

        P_max : int >= 1
            Maximal number of periods to use to make sure that the Fourier
            transform of the filters is periodic. P_max = 5 is more than enough
            for double precision.

            The goal's to make the sampling grid long enough for the wavelet to
            decay below machine epsilon (`eps`), yielding an ~unaliased wavelet.
            For example, large `sigma` wavelets need greater `P_max`.

            Configurable via `wavespin.CFG`.
            Defaults to 5.

        eps : float
            Required machine precision for periodization in filter construction
            (single floating point is enough for deep learning applications).

            Configurable via `wavespin.CFG`.
            Defaults to 1e-7.

        criterion_amplitude : float
            The overarching precision parameter. Controls

                 - Boundary effects: error due to.
                   See `wavespin.measures.compute_spatial_support`.

                 - Filter decay: sufficiency of.
                   See `wavespin.measures.compute_spatial_support`.

                 - Aliasing: amount of, due to subsampling.
                   See `wavespin.measures.compute_max_dyadic_subsampling`.
                   See `wavespin.measures.compute_bandwidth`.

                 - Filter meta: 'j', 'width', 'support', 'scale', 'bw', 'bw_idxs'.
                   See `scattering_filter_factory` in
                   `wavespin.scattering1d.filter_bank`.

            May replace `sigma0`, `P_max`, and `eps` in the future.

            Configurable via `wavespin.CFG`.
            Defaults to 1e-3.

        DYNAMIC_PARAMETERS : set[str]
            Names of parameters that can be changed after object creation.
            These only affect the runtime computation of coefficients (no need
            to e.g. rebuild filterbank).

            *Note*, not all listed parameters are always safe to change, refer
            to the individual parameters' docs for exceptions.
        """

    _doc_scattering = \
        """
        Apply the scattering transform. Output format is specified by `out_type`,
        see `help(wavespin.Scattering1D())`.

        If `average=False`, outputs aren't averaged, but are simply the modulus
        of convolutions of wavelets with the signal - for first-order, this is
        the scalogram, or modulus of Continuous Wavelet Transform.

        Parameters
        ----------
        x : {array}
            An input `{array}` of size `(B, N)` or `(N,)`.
            `B` is integer or tuple of integers.

        Returns
        -------
        S : tensor / list[dict]
            Depends on `out_type` as described above.

        References
        ----------
        This is a modification of
        https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
        frontend/base_frontend.py
        Kymatio, (C) 2018-present. The Kymatio developers.
        """

    @classmethod
    def _document(cls):
        cls.__doc__ = ScatteringBase1D._doc_class.format(
            array=cls._doc_array,
            alias_name=cls._doc_alias_name,
            frontend_paragraph=cls._doc_frontend_paragraph,
            parameters=cls._doc_params,
            attributes=cls._doc_attrs,
        )
        cls.scattering.__doc__ = ScatteringBase1D._doc_scattering.format(
            array=cls._doc_array)


class TimeFrequencyScatteringBase1D():
    SUPPORTED_KWARGS_JTFS = {
        'aligned', 'out_3D', 'sampling_filters_fr',
        'analytic_fr', 'F_kind', 'max_pad_factor_fr',
        'pad_mode_fr', 'normalize_fr',
        'r_psi_fr', 'oversampling_fr', 'max_noncqt_fr',
        'out_exclude', 'paths_exclude',
    }
    DEFAULT_KWARGS_JTFS = dict(
        aligned=None, out_3D=False, sampling_filters_fr=('exclude', 'resample'),
        analytic_fr=True, F_kind='average', max_pad_factor_fr=2,
        pad_mode_fr='zero', normalize_fr='l1-energy',
        r_psi_fr=math.sqrt(.5), oversampling_fr=0, max_noncqt_fr=None,
        out_exclude=None, paths_exclude=None,
    )
    DYNAMIC_PARAMETERS_JTFS = {
        'oversampling', 'oversampling_fr', 'out_type', 'out_exclude',
        'paths_exclude', 'pad_mode', 'pad_mode_fr',
    }
    def __init__(self, J_fr=None, Q_fr=2, F=None, average_fr=False,
                 out_type='array', smart_paths=.007, implementation=None,
                 **kwargs):
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.average_fr = average_fr
        self.out_type = out_type
        self.smart_paths = smart_paths
        self.implementation = implementation
        self.kwargs_jtfs = kwargs

    def build(self):
        """Check args and instantiate `_FrequencyScatteringBase1D` object
        (which builds filters).

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation.
        """
        # if config yields no second order coeffs, we cannot do joint scattering
        if self._no_second_order_filters:
            raise ValueError("configuration yields no second-order filters; "
                             "try increasing `J`")

        # set standardized pair order to be followed by the API
        self._api_pair_order = ('S0', 'S1',
                                'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
                                'psi_t * psi_f_up', 'psi_t * psi_f_dn')

        # handle `kwargs`, `implementation` ##################################
        # validate
        if self.implementation is not None:
            if len(self.kwargs_jtfs) > 0:
                raise ValueError("if `implementation` is passed, `**kwargs` must "
                                 "be empty; got\n%s" % self.kwargs_jtfs)
            elif not (isinstance(self.implementation, int) and
                      self.implementation in range(1, 6)):
                raise ValueError("`implementation` must be None, or an integer "
                                 "1-5; got %s" % str(self.implementation))

        # fill defaults
        if len(self.kwargs_jtfs) > 0:
            I = fill_default_args(self.kwargs_jtfs, self.default_kwargs_jtfs,
                                  copy_original=True)
        else:
            I = self.default_kwargs_jtfs
        # handle `None`s
        if I['aligned'] is None:
            not_recalibrate = bool(I['sampling_filters_fr'] not in
                                   ('recalibrate', ('recalibrate', 'recalibrate'))
                                   )
            I['aligned'] = bool(not_recalibrate and I['out_3D'])

        # store for reference
        self.kwargs_filled_jtfs = deepcopy(I)

        # set args
        for name in TimeFrequencyScatteringBase1D.SUPPORTED_KWARGS_JTFS:
            setattr(self, name, I.pop(name))

        # invalid arg check
        if len(I) != 0:
            raise ValueError("unknown kwargs:\n{}\nSupported are:\n{}".format(
                I, TimeFrequencyScatteringBase1D.SUPPORTED_KWARGS_JTFS))

        # define presets
        self._implementation_presets = {
            1: dict(average_fr=False, aligned=False, out_3D=False,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='array'),
            2: dict(average_fr=True,  aligned=True,  out_3D=True,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='array'),
            3: dict(average_fr=True,  aligned=True,  out_3D=True,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='dict:list'),
            4: dict(average_fr=True,  aligned=False, out_3D=True,
                    sampling_filters_fr=('exclude', 'recalibrate'),
                    out_type='array'),
            5: dict(average_fr=True,  aligned=False, out_3D=True,
                    sampling_filters_fr=('recalibrate', 'recalibrate'),
                    out_type='dict:list'),
        }
        # override defaults with presets
        if isinstance(self.implementation, int):
            for k, v in self._implementation_presets[self.implementation].items():
                setattr(self, k, v)

        # handle `configs.py` that need handling here ########################
        if CFG['JTFS']['N_fr_p2up'] is None:
            self.N_fr_p2up = bool(self.out_3D)
        else:
            self.N_fr_p2up = CFG['JTFS']['N_fr_p2up']
        self.N_frs_min_global = CFG['JTFS']['N_frs_min_global']

        # `out_structure`
        if isinstance(self.implementation, int) and self.implementation in (3, 5):
            self.out_structure = 3
        else:
            self.out_structure = None

        # handle `out_exclude`
        if self.out_exclude is not None:
            if isinstance(self.out_exclude, str):
                self.out_exclude = [self.out_exclude]
            # ensure all names are valid
            supported = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                         'psi_t * phi_f', 'psi_t * psi_f_up', 'psi_t * psi_f_dn')
            for name in self.out_exclude:
                if name not in supported:
                    raise ValueError(("'{}' is an invalid coefficient name; "
                                      "must be one of: {}").format(
                                          name, ', '.join(supported)))

        # handle `F`
        if self.F is None:
            # default to one octave (Q wavelets per octave, J octaves,
            # approx Q*J total frequency rows, so averaging scale is `Q/total`)
            # F is processed further in `_FrequencyScatteringBase1D`
            self.F = self.Q[0]

        # handle `max_noncqt_fr`
        if self.max_noncqt_fr is not None:
            if not isinstance(self.max_noncqt_fr, (str, int)):
                raise TypeError("`max_noncqt_fr` must be str, int, or None, "
                                "got %s" % type(self.max_noncqt_fr))
            if self.max_noncqt_fr == 'Q':
                self.max_noncqt_fr = self.Q[0] // 2

        # `paths_exclude`, `smart_paths` (need earlier for N_frs)
        if not isinstance(self.paths_exclude, dict):
            self.paths_exclude = {}
        _handle_smart_paths(self.smart_paths, self.paths_exclude,
                            self.psi1_f, self.psi2_f)
        self.handle_paths_exclude_jtfs(n1_fr=False)

        # frequential scattering object ######################################
        paths_include_build, N_frs = self.build_scattering_paths()
        # number of psi1 filters
        self._n_psi1_f = len(self.psi1_f)
        max_order_fr = 1

        self.scf = _FrequencyScatteringBase1D(
            paths_include_build, N_frs,
            self.J_fr, self.Q_fr, self.F, max_order_fr,
            **{k: getattr(self, k) for k in (
                'average_fr', 'aligned', 'oversampling_fr',
                'sampling_filters_fr', 'out_3D',
                'max_pad_factor_fr', 'pad_mode_fr', 'analytic_fr',
                'max_noncqt_fr', 'normalize_fr', 'F_kind', 'r_psi_fr',
                '_n_psi1_f', 'precision', 'backend',
            )})
        self.finish_creating_filters()
        self.handle_paths_exclude_jtfs(n1_fr=True)

        # detach __init__ args, instead access `scf`'s via `__getattr__` #####
        # this is so that changes in attributes are reflected here
        init_args = ('J_fr', 'Q_fr', 'F', 'average_fr', 'oversampling_fr',
                     'sampling_filters_fr', 'max_pad_factor_fr', 'pad_mode_fr',
                     'r_psi_fr', 'out_3D')
        for init_arg in init_args:
            delattr(self, init_arg)

        # sanity warning #####################################################
        try:
            self.meta()
        except:
            warnings.warn(("Failed to build meta; the implementation may be "
                           "faulty. Try another configuration, or call "
                           "`jtfs.meta()` to debug."))

    def build_scattering_paths(self):
        """Determines all (n2, n1) pairs that will be scattered, not accounting
        for `paths_exclude`.

        Method exists since path inclusion criterion is complicated and
        reproducing it everywhere where relevant is verbose and error-prone.

        Paths algorithm
        ---------------
        Whether (n2, n1) is included is determined by

            1. `j2 >= j1`
               Precedes all else. Initial paths determination.
            2. `paths_exclude['n2, n1']`
               Overridden by 1, 3, 4. Secondary paths determination.
            3. `max_noncqt_fr`
               Overridden by 1. Secondary paths determination, alongside 2.
            4. `N_frs_min_global`
               Overridden by 1, 3. Tertiary paths determination.
            5. `N_fr_p2up`
               Overridden by 1, 3. Tertiary paths determination.

        First we compute jointly per 1, 2 and 3, then adjust per 4 and 5 but
        still constrained by 1 and 3. User-supplied `paths_exclude` is ignored,
        as it's meant only to control outputs, not filterbank build.

        Paths criteria
        --------------
        The resulting structure must satisfy

            1. `n1`'s must increment only by 1.
            2. If any `n1` is included for a given `n2`, then `n1=0` is included.

        This enforces consistency across all four path dimensions. Namely,
        padding and unpadding along frequency are needed only from one direction,
        and the `n1` dimension isn't ragged in outputs' starting index. Hence,

          - a 4D tensor is built by filling zeros only toward lower first-order
            frequencies
          - all `(n2, n1_fr)` joint slices begin at maximum first-order frequency

        The criterion is pertinent to any output configuration, 1D through 4D, as
        path exclusion is determined by maximum retrievable synthesis information,
        which is always contiguous and peaks near Nyquist by design of CQT.
        """
        # build paths ########################################################
        n_non_cqts = np.cumsum([not p['is_cqt'] for p in self.psi1_f])

        def is_cqt_if_need_cqt(n1):
            if self.max_noncqt_fr is None:
                return True
            return n_non_cqts[n1] <= self.max_noncqt_fr

        paths_include_build, N_frs = {}, []
        for n2 in range(len(self.psi2_f)):
            j2 = self.psi2_f[n2]['j']
            if j2 == 0:
                paths_include_build[n2] = []
                N_frs.append(0)
                continue

            n1s_include = []
            N_fr = 0

            # include all `n1` with `j2 >= j1` and not in `paths_exclude`,
            # also account for `max_noncqt_fr`.
            for n1, p1 in enumerate(self.psi1_f):
                j1 = p1['j']
                if (j2 >= j1 and
                        (n2, n1) not in self.paths_exclude['n2, n1'] and
                        is_cqt_if_need_cqt(n1)):
                    n1s_include.append(n1)
                    N_fr += 1

            # Account for `N_frs_min_global`
            if self.N_frs_min_global//2 < N_fr < self.N_frs_min_global:
                for n1, p1 in enumerate(self.psi1_f):
                    j1 = p1['j']
                    # must still satisfy j2 >= j1 and max_noncqt_fr
                    if (N_fr < self.N_frs_min_global and
                            n1 not in n1s_include and
                            j2 >= j1 and
                            is_cqt_if_need_cqt(n1)):
                        n1s_include.append(n1)
                        N_fr += 1

            if N_fr < self.N_frs_min_global:
                # if we failed, exclude this `n2`
                n1s_include = []
                N_fr = 0
            else:
                # Account for `N_fr_p2up`; overrides `paths_exclude`
                if self.N_fr_p2up:
                    N_fr_p2up = int(2**np.ceil(np.log2(N_fr)))
                    for n1, p1 in enumerate(self.psi1_f):
                        j1 = p1['j']
                        # must still satisfy j2 >= j1 and max_noncqt_fr
                        if (N_fr < N_fr_p2up and
                                n1 not in n1s_include and
                                j2 >= j1 and
                                is_cqt_if_need_cqt(n1)):
                            n1s_include.append(n1)
                            N_fr += 1

            # append paths
            paths_include_build[n2] = n1s_include
            N_frs.append(N_fr)

        # drop empty paths
        n2s = list(paths_include_build)
        for n2 in n2s:
            if paths_include_build[n2] == []:
                del paths_include_build[n2]

        # validate paths #####################################################
        # edge case
        if len(paths_include_build) == 0:
            # also fixable by lowering `N_frs_min_global` in `_configs.py`,
            # but bad idea so don't recommend
            raise Exception("Configuration failed to produce any joint coeffs. "
                            "Try raising J, changing Q, or setting "
                            "max_noncqt_fr=None")

        # n1s increment by 1, and n1==0 is present
        for n2, n1s in paths_include_build.items():
            assert np.all(np.diff(n1s) == 1), n1s
            assert n1s[0] == 0, n1s

        return paths_include_build, N_frs

    def handle_paths_exclude_jtfs(self, n1_fr=False):
        supported = {'n2', 'n1_fr', 'j2', 'j1_fr'}

        # n2
        if not n1_fr:
            j_all = [p['j'] for p in self.psi2_f]
            n_psis = len(self.psi2_f)
            self.paths_exclude = _handle_paths_exclude(
                self.paths_exclude, j_all, n_psis, supported, names=('n2', 'j2')
                )
        # n1_fr
        if n1_fr:
            j_all = self.psi1_f_fr_up['j'][0]
            n_psis = len(self.psi1_f_fr_up[0])
            _handle_paths_exclude(
                self.paths_exclude, j_all, n_psis, supported,
                names=('n1_fr', 'j1_fr'))

    def finish_creating_filters(self):
        """Handles necessary adjustments in time scattering filters unaccounted
        for in default construction.
        """
        # ensure phi is subsampled up to log2_T for `phi_t * psi_f` pairs
        max_sub_phi = lambda: max(k for k in self.phi_f if isinstance(k, int))
        while max_sub_phi() < self.log2_T:
            self.phi_f[max_sub_phi() + 1] = fold_filter_fourier(
                self.phi_f[0], nperiods=2**(max_sub_phi() + 1))

        # for early unpadding in joint scattering
        # copy filters, assign to `0` trim (time's `subsample_equiv_due_to_pad`)
        phi_f = {0: [v for k, v in self.phi_f.items() if isinstance(k, int)]}
        # copy meta
        for k, v in self.phi_f.items():
            if not isinstance(k, int):
                phi_f[k] = v

        diff = min(max(self.J) - self.log2_T, self.J_pad - self.N_scale)
        if diff > 0:
            for trim_tm in range(1, diff + 1):
                # subsample in Fourier <-> trim in time
                phi_f[trim_tm] = [v[::2**trim_tm] for v in phi_f[0]]
        self.phi_f = phi_f

        # adjust padding
        ind_start = {0: {k: v for k, v in self.ind_start.items()}}
        ind_end   = {0: {k: v for k, v in self.ind_end.items()}}
        if diff > 0:
            for trim_tm in range(1, diff + 1):
                pad_left, pad_right = compute_padding(self.J_pad - trim_tm,
                                                      self.N)
                start, end = compute_border_indices(
                    self.log2_T, self.J, pad_left, pad_left + self.N)
                ind_start[trim_tm] = start
                ind_end[trim_tm] = end
        self.ind_start, self.ind_end = ind_start, ind_end

    def scattering(self, x, Tx=None):
        # input checks #######################################################
        _check_runtime_args_common(x)
        _check_runtime_args_jtfs(self.average, self.average_fr, self.out_type,
                                 self.out_3D)
        x, batch_shape, backend_obj = _handle_input_and_backend(self, x)

        # scatter, postprocess, return #######################################
        Scx = timefrequency_scattering1d(
            x, Tx, self.backend.unpad, self.backend,
            **{arg: getattr(self, arg) for arg in (
                'J', 'log2_T', 'psi1_f', 'psi2_f', 'phi_f', 'scf', 'pad_fn',
                'pad_mode', 'pad_left', 'pad_right', 'ind_start', 'ind_end',
                'oversampling', 'oversampling_fr', 'aligned', 'F_kind',
                'average', 'average_global', 'average_global_phi',
                'out_type', 'out_3D', 'out_exclude', 'paths_exclude',
                'api_pair_order',
                )}
        )
        Scx = self._post_scattering(Scx, batch_shape, backend_obj)
        return Scx

    def _post_scattering(self, Scx, batch_shape, backend_obj):
        """Split from `scattering` for testing purposes."""
        if self.out_structure is not None:
            separate_lowpass = bool(self.out_structure != 5)
            Scx = pack_coeffs_jtfs(
                Scx, self.meta(), structure=self.out_structure,
                separate_lowpass=separate_lowpass,
                sampling_psi_fr=self.sampling_psi_fr, out_3D=self.out_3D)

        if len(batch_shape) > 1:
            Scx = _restore_batch_shape(
                Scx, batch_shape, self.frontend_name, self.out_type, backend_obj,
                self.out_3D, self.out_structure, is_jtfs=True)
        return Scx

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_jtfs()` with the parameters of the
        transform object.

        Returns
        ------
        meta : dictionary
            See `help(wavespin.scattering1d.utils.compute_meta_jtfs)`.
        """
        return compute_meta_jtfs(self.scf, **{arg: getattr(self, arg) for arg in (
            'psi1_f', 'psi2_f', 'phi_f', 'log2_T', 'sigma0',
            'average', 'average_global', 'average_global_phi', 'oversampling',
            'out_type', 'out_exclude', 'paths_exclude', 'api_pair_order',
            )})

    @property
    def fr_attributes(self):
        """Exposes `scf`'s attributes via main object."""
        return ('J_fr', 'Q_fr', 'N_frs', 'N_frs_max', 'N_frs_min',
                'N_fr_scales_max', 'N_fr_scales_min', 'scale_diffs', 'psi_ids',
                'J_pad_frs', 'J_pad_frs_max', 'J_pad_frs_max_init',
                'average_fr', 'average_fr_global', 'aligned', 'oversampling_fr',
                'F', 'log2_F', 'max_order_fr', 'max_pad_factor_fr', 'out_3D',
                'sampling_filters_fr', 'sampling_psi_fr', 'sampling_phi_fr',
                'phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn')

    @property
    def default_kwargs_jtfs(self):
        return deepcopy(TimeFrequencyScatteringBase1D.DEFAULT_KWARGS_JTFS)

    @property
    def api_pair_order(self):
        return self._api_pair_order

    def __getattr__(self, name):
        # access key attributes via frequential class
        # only called when default attribute lookup fails
        # `hasattr` in case called from Scattering1D
        if name in self.fr_attributes and hasattr(self, 'scf'):
            return getattr(self.scf, name)
        raise AttributeError(f"'{type(self).__name__}' object has no "
                             f"attribute '{name}'")  # standard attribute error

    # docs ###################################################################
    @classmethod
    def _document(cls):
        cls.__doc__ = TimeFrequencyScatteringBase1D._doc_class.format(
            frontend_paragraph=cls._doc_frontend_paragraph,
            parameters=cls._doc_params,
            attributes=cls._doc_attrs,
            terminology=cls._terminology,
        )
        cls.scattering.__doc__ = (
            TimeFrequencyScatteringBase1D._doc_scattering.format(
                array=cls._doc_array,
            )
        )

    def output_size(self):
        raise NotImplementedError("Not implemented for JTFS.")

    def create_filters(self):
        raise NotImplementedError("Implemented in `_FrequencyScatteringBase1D`.")

    _doc_class = \
        r"""
        The 1D Joint Time-Frequency Scattering transform.

        JTFS builds on time scattering by convolving first order coefficients
        with joint 2D wavelets along time and frequency, increasing
        discriminability while preserving time-shift invariance and time-warp
        stability. Invariance to frequency transposition can be imposed via
        frequential averaging, while preserving sensitivity to
        frequency-dependent time shifts.

        Joint wavelets are defined separably in time and frequency and permit fast
        separable convolution. Convolutions are followed by complex modulus and
        optionally averaging.

        The JTFS of a 1D signal :math:`x(t)` may be written as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_{{J, J_{{fr}}}}^{{(0)}} x(t) = x \star \phi_T(t),$

            $S_{{J, J_{{fr}}}}^{{(1)}} x(t, \lambda) =
            |x \star \psi_\lambda^{{(1)}}| \star \phi_T,$ and

            $S_{{J, J_{{fr}}}}^{{(2)}} x(t, \lambda, \mu, l, s) =
            ||x \star \psi_\lambda^{{(1)}}| \star \Psi_{{\mu, l, s}}|
            \star \Phi_{{T, F}}.$

        $\Psi_{{\mu, l, s}}$ comprises of five kinds of joint wavelets:

            $\Psi_{{\mu, l, +1}}(t, \lambda) =
            \psi_\mu^{{(2)}}(t) \psi_{{l, s}}(+\lambda)$
            spin up bandpass

            $\Psi_{{\mu, l, -1}}(t, \lambda) =
            \psi_\mu^{{(2)}}(t) \psi_{{l, s}}(-\lambda)$
            spin down bandpass

            $\Psi_{{\mu, -\infty, 0}}(t, \lambda) =
            \psi_\mu^{{(2)}}(t) \phi_F(\lambda)$
            temporal bandpass, frequential lowpass

            $\Psi_{{-\infty, l, 0}}(t, \lambda) =
            \phi_T(t) \psi_{{l, s}}(\lambda)$
            temporal lowpass, frequential bandpass

            $\Psi_{{-\infty, -\infty, 0}}(t, \lambda)
            = \phi_T(t) \phi_F(\lambda)$
            joint lowpass

        and $\Phi_{{T, F}}$ optionally does temporal and/or frequential averaging:

            $\Phi_{{T, F}}(t, \lambda) = \phi_T(t) \phi_F(\lambda)$

        Above, :math:`\star` denotes convolution in time and/or frequency. The
        filters $\psi_\lambda^{{(1)}}(t)$ and $\psi_\mu^{{(2)}}(t)$ are analytic
        wavelets with center frequencies $\lambda$ and $\mu$, while
        $\phi_T(t)$ is a real lowpass filter centered at the zero frequency.
        $\psi_{{l, s}}(+\lambda)$ is like $\psi_\lambda^{{(1)}}(t)$ but with
        its own parameters (center frequency, support, etc), and an anti-analytic
        complement (spin up is analytic).

        Filters are built at initialization. While the wavelets are fixed, other
        parameters may be changed after the object is created, such as `out_type`.

        {frontend_paragraph}

        Example
        -------
        ::

            # Set the parameters of the scattering transform
            J = 6
            N = 2 ** 13
            Q = 8

            # Generate a sample signal
            x = np.random.randn(N)

            # Instantiate the JTFS object
            jtfs = TimeFrequencyScattering1D(N, J, Q)

            # Compute
            Scx = jtfs(x)

            # Print relevant network info
            jtfs.info()

        Above, the length of the signal is :math:`N = 2^{{13}} = 8192`, while the
        maximum scale of the scattering transform is set to :math:`2^J = 2^6 =
        64`. The time-frequency resolution of the first-order wavelets
        :math:`\psi_\lambda^{{(1)}}(t)` is controlled by `Q = 8`, which sets
        the number of wavelets per octave to `8` while preserving redundancy,
        which increases frequency resolution with higher `Q`.

        The second-order wavelets :math:`\psi_\mu^{{(2)}}(t)` have one wavelet
        per octave by default, but can be set like `Q = (8, 2)`. Internally,
        `J_fr` and `Q_fr`, the frequential variants of `J` and `Q`, are defaulted,
        but can be specified as well.

        For further description and visuals, refer to:

            - JTFS overview https://dsp.stackexchange.com/a/78623/50076
            - JTFS structure & implem https://dsp.stackexchange.com/a/78625/50076

        {parameters}

        {attributes}

        {terminology}
        """

    _doc_params = \
        r"""
        Parameters
        ----------
        shape, T, average, oversampling, pad_mode :
            See `help(wavespin.Scattering1D)`.

            Unlike in time scattering, `T` plays a role even if `average=False`,
            to compute `phi_t` pairs.

        J : int / tuple[int]
            (Extended docs for JTFS, main in `help(wavespin.Scattering1D)`).

            Greater `J1` extends time-warp stability to lower frequencies, and
            other desired properties, as greater portion of the transform is CQT
            (fixed `xi` to `sigma` ratio and both exponentially spaced, as opposed
            to fixed `sigma` and linearly spaced `xi`). The number of CQT rows
            is approx `(J1 - 1)*Q1` (last octave is non-CQT), so the ratio of CQT
            to non-CQT is `(J1 - 1)/J1`, which is greater if `J1` is greater.

        Q : int / tuple[int]
            (Extended docs for JTFS)

              - `Q1`, together with `J`, determines `N_frs_max` and `N_frs`,
                or length of inputs to frequential scattering.
              - `Q2`, together with `J`, determines `N_frs` (via the `j2 >= j1`
                criterion), and total number of joint slices.
              - Greater `Q2` values better capture temporal AM modulations (AMM)
                of multiple rates. Suited for inputs of multirate or intricate AM.
                `Q2=2` is in close correspondence with the mamallian auditory
                cortex: https://asa.scitation.org/doi/full/10.1121/1.1945807
                Also see note in `Scattering1D` docs.

        J_fr : int
            `J` but for frequential scattering; see `J` docs in
            `help(wavespin.Scattering1D())`.

            Greater values increase the maximum captured FDTS slope and the
            longest captured structure along log-frequency, useful if signal
            behavior spans multiple octaves.

            Recommended to not set this above the default; see `J` docs.

            Default is determined at instantiation from longest frequential row
            in frequential scattering, set to `log2(nextpow2(N_frs_max)) - 2`,
            i.e. maximum possible minus 2, but no less than 3, and no more than
            max.

        Q_fr : int
            `Q` but for frequential scattering; see `J` docs in
            `help(wavespin.Scattering1D())`.

            Greater values better capture quefrential variations of multiple rates
            - that is, variations and structures along frequency axis of the
            wavelet transform's 2D time-frequency plane. Suited for inputs of many
            frequencies or intricate AM-FM variations. 2 or 1 should work for
            most purposes.

        F : int / str['global'] / None
            Temporal width of frequential low-pass filter, controlling amount of
            imposed frequency transposition invariance and maximum frequential
            subsampling. Defaults to `Q1`, i.e. one octave.

              - If `'global'`, sets to maximum possible `F` based on `N_frs_max`,
                and pads less (ignores `phi_f_fr`'s requirement).
              - Used even with `average_fr=False` (see its docs); this is likewise
                true of `T` for `phi_t * phi_f` and `phi_t * psi_f` pairs.

        average_fr : bool (default False)
            Whether to average (lowpass) along frequency axis.

            If `False`, `phi_t * phi_f` and `psi_t * phi_f` pairs are still
            computed.

        out_type : str
            Affects output format (but not how coefficients are computed).
            See `help(wavespin.TimeFrequencyScattering1D().scattering)`
            for further info.

                - `'list'`: coeffs are packed in a list of dictionaries, each
                  dict storing meta info, and output tensor keyed by `'coef.`.

                - `'array'`: concatenated along slices (`out_3D=True`) or mixed
                  slice-frequency dimension (`out_3D=False`). Both require
                  `average=True` (and `out_3D=True` additionally
                  `average_fr=True`).

                - `'dict:list' || 'dict:array'`: same as 'array' and 'list',
                  except coefficients will not be concatenated across pairs
                  - e.g. tensors from `'S1'` will be kept separate from those
                  from `'phi_t * psi_f'`.

                - See `out_3D` for all behavior controlled by `out_3D`, and
                  `aligned` for its behavior and interactions with `out_3D`.

          Can be changed after instantiation, see `DYNAMIC_PARAMETERS_JTFS` doc.

          **Shapes:**

          Shapes of individual coefficients, for `S0` and `S1` pairs and all
          others, respectively:

              - `'list'`:
                  - `(B, 1, t)`
                  - `(B, n_freqs, t)`

              - `'array'`:
                  - `(B, n_freqs, t)`
                  - `(B, n_freqs, t)` if `out_3D=False` else
                    `(B, mixed, n_freqs, t)`

          where `B` is batch shape, `t` is time, `n_freqs` refers to `n1`
          and `mixed` to the unrolling of `n2` & `n1_fr` (see Terminology).

          `'dict:list'` and `'dict:array'` follow `'list'` and `'array'` within
          each pair keying.

          See `examples/more/jtfs_out_shapes.py` for code to iterate the
          coefficients and print their shapes, along full example outputs.

        smart_paths : float / tuple
            See `help(wavespin.Scattering1D())`.

            For JTFS, this is simply a modification of `N_frs`.

            The default energy thresholding may be lower for JTFS, since
            second-order coefficients - namely, spinned - are more important
            than for time scattering, and since lesser input discontinuity
            along frequency improves quality of frequential scattering.

        implementation : int / None
            Preset configuration to use. Overrides the following parameters:

                - `average_fr, aligned, out_3D, sampling_filters_fr, out_type`

            If not `None`, then `**kwargs` must not be passed.

            **Implementations:**

                1: Standard for 1D convs. `(n1_fr * n2 * n1, t)`.
                  - average_fr = False
                  - aligned = False
                  - out_3D = False
                  - sampling_psi_fr = 'exclude'
                  - sampling_phi_fr = 'resample'

                2: Standard for 2D convs. `(n1_fr * n2, n1, t)`.
                  - average_fr = True
                  - aligned = True
                  - out_3D = True
                  - sampling_psi_fr = 'exclude'
                  - sampling_phi_fr = 'resample'

                3: Standard for 3D/4D convs. `(n1_fr, n2, n1, t)`. [2] but
                  - out_structure = 3

                4: Efficient for 2D convs. [2] but
                  - aligned = False
                  - sampling_phi_fr = 'recalibrate'

                5: Efficient for 3D convs. [3] but
                  - aligned = False
                  - sampling_psi_fr = 'recalibrate'
                  - sampling_phi_fr = 'recalibrate'

            where `n1` is the index of the first-order wavelet used to produce
            the output, `n2` is second-order and `n1_fr` is frequential
            scattering. Above it's used to refer to which filters are packed
            along which dimension.

            For beginners, `1` or `2` is recommended, or simply `None`.

            **Extended description:**

                - `out_structure` refers to packing output coefficients via
                  `pack_coeffs_jtfs(..., out_structure)`. This zero-pads and
                  reshapes coefficients, but does not affect their values or
                  computation in any way. (Thus, 3==2 except for shape). Requires
                  `out_type` 'dict:list' (default) or 'dict:array'; if
                  'dict:array' is passed, will use it instead.
                  See `help(wavespin.toolkit.pack_coeffs_jtfs)`.

                - `'exclude'` in `sampling_psi_fr` can be replaced with
                  `'resample'`, which yields significantly more coefficients and
                  doens't lose information (which `'exclude'` strives to
                  minimize), but is slower and the coefficients are mostly
                  "synthetic zeros" and uninformative.

                - `5` also makes sense with `sampling_phi_fr = 'resample'` and
                  small `F` (small enough to let `J_pad_frs` drop below max),
                  but the argument will only set `'recalibrate'`.

        kwargs : dict
            Keyword arguments controlling advanced configurations, passed via
            `**kwargs`.
            See `help(TimeFrequencyScattering1D.SUPPORTED_KWARGS_JTFS)`.
            These args are documented below.

        aligned : bool / None
            Whether to enforce uniformity of the full 4D JTFS tensor.
            If True, rows of joint slices index to same frequency for all slices.
            E.g. `S_2[3][5]` and `S_2[4][5]` (fifth row of third and fourth joint
            slices) correspond to same frequency.

            For any config, `aligned=True` enforces same total frequential stride
            for all slices, while `aligned=False` uses stride that maximizes
            information richness and density.
            See "Compute logic: stride, padding" in `core`, specifically
            'recalibrate'

            Defaults to True if `sampling_filters_fr != 'recalibrate'` and
            `out_3D=True`.

            **Extended description:**

            With `aligned=True`:

              - `out_3D=True`: all slices are zero-padded to have same number of
                rows. Earliest (low `n2`, i.e. high second-order freq) slices are
                likely to be mostly zero per `psi2` convolving with minority of
                first-order coefficients.

              - `out_3D=False`: all slices are padded by minimal amount needed to
                avert boundary effects.

                  - `average_fr=True`: number of output frequency rows will vary
                    across slices but be same *per `psi2_f`*.
                  - `average_fr=False`: number of rows will vary across and within
                    slices (`psi1_f_fr_up`-to-`psi1_f_fr_up`, and down).

            Note: `sampling_psi_fr = 'recalibrate'` breaks global alignment per
            shifting `xi_frs`, but preserves it on per-`N_fr_scale` (so also
            per-`n2`) basis.

            **Illustration**:

            Intended usage is `aligned=True` && `sampling_filters_fr='resample'`
            and `aligned=False` && `sampling_filters_fr='recalibrate'`. Below
            example assumes these.

            `x` == zero; `0, 4, ...` == indices of actual (nonpadded) data.
            That is, `x` means the convolution kernel (wavelet or lowpass) is
            centered in the padded region and contains less (or no) information,
            whereas `4 ` centers at `input[4]`. And `input` is `U1`, so the
            numbers are indexing `xi1` (i.e. are `n1`).

            ::

                data -> padded
                16   -> 128
                64   -> 128

                False:
                  [0,  4,  8, 16]  # stride 4
                  [0, 16, 32, 48]  # stride 16

                True:
                  [0,  x,  x,  x]  # stride 16
                  [0, 16, 32, 48]  # stride 16

            `False` is more information rich, containing fewer `x`. Further,
            with `out_3D=False`, it allows `stride_fr > log2_F`, making it more
            information dense
            (same info with fewer datapoints <=> non-oversampled).

            In terms of unpadding with `out_3D=True`:

                - `aligned=True`: we always have fr stride == `log2_F`, with
                  which we index `ind_start_fr_max` and `ind_end_fr_max`
                  (i.e. take maximum of all unpads across `n2` from this factor
                  and reuse it across all other `n2`).
                - `aligned=False`: decided from `N_fr_scales_max` case, where we
                  compute `unpad_len_common_at_max_fr_stride`. Other `N_fr_scales`
                  use that quantity to compute `min_stride_to_unpad_like_max`.
                  See "Compute logic: stride, padding" in `core`, specifically
                  'recalibrate'.

            The only exception is with `average_fr_global_phi and not average_fr`:
            spinned pairs will have zero stride, but `phi_f` pairs will have max.

        out_3D : bool (default False)
            `True` (requires `average_fr=True`) adjusts frequential scattering
            to enable concatenation along joint slices dimension, as opposed to
            flattening (mixing joint slices and frequencies).

            Both `True` and `False` can still be concatenated into the "true" JTFS
            4D structure; see `help(wavespin.toolkit.pack_coeffs_jtfs)` for a
            complete description. The difference is in how values are computed,
            especially near boundaries. More importantly, `True` enforces
            `aligned=True` on *per-`n2`* basis, enabling 3D convs even with
            `aligned=False`.

            **Extended description**:

                - `False` will unpad freq by exact amounts for each joint slice,
                  whereas `True` will unpad by minimum amount common to all
                  slices at a given subsampling factor to enable concatenation.
                  See `scf_compute_padding_fr()`.
                - See `aligned` for its interactions with `out_3D` (also below).
                - `aligned and out_3D` may sometimes be significantly more
                  compute-intensive than just `aligned`. `aligned and not out_3D`
                  is an alternative worth inspecting with `visuals.viz_jtfs_2d`,
                  as the added compute often won't justify the added information.

            **`aligned` and `out_3D`:**

            From an information/classification standpoint,

              - `True` is more information-rich. The 1D equivalent case is
                unpadding by 3, instead of by 6 and then zero-padding by 3: same
                final length, but former fills gaps with partial convolutions
                where latter fills with zeros.
              - `False` is the "latter" case.

            Above distinction is emphasized. `out_3D=True` && `aligned=True`
            imposes a large compute overhead by padding all `N_frs`
            near-maximally. If a given `N_fr` is treated as a complete input,
            then unpadding anything more than `N_fr/stride` includes convolutions
            from completely outside of this input, which we never do elsewhere.

              - However, we note, if `N_fr=20` for `n2=2` and `N_frs_max=100`,
                what this really says is, we *expect* the 80 lowest frequency rows
                to yield negligible energy after convolving with `psi2_f`.
                That is, zeros (i.e. padding) are the *true* continuation of the
                input (hence why 'conj-reflect-zero'), and hence, unpadding by
                more than `N_fr/stride` is actually within bounds.
              - Hence, unpadding by `N_fr/stride` and then re-padding (i.e.
                treating `N_fr` as a complete input) is actually a distortion and
                is incorrect.
                Namely, the complete scattering, without any
                shortcuts/optimizations on stride or padding, is consistent with
                unpadding `> N_fr/stride`.
                At the same time, depending on our feature goals, especially if
                slices are processed independently, such distortion might be
                preferable to avoid air-packing (see "Illustration" in `aligned`).
              - The described re-padding happens with `aligned=True` &&
                `out_3D=False` packed into a 3D/4D tensor; even without
                re-padding, this config tosses out valid outputs
                (unpads to `N_fr/stride`), though less informative ones.

        sampling_filters_fr : str / tuple[str]
            Controls filter properties for frequential input lengths (`N_frs`)
            below maximum. That is, as JTFS is a multi-input network along
            frequential scattering, we may opt to treat different input lengths
            differently.

              - 'resample': preserve physical dimensionality
                (center frequeny, width) at every length (trimming in time
                domain).
                E.g. `psi = psi_fn(N/2) == psi_fn(N)[N/4:-N/4]`.

              - 'recalibrate': recalibrate filters to each length.

                - widths (in time): widest filter is halved in width, narrowest is
                  kept unchanged, and other widths are re-distributed from the
                  new minimum to same maximum.
                - center frequencies: all redistribute between new min and max.
                  New min is set as `2 / new_length`
                  (old min was `2 / max_length`).
                  New max is set by halving distance between old max and 0.5
                  (greatest possible), e.g. 0.44 -> 0.47, then 0.47 -> 0.485, etc.

              - 'exclude': same as 'resample' except filters wider than
                `widest / 2` are excluded. (and `widest / 4` for next
                `N_fr_scales`, etc).

            Tuple can set separately `(sampling_psi_fr, sampling_phi_fr)`, else
            both set to same value.

            From an information/classification standpoint:

                - 'resample' enforces freq invariance imposed by `phi_f_fr` and
                  physical scale of extracted modulations by `psi1_f_fr_up`
                  (& down). This is consistent with scattering theory and is the
                  standard used in existing applied literature.
                - 'recalibrate' remedies a problem with 'resample'. 'resample'
                  calibrates all filters relative to longest input; when the
                  shortest input is very different in comparison, it makes most
                  filters appear lowpass-ish. In contrast, recalibration enables
                  better exploitation of fine structure over the smaller interval
                  (which is the main motivation behind wavelets,
                  a "multi-scale zoom".)
                - 'exclude' circumvents the problem by simply excluding wide
                  filters. 'exclude' is simply a subset of 'resample', preserving
                  all center frequencies and widths - a 3D/4D coefficient packing
                  will zero-pad to compensate
                  (see `help(wavespin.toolkit.pack_coeffs_jtfs)`).

            Note: `sampling_phi_fr = 'exclude'` will re-set to `'resample'`, as
            `'exclude'` isn't a valid option (there must exist a lowpass for every
            fr input length).

        analytic_fr : bool (default True)
            If True, will enforce strict analyticity/anti-analyticity:

                - zero negative frequencies for temporal and spin up bandpasses
                - zero positive frequencies for spin down bandpasses
                - halve the Nyquist bin for both spins

            `True` improves FDTS-discriminability, especially for
            `r_psi > sqrt(.5)`, but may slightly worsen wavelet time decay.
            Also see `analytic` in `help(wavespin.Scattering1D())`.

        F_kind : str['average', 'decimate']
            Kind of lowpass filter to use for spinned coefficients:

                - 'average': Gaussian, standard for scattering. Imposes time-shift
                  invariance.

                - 'decimate': Hamming-windowed sinc (~brickwall in freq domain).
                  Decimates coefficients: used for unaliased downsampling,
                  without imposing invariance.

                   - Preserves more information along frequency than 'average'
                     (see "Info preservation" below).
                   - Ignores padding specifications and pads its own way
                     (future TODO)
                   - Corrects negative outputs via absolute value; the negatives
                     are possible since the kernel contains negatives, but are in
                     minority and are small in magnitude.

            Does not interact with other parameters in any way - that is, won't
            affect stride, padding, etc - only changes the lowpass filter for
            spinned pairs. `phi_f` pairs will still use Gaussian, and `phi_f_fr`
            remains Gaussian but is used only for `phi_f` pairs. Has no effect
            with `average_fr=False`.

            'decimate' is an experimental but tested feature:

                - 'torch' backend:
                    - will assume GPU use and move built filters to GPU
                    - lacks `register_filters` support, so filters are invisible
                      to `nn.Module`
                - filters are built dynamically, on per-requested basis. The first
                  run is slower than the rest as a result
                - `oversampling_fr != 0` is not supported
                - is differentiable

            **Info preservation:**

            `'decimate'`

              - 1) Increases amount of information preserved.

                  - Its cutoff spills over the alias threshold, and there's
                    notable amount of aliasing (subject to future improvement).
                  - Its main lobe is narrower than Gauss's, hence better
                    preserving component separation along frequency, at expense
                    of longer tails.
                  - Limited reconstruction experiments did not reveal a definitive
                    advantage over Gaussian: either won depending on transform and
                    optimizer configurations. Further study is required.

              - 2) Reduces distortion of preserved information.

                  - The Gaussian changes relative scalings of bins, progressively
                    attenuating higher frequencies, whereas windowed sinc is ~flat
                    in frequency until reaching cutoff (i.e. it copies input's
                    spectrum). As a result, Gaussian blurs, while sinc faithfully
                    captures the original.
                  - At the same time, sinc increases distortion per aliasing, but
                    the net effect is a benefit.

              - 3) Increases distortion of preserved information.

                  - Due to notable aliasing. Amount of energy aliased is ~1/110 of
                    total energy, while for Kymatio's Gaussian, it's <1/1000000.
                  - Due to the time-domain kernel having negatives, which
                    sometimes outputs negatives for a non-negative input,
                    requiring correction.
                  - `2)` benefits much more than `3)` harms

            `2)` is the main advantage and is the main motivation for 'decimate':
            we want a smaller unaveraged output, that resembles the full original.

        max_pad_factor_fr : int / None (default) / list[int]
            `max_pad_factor` for frequential axis in frequential scattering.

                - None: unrestricted; will pad as much as needed.

                - list[int]: controls max padding for each `N_fr_scales`
                  separately, in reverse order (max to min).

                    - Values may not be such that they yield increasing
                      `J_pad_frs`
                    - If the list is insufficiently long (less than number of
                      scales), will extend list with the last provided value
                      (e.g. `[1, 2] -> [1, 2, 2, 2]`).
                    - Indexed by `scale_diff == N_fr_scales_max - N_fr_scales`

                - int: will convert to list[int] of same value.

            Specified values aren't guaranteed to be realized. They override some
            padding values, but are overridden by others.

            Overrides:

                - Padding that lessens boundary effects and wavelet distortion
                  (`min_to_pad`).

            Overridden by:

                - `J_pad_frs_min_limit_due_to_phi`
                - `J_pad_frs_min_limit_due_to_psi`
                - Will not allow any `J_pad_fr > J_pad_frs_max_init`
                - With `sampling_psi_fr = 'resample'`, will not allow `J_pad_fr`
                  that yields a pure sinusoid wavelet (raises `ValueError` in
                  `filter_bank.get_normalizing_factor`).

            A limitation of `None` with `analytic=True` is,
            `compute_minimum_support_to_pad` does not account for it.

        pad_mode_fr : str['zero', 'conj-reflect-zero'] / function
            Name of frequential padding mode to use:

                - 'zero': zero-padding. Faster but worse at energy conservation
                  for large `J_fr`.
                - 'conj-reflect-zero': zero-pad lower frequency portion, and
                  conjugate + 'reflect' all else. Recommended for large `J_fr`.

            Can also be a function with signature `pad_fn_fr(x, pad_fr, scf, B)`;
            see `_right_pad` in
            `wavespin.scattering1d.core.timefrequency_scattering1d`.

            If using `pad_mode = 'reflect'` and `average = True`, reflected
            portions will be automatically conjugated before frequential
            scattering to avoid spin cancellation. For same reason, there isn't
            `pad_mode_fr = 'reflect'`.

            **Extended description:**

                - 'zero' is default only because it's faster; in general, if
                  `J_fr >= log2(N_frs_max) - 3`, 'conj-reflect-zero' should be
                  preferred.
                  See https://github.com/kymatio/kymatio/discussions/
                  752#discussioncomment-864234
                - Also, note that docs and comments tend to mention only `J, J_fr`
                  and `T, F`, but `Q, Q_fr` also significantly affect max scale:
                  higher -> greater max scale.

            Can be safely changed after instantiation IF the original
            `pad_mode_fr` wasn't `'zero'`. See `pad_mode` in
            `help(wavespin.Scattering1D())`.

        normalize_fr : str
            See `normalize` in `help(wavespin.Scattering1D())`.
            Applies to `psi1_f_fr_up`, `psi1_f_fr_dn`, `phi_f_fr`.

        r_psi_fr : float
            See `r_psi` in `help(wavespin.Scattering1D())`.
            See `help(wavespin.scattering1d.utils.calibrate_scattering_filters)`.

        oversampling_fr : int >= 0 (default 0)
            How much to oversample along frequency axis.
            Also see `oversampling` in `help(wavespin.Scattering1D())`.

            If `average_fr_global=True`, doesn't change output size and has no
            effect on lowpassing, instead only reducing intermediate subsampling
            (conv with wavelets).

            Can be changed after instantiation, see `DYNAMIC_PARAMETERS_JTFS` doc.

        max_noncqt_fr : int / None / str['Q']
            Maximum non-CQT rows (`U1` vectors) to include in frequential
            scattering, i.e. rows derived from `not psi1_f[n1]['is_cqt']`.

              - `0` means CQT-only; `3` means *up to* 3 rows (rather than
                *at least*) for any given `N_fr` (see `N_frs`).
              - `None` means all non-CQT are permitted
              - `'Q'` means up to `Q1//2` non-CQT are permitted

            Non-CQT rows are sub-ideal for frequential scattering, as they violate
            the basic assumption of convolution that the input is uniformly
            spaced.

            **Extended description:**

            CQT rows are uniformly spaced in log-space, non-CQT in linear space,
            so the two aren't directly compatible and form a discontinuity
            boundary.

              - This lowers FDTS discriminability, albeit not considerably.
              - It also affects frequency transposition invariance and time-warp
                stability, as a shift in log space is a shift by different amount
                in linear (& fixed wavelet bandwidth) space. The extent is again
                acceptable.
              - At the same time, excluding such rows loses information.
              - `max_noncqt_fr` can control this tradeoff, but in general, `None`
                (the default) is recommended.
              - Higher `J` (namely `J1`) increases the CQT portion (see `J`),
                mediating aforementioned effects.

        out_exclude : list/tuple[str] / None
            Will exclude coefficients with these names from computation and output
            All names:

            ::

                'S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
                'psi_t * psi_f_up', 'psi_t * psi_f_dn'

          Note, `'S1'` is always computed, but will still exclude from output.
          Can be changed after instantiation, see `DYNAMIC_PARAMETERS_JTFS` doc.

        paths_exclude : dict[str: list[int]] / dict[str: int] / None
            Will exclude coefficients with these paths from computation and
            output.
            Supported keys: 'n2', 'n1_fr', 'j2', 'j1_fr'. E.g.:

                - {'n2': [2, 3, 5], 'n1_fr': [0, -1]}
                - {'j2': 1, 'j1_fr': [3, 1]}
                - {'n2': [0, 1], 'j2': [-1]}

            Negatives wrap around like indexing, e.g. `j2=-1` means `max(j2s)`.
            `dict[str: int]` will convert to `dict[str: list[int]]`.
            `n2=3` means will exclude `self.psi2_f[3]`.
            `j2=3` excludes all `p2f = self.psi2_f` with `p2f['j'] == 3`.
            `n1_fr=3` excludes `self.psi1_f_fr_up[psi_id][3]` for all `psi_id`.

            Excluding `j2==1` paths yields greatest speedup, and is recommended
            in compute-restricted settings, as they're the lowest energy paths
            (i.e. generally least informative). However, generally it's
            recommended to tune `smart_paths` instead.

            Note, `n2` and `n1_fr` only affect `psi_t *` pairs. To exclude
            `phi_t *` pairs, use `out_exclude`.

            Can be changed after instantiation, *except for* the 'n2, n1' key
            (which isn't meant for user setting in the first place).
            See `DYNAMIC_PARAMETERS_JTFS` doc.
        """

    _doc_attrs = \
        r"""
        Attributes
        ----------
        scf : `_FrequencyScatteringBase1D`
            Frequential scattering object, storing pertinent attributes and
            filters. Temporal scattering's attributes are accessed directly via
            `self`.

            "scf" abbreviates "scattering frequency" (i.e. frequential
            scattering).

        N_frs : list[int]
            List of lengths of frequential columns (i.e. numbers of frequential
            rows) in joint scattering, indexed by `n2` (second-order temporal
            wavelet index).
            E.g. `N_frs[3]==52` means 52 highest-frequency vectors from
            first-order time scattering are fed to `psi2_f[3]` (effectively, a
            multi-input network).

        N_frs_max : int
            `== max(N_frs)`.

        N_frs_min : int
            `== min(N_frs_realized)`

        N_frs_realized: list[int]
            `N_frs` without `0`s.
            Unaffected by `paths_exclude` to allow `paths_exclude` to be
            dynamically configurable.

        N_frs_max_all : int
            `== _n_psi1_f`. Used to compute `_J_pad_frs_fo` (unused quantity),
            and `n_zeros` in `_pad_conj_reflect_zero` (`core/timefreq...`).

        N_fr_scales : list[int]
            `== nextpow2(N_frs)`. Filters are calibrated relative to these
            (for 'exclude' & 'recalibrate' `sampling_psi_fr`).

        N_fr_scales_max : int
            `== max(N_fr_scales)`. Used to set `J_pad_frs_max` and
            `J_pad_frs_max_init`.

                - `J_fr` default is set using this value, and `J_fr` cannot
                  exceed it. If `F == 2**J_fr`, then `average_fr_global=True`.
                - Used in `compute_J_pad_fr()` and `psi_fr_factory()`.

        N_fr_scales_min : int
            `== min(N_fr_scales)`.

            Used in `scf._compute_J_pad_frs_min_limit_due_to_psi`.

        N_fr_scales_unique : list[int]
            `N_fr_scales` without duplicate entries.

        scale_diffs : list[int]
            `scale_diff == N_fr_scales_max - N_fr_scale`.
            0-indexed surrogate for `N_fr_scale`, indexing multi-length logic
            for building filterbanks and computing stride and padding.

        scale_diffs_unique : list[int]
            `scale_diffs` without duplicate entries.

        scale_diff_max_recalibrate : int / None
            Max permitted `scale_diff`, per `sampling_psi_fr='recalibrate'`
            and `sigma_max_to_min_max_ratio`. Build terminates to avoid filters
            more time-localized than the most time-localized original wavelet
            (max sigma in freq domain), within set tolerance, as a quality check.

        total_conv_stride_over_U1s : dict[int: list[int]]
            Stores total strides for frequential scattering (`psi_f` pairs,
            followed by `phi_f_fr`):

                {scale_diff: [stride0, stride1, ...]}  # list indexed by `n1_fr`

            `J_pad_frs` is built to accomodate stride.
            See `help(scf.compute_stride_fr)`.
            See "Compute logic: stride, padding" in
            `core.timefrequency_scattering1d`.

            `over_U1` seeks to emphasize that it is the stride over first order
            coefficients.

        total_conv_stride_over_U1s_phi : dict[int: int]
            Stores strides for frequential scattering (`phi_f` pairs):

                {scale_diff: stride}

            Derives from `total_conv_stride_over_U1s`, differently depending on
            `average_fr`, `aligned`, and `sampling_phi_fr`.
            See "Stride, padding: `phi_f` pairs" in
            `core.timefrequency_scattering1d`.

        n1_fr_subsamples : dict[str: dict[int: list[int]]]
            Stores strides for frequential scattering (`psi_f` pairs).
            Accounts for both `j1_fr` and `log2_F_phi`, so subsampling won't alias
            the lowpass.

                {'spinned: {scale_diff: [...]},
                 'phi':    {scale_diff: [...]}}

            See `scf._compute_scale_and_stride_logic`.

        log2_F_phis : dict[str: dict[int: list[int]]]
            `log2_F`-equivalent - that is, maximum permitted subsampling, and
            dyadic scale of invariance - of lowpass filters used for a given
            pair, `N_fr_scale`, and `n1_fr` -

                {'spinned: {scale_diff: [...]},
                 'phi':    {scale_diff: [...]}}

            Equals `log2_F` everywhere with `sampling_phi_fr='resample'`.
            Is `None` for 'spinned' if
            `not (average_fr and not average_fr_global)`
            (since convolution with lowpass isn't used).

        log2_F_phi_diffs : dict[str: dict[int: list[int]]]
            `== log2_F - log2_F_phi`. See `log2_F_phis`.

        unpad_len_common_at_max_fr_stride : int
            Unpad length at `N_fr_scales_max`, with whatever frequential stride
            happens to be there. Used when `out_3D=True` && `aligned=False` to set
            unpad length for other `N_fr_scales`, via
            `min_stride_to_unpad_like_max`.

            See "Compute logic: stride, padding" in
            `core.timefrequency_scattering1d`, specifically 'recalibrate'

        phi_f_fr : dict[int: dict, str: dict]
            Contains the frequential lowpass filter at all resolutions.
            See `help(wavespin.scattering1d.filter_bank.phi_fr_factory)`.

            Full type spec:

                dict[int: dict[int: list[tensor[float]]],
                     str: dict[int: dict[int: list[int]], float]]

        psi1_f_fr_up : dict[int: dict, str: dict]
            List of dictionaries containing all frequential scattering filters
            with "up" spin.
            See `help(wavespin.scattering1d.filter_bank.psi_fr_factory)`.

            Full type spec:

                dict[int: list[tensor[float]],
                     str: dict[int: list[int/float]]]

        psi1_f_fr_dn : dict[int: dict, str: dict]
            `psi1_f_fr_up`, but with "down" spin, forming a complementary pair.

            Full type spec:

                dict[int: list[tensor[float]],
                     str: dict[int: list[int/float]]]

        psi_ids : dict[int: int]
            See `help(wavespin.scattering1d.filter_bank_jtfs.psi_fr_factory)`.

        psi_fr_params : dict[int:dict[str:list]]
            Parameters used to build filterbanks for frequential scattering.
            See `help(scf._compute_psi_fr_params)` and
            `help(wavespin.scattering1d.filter_bank_jtfs.psi_fr_factory)`.

        average_fr_global_phi : bool
            True if `F == nextpow2(N_frs_max)`, i.e. `F` is maximum possible
            and equivalent to global averaging, in which case lowpassing is
            replaced by simple arithmetic mean.

            If True, `sampling_phi_fr` has no effect.

            In case of `average_fr==False`, controls scattering logic for
            `phi_f` pairs.

        average_fr_global : bool
            True if `average_fr_global_phi and average_fr`. Same as
            `average_fr_global_phi` if `average_fr==True`.

              - In case of `average_fr==False`, controls scattering logic for
                `psi_f` pairs.
              - If `True`, `phi_fr` filters are never used (but are still
                created).
              - Results are very close to lowpassing w/ `F == 2**N_fr_scales_max`.
                Unlike with such lowpassing, `psi_fr` filters are allowed to be
                created at lower `J_pad_fr` than shortest `phi_fr` (which also is
                where greatest deviation with `not average_fr_global` occurs).

        log2_F : int
            Equal to `floor(log2(F))`; is the maximum frequential subsampling
            factor if `average_fr=True` (otherwise that factor is up to `J_fr`).

        J_pad_frs : list[int]
            log2 of padding lengths of frequential columns in joint scattering
            (column lengths given by `N_frs`). See `scf.compute_padding_fr()`.

        J_pad_frs_max_init : int
            Set as reference for computing other `J_pad_fr`.

            Serves to create the initial frequential filterbank, and equates to
            `J_pad_frs_max` with `sampling_psi_fr='resample'` &
            `sampling_phi_fr='resample'`. Namely, it is the maximum padding under
            "standard" frequential scattering configurations.

        J_pad_frs_max : int
            `== max(J_pad_frs)`.

        J_pad_frs_min : int
            `== min(J_pad_frs)` (excluding -1).

        J_pad_frs_min_limit_due_to_psi: int / None
            Controls minimal padding.
            Prevents severe filterbank distortions due to insufficient padding.
            See docs for `_compute_J_pad_frs_min_limit_due_to_psi`,
            in `filter_bank_jtfs.py`.

        _J_pad_fr_fo : int
            Padding for the `phi_t` pairs. Used only in edge case testing,
            and to warn of an edge case handling in `core`.

            `phi_t` pairs reuse spinned pairs' largest padding, yet the `N_fr` of
            `phi_t` pairs is always greater than or equal to that of spinned's,
            which at times otherwise yields greater padding.
            This is done to simplify implementation, with minimal or negligible
            effect on `phi_t` pairs.

            `core` edge case: `max_pad_factor_fr=0` with
            `N_fr_scales_max < N_fr_scale_fo` means the padded length will be
            less than `_n_psi1_f`. Accounting for this requires changing
            `J_pad_frs_max_init`, yet `compute_padding_fr` doesn't reuse
            `J_pad_frs_max_init`, hence accounting for this is complicated and
            unworthwhile. Instead, will only include up to `2**N_fr_scales_max`
            rows from `U1`.

        min_to_pad_fr_max : int
            `min_to_pad` from `compute_minimum_support_to_pad(N=N_frs_max)`.
            Used in computing `J_pad_fr`. See `scf.compute_J_pad_fr()`.

        unrestricted_pad_fr : bool
            `True` if `max_pad_factor is None`. Affects padding computation and
            filter creation:

              - `phi_f_fr` w/ `sampling_phi_fr=='resample'`:

                - `True`: will limit the shortest `phi_f_fr` to avoid distorting
                  its time-domain shape
                - `False`: will compute `phi_f_fr` at every `J_pad_fr`

              - `psi_f_fr` w/ `sampling_psi_fr=='resample'`: same as for phi

        subsample_equiv_relative_to_max_pad_init : int
            Amount of *equivalent subsampling* of frequential padding relative to
            `J_pad_frs_max_init`, indexed by `n2`.
            See `help(scf.compute_padding_fr())`.

        scale_diff_max_to_build: int / None
            Largest `scale_diff` (smallest `N_fr_scale`) for which a filterbank
            will be built; lesser `N_fr_scales` will reuse it. Used alongside
            other attributes to control said building, also as an additional
            sanity and clarity layer.
            Prevents severe filterbank distortions due to insufficient padding.

              - Affected by `sampling_psi_fr`, padding, and filterbank param
                choices.
                See docs for `_compute_J_pad_frs_min_limit_due_to_psi`,
                in `filter_bank_jtfs.py`.
              - With 'recalibrate', `scale_diff_max_to_build=None` if build didn't
                terminate per `sigma_max_to_min_max_ratio`.

        sigma_max_to_min_max_ratio : float >= 1
            Largest permitted `max(sigma) / min(sigma)`. Used with 'recalibrate'
            `sampling_psi_fr` to restrict how large the smallest sigma can get.

            Worst cases (high `subsample_equiv_due_to_pad`):

              - A value of `< 1` means a lower center frequency will have
                the narrowest temporal width, which is undesired.
              - A value of `1` means all center frequencies will have the same
                temporal width, which is undesired.
              - The `1.2` default was chosen arbitrarily as a seemingly good
                compromise between not overly restricting sigma and closeness to
                `1`.

            Configurable via `wavespin.CFG`.

        width_exclude_ratio : float > 0
            Ratio to use in `sampling_psi_fr = 'exclude'` and `'recalibrate'`.

            - 'exclude': a frequential scattering filter is excluded if

                  width > 2**N_fr_scale * width_exclude_ratio

              As the default is `0.5`, this means we keep filters with width
              being at most half the (p2up of) frequential input length.

            - 'recalibrate': the original (max input length) filterbank is
              reused as long as its highest width wavelet satisfies

                  width < 2**N_fr_scale * width_exclude_ratio

            Configurable via `wavespin.CFG`.

        N_fr_p2up : bool / None
            Whether to include, in frequential scattering, first-order rows
            all the way up to the next power of 2 relative to the `N_fr` we'd get
            otherwise. Still satisfies `j2 >= j1`.

            Defaults to `True` if `out_3D=True`, and to `False` otherwise.
            Overridden by `max_noncqt_fr` where applicable.

            The added rows add information at minimal expense of speed.
            With `out_3D=True`, there is no change in output shape, but with
            `False`, there is an increase, hence the choice of defaults.
            `False` is recommended when output size changes in spirit of
            `smart_paths`.

            Configurable via `wavespin.CFG`.

        N_frs_min_global : int
            Enforces "min(N_frs) >= N_frs_min_global". Is used to exclude `n2`'s
            that derive from very few `n1`'s, which are uninformative and heavy
            with transform artifacts. The algorithm is, for any `n2`:

                1. N_frs_min_global//2 < n_n1s < N_frs_min_global
                   Appends n1s until `n_n1s == N_frs_min_global`.
                2. n_n1s <= N_frs_min_global//2
                   Discards the n2. Per diminishing scattering energies, appending
                   is ineffective.
                3. n_n1s >= N_frs_min_global
                   Do nothing.

            Set to `0` to disable.
            Configurable via `wavespin.CFG`.

        _n_phi_f_fr : int
            `== len(phi_f_fr)`.
            Used for setting `max_subsample_equiv_before_phi_fr`.

        pad_left_fr : int
            Amount of padding to left  of frequential columns
            (or top of joint matrix). Unused in implementation; can be used
            by user if `pad_mode` is a function.

        pad_right_fr : int
            Amount of padding to right of frequential columns
            (or bottom of joint matrix).

        ind_start_fr : list[list[int]]
            Frequential unpadding start index, indexed by `n2` (`N_fr`) and
            stride:

                `ind_start_fr[n2][stride]`

            See `help(scf.compute_padding_fr)` and `scf.compute_unpadding_fr`.

        ind_end_fr : list[list[int]]
            Frequential unpadding end index. See `ind_start_fr`.

        ind_start_fr_max : list[int]
            Frequential unpadding start index common to all `N_frs` for
            `out_3D=True`, determined from `N_frs_max` case, indexed by stride:

                `ind_start_fr_max[stride]`

            See `ind_start_fr`.

        ind_end_fr_max : list[int]
            Frequential unpadding end index common to all `N_frs` for
            `out_3D=True`.
            See `ind_start_fr_max`.

        max_order_fr : int == 1
            Frequential scattering's `max_order`. Unused.

        paths_exclude : dict
            Additionally internally sets 'n2, n1' pair via `smart_paths`.

        out_structure : int / None
            See `implementation` docs.

        api_pair_order : set[str]
            Standardized order of JTFS coefficient pairs for internal
            consistency.

        DYNAMIC_PARAMETERS_JTFS : set[str]
            See `DYNAMIC_PARAMETERS` in `help(wavespin.Scattering1D())`.

            *Note*, not all listed parameters are always safe to change, refer
            to the individual parameters' docs for exceptions.
        """

    _terminology = \
        r"""
        Terminoloy
        ----------
        FDTS :
            Frequency-Dependent Time Shift. JTFS's main purpose is to detect
            these. Down spin wavelet resonates with up chirp (rising; right-shifts
            with increasing freq), up spin with down chirp (left-shifts with
            increasing freq). Spin up is crest down (oscillation up, with maxima
            perpendicularly, i.e. down), spin down is crest up, so crest up
            resonates with chirp up, and crest down with chirp down.

        Frequency transposition :
            i.e. frequency shift, except in context of wavelet transform (hence
            scattering) it means log-frequency shift.

        n1 : int
            Index of temporal wavelet in first-order scattering:
            `self.psi1_f[n1]`.
            Is used to refer to said wavelets or the coefficients they produce.

        n2 : int
            `n1` but for second order,
            `self.psi2_f[n2]`.

        n1_fr : int
            `n1` but for frequential scattering,
            `self.psi1_f_fr_up[psi_id][n1_fr]`.

        psi_id : int
            Index of frequential filterbank, see `psi_ids`.

        n1_fr_subsample: int
            Subsampling done after convolving with `psi_fr`
            See `help(wavespin.scattering1d.core.timefrequency_scattering1d)`.
        """

    _doc_scattering = \
        """
        Apply the Joint Time-Frequency Scattering transform.

        Given an input `{array}` of size `(B, N)`, where `B` is the batch size
        (can be tuple) and `N` is the length of the individual signals, computes
        its JTFS.

        Output format is specified by `out_type`: a list, array, tuple, or
        dictionary of lists or arrays with keys specifying coefficient names as
        follows:

        ::

            {{'S0': ...,                # (time)  zeroth order
             'S1': ...,                # (time)  first order
             'phi_t * phi_f': ...,     # (joint) joint lowpass
             'phi_t * psi_f': ...,     # (joint) time lowpass (w/ freq bandpass)
             'psi_t * phi_f': ...,     # (joint) freq lowpass (w/ time bandpass)
             'psi_t * psi_f_up': ...,  # (joint) spin up
             'psi_t * psi_f_dn': ...,  # (joint) spin down
             }}

        Coefficient structure depends on `average, average_fr, aligned, out_3D`,
        and `sampling_filters_fr`. See `help(wavespin.toolkit.pack_coeffs_jtfs)`
        for a complete description.

        Parameters
        ----------
        x : {array}
            An input `{array}` of size `(B, N)` or `(N,)`.

        Returns
        -------
        S : dict[tensor/list] / tensor/list / tuple of former two
            See above.
        """


__all__ = ['ScatteringBase1D', 'TimeFrequencyScatteringBase1D']
