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
import inspect
from copy import deepcopy

import numpy as np

from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering1d import timefrequency_scattering1d
from ..core.cwt1d import cwt1d
from ..filter_bank import (scattering_filter_factory, fold_filter_fourier,
                           N_and_pad_to_J_pad, n2_n1_cond)
from ..refining import energy_norm_filterbank_tm
from ..filter_bank_jtfs import _FrequencyScatteringBase1D
from ..scat_utils import (
    compute_border_indices, compute_padding, compute_minimum_support_to_pad,
    build_compute_graph_tm, compute_meta_scattering, build_cwt_unpad_indices,
)
from ..scat_utils_jtfs import (
    build_compute_graph_fr, compute_meta_jtfs,
    make_psi1_f_fr_stacked_dict, pack_runtime_spinned,
)
from .frontend_utils import (
    _handle_paths_exclude, _handle_smart_paths, _handle_input_and_backend,
    _check_runtime_args_common, _check_runtime_args_scat1d,
    _check_runtime_args_jtfs, _restore_batch_shape, _ensure_positive_integer,
    _check_jax_double_precision,
    _raise_reactive_setter, _setattr_and_handle_reactives,
    _handle_device_non_filters_jtfs, _warn_boundary_effects,
    _handle_pad_mode, _handle_pad_mode_fr,
)
from ...utils.gen_utils import fill_default_args
from ...toolkit import pack_coeffs_jtfs, scattering_info
from ... import CFG


class ScatteringBase1D(ScatteringBase):
    DEFAULT_KWARGS = dict(
        normalize='l1-energy',
        r_psi=math.sqrt(.5),
        max_pad_factor=1,
        analytic=True,
        paths_exclude=None,
        precision=None,
    )
    SUPPORTED_KWARGS = tuple(DEFAULT_KWARGS)
    DYNAMIC_PARAMETERS = {
        'oversampling', 'out_type', 'paths_exclude', 'pad_mode',
    }
    REACTIVE_PARAMETERS = {
        'oversampling', 'paths_exclude', 'pad_mode',
    }

    def __init__(self, shape, J=None, Q=8, T=None, average=True, oversampling=0,
                 out_type='array', pad_mode='reflect', smart_paths=.01,
                 max_order=2, vectorized=None, backend=None, **kwargs):
        super(ScatteringBase1D, self).__init__()

        # set args while accounting for special cases
        args = dict(
            shape=shape,
            J=J,
            Q=Q,
            T=T,
            average=average,
            oversampling=oversampling,
            out_type=out_type,
            pad_mode=pad_mode,
            smart_paths=smart_paths,
            max_order=max_order,
            vectorized=vectorized,
            backend=backend,
        )
        for name, value in args.items():
            if name == 'J' and not isinstance(value, (tuple, list)):
                value = (value, value)
            elif name == 'Q' and not isinstance(value, (tuple, list)):
                value = (value, 1)
            _setattr_and_handle_reactives(
                self, name, value, ScatteringBase1D.REACTIVE_PARAMETERS)
        self.kwargs = kwargs

        # instantiate certain internals
        self._moved_to_device = False

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
        self.halve_zero_pad = CFG['S1D']['halve_zero_pad']

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
            if name in ScatteringBase1D.REACTIVE_PARAMETERS:
                setattr(self, f'_{name}', I.pop(name))
            else:
                setattr(self, name, I.pop(name))

        # `r_psi`, `normalize` formatting
        if not isinstance(self.r_psi, tuple):
            self.r_psi = (self.r_psi, self.default_kwargs['r_psi'])
        if not isinstance(self.normalize, tuple):
            self.normalize = (self.normalize, self.normalize)

        # invalid arg check
        if len(I) != 0:  # no-cov
            supported = (inspect.getfullargspec(ScatteringBase1D).args +
                         list(ScatteringBase1D.SUPPORTED_KWARGS))
            supported = [nm for nm in supported if nm != 'self']
            raise ValueError("unknown args: {}\n\nSupported are:\n{}".format(
                list(I), supported))
        # --------------------------------------------------------------------

        # handle `vectorized`
        self.vectorized_early_U_1 = bool(self.vectorized and
                                         self.vectorized != 2)

        # handle `precision`
        if self.precision is None:
            if self.frontend_name == 'numpy':
                self.precision = 'double'
            else:
                self.precision = 'single'
        elif self.precision not in ('single', 'double'):  # no-cov
            raise ValueError("`precision` must be 'single', 'double', or None, "
                             "got %s" % str(self.precision))
        if self.precision == 'double' and self.frontend_name == 'jax':
            _check_jax_double_precision()

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

        # check `J`, `Q`, `T`
        _ensure_positive_integer(self, ('J', 'Q', 'T'))

        # handle `J`
        if None in self.J:
            default_J = self.N_scale - 3
            self.J = list(self.J)
            self.J[0] = default_J if self.J[0] is None else self.J[0]
            self.J[1] = default_J if self.J[1] is None else self.J[1]
            self.J = tuple(self.J)

        # check `pad_mode`, set `pad_fn`
        _handle_pad_mode(self)

        # check `normalize`
        supported = ('l1', 'l2', 'l1-energy', 'l2-energy')
        if any(n not in supported for n in self.normalize):  # no-cov
            raise ValueError(("unsupported `normalize`; must be one of: {}\n"
                              "got {}").format(supported, self.normalize))

        # ensure 2**max(J) <= nextpow2(N)
        Np2up = 2**self.N_scale
        if 2**max(self.J) > Np2up:  # no-cov
            raise ValueError(("2**J cannot exceed input length (rounded up to "
                              "pow2) (got {} > {})".format(
                                  2**max(self.J), Np2up)))

        # validate `max_pad_factor`
        # 1/2**J < 1/Np2up so impossible to create wavelet without padding
        if max(self.J) == self.N_scale and self.max_pad_factor == 0:  # no-cov
            raise ValueError("`max_pad_factor` can't be 0 if "
                             "max(J) == log2(nextpow2(N)). Got J=%s, N=%s" % (
                                 str(self.J), self.N))

        # check T or set default
        if self.T is None:
            self.T = 2**max(self.J)
        elif self.T == 'global':
            self.T = Np2up
        elif self.T > Np2up:  # no-cov
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
            normalize=self.normalize, pad_mode=self.pad_mode,
            halve_zero_pad=self.halve_zero_pad)
        if self.average_global:
            min_to_pad = max(pad_psi1, pad_psi2)  # ignore phi's padding

        J_pad_ideal = N_and_pad_to_J_pad(self.N, min_to_pad)
        if self.max_pad_factor is None:
            self.J_pad = J_pad_ideal
        else:
            self.J_pad = min(J_pad_ideal, self.N_scale + self.max_pad_factor)
            diff = J_pad_ideal - self.J_pad
            if diff > 0:
                _warn_boundary_effects(diff, self.J_pad, min_to_pad, self.N)

        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.N)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.log2_T, self.J, self.pad_left, 2**self.J_pad - self.pad_right)

        # in case we want to do CWT
        self.cwt_unpad_indices = build_cwt_unpad_indices(
            self.N, self.J_pad, self.pad_left)

    def create_filters(self):
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

    def finish_build(self):
        """Post-filter-creation steps."""
        # `paths_exclude`, `smart_paths`
        self.handle_paths_exclude()
        # `paths_include_n2n1`
        self.build_paths_include_n2n1_scat1d()
        # build the runtime compute graph
        self._compute_graph = build_compute_graph_tm(self)

        # record whether configuration yields second order filters
        meta = ScatteringBase1D.meta(self)
        self._no_second_order_filters = (self.max_order < 2 or
                                         bool(np.isnan(meta['n'][-1][1])))

        # handle `vectorized`
        if self.vectorized_early_U_1:
            self.psi1_f_stacked = np.array([p[0] for p in self.psi1_f])[None]
            self.deduplicate_stacked_filters()
        else:
            self.psi1_f_stacked = None

        # CWT: determine max safe hop_sizes -- see attribute docs
        # `.9` as `1` may be too liberal, in general and esp. for float32
        self.max_invertible_hop_size = .9 * min(
            p['support'][0] for p in self.psi1_f)
        # `.55` is taken from `compute_bandwidth` w/
        # `criterion_amplitude=sqrt(.01/(1-.01))`, in spirit of `smart_paths=.01`.
        # `.5` is ~the ratio of result on a Morlet to `criterion_amplitude=1e-3`.
        self.max_low_alias_hop_size = (2**self.J_pad / 2) / max(
            .5*p['bw'][0] for p in self.psi1_f)

        # declare build finished
        self.pack_runtime_filters_scat1d()

    def handle_paths_exclude(self):
        """
          - `paths_exclude` validation and formatting corrections
          - updating `paths_exclude` per `smart_paths`
        """
        supported = {'n2', 'j2', 'n2, n1'}
        if self.paths_exclude is None:
            self._paths_exclude = {nm: [] for nm in supported}
        # fill missing keys
        for pair in supported:
            self._paths_exclude[pair] = self.paths_exclude.get(pair, [])

        # smart_paths
        _handle_smart_paths(self.smart_paths, self._paths_exclude,
                            self.psi1_f, self.psi2_f, self.r_psi)

        # user paths
        j_all = [p['j'] for p in self.psi2_f]
        n_psis = len(self.psi2_f)
        _handle_paths_exclude(
            self._paths_exclude, j_all, n_psis, supported, names=('n2', 'j2'))

    def build_paths_include_n2n1(self):
        """Build according to the finalized `paths_exclude`."""
        self.build_paths_include_n2n1_scat1d()

    def build_paths_include_n2n1_scat1d(self):
        self._paths_include_n2n1 = {}
        for n2, p2f in enumerate(self.psi2_f):
            j2 = p2f['j']
            self._paths_include_n2n1[n2] = []
            for n1, p1f in enumerate(self.psi1_f):
                j1 = p1f['j']
                if n2_n1_cond(n1, n2, j1, j2, self.paths_exclude):
                    self._paths_include_n2n1[n2].append(n1)

        # delete empty keys
        for n2 in range(len(self.psi2_f)):
            if self._paths_include_n2n1[n2] == []:
                del self._paths_include_n2n1[n2]

    @property
    def scattering1d_kwargs(self):
        return {arg: getattr(self, arg) for arg in
                ('pad_fn', 'backend', 'compute_graph',
                 'log2_T', 'psi1_f', 'psi2_f', 'phi_f',
                 'max_order', 'average', 'ind_start', 'ind_end',
                 'oversampling', 'out_type', 'average_global', 'vectorized',
                 'vectorized_early_U_1', 'psi1_f_stacked',
                 )}

    def scattering(self, x):
        # input checks
        _check_runtime_args_common(x)
        _check_runtime_args_scat1d(self.out_type, self.average)
        x, batch_shape, backend_obj = _handle_input_and_backend(self, x)

        # scatter, postprocess, return
        Scx = scattering1d(x, **self.scattering1d_kwargs)
        Scx = _restore_batch_shape(Scx, batch_shape, self.frontend_name,
                                   self.out_type, backend_obj)
        return Scx

    def cwt(self, x, hop_size=1):
        # input checks
        if hop_size not in self.cwt_unpad_indices:
            valid_hops = np.array(list(self.cwt_unpad_indices))
            closest_alt = valid_hops[np.argmin(np.abs(valid_hops - hop_size))]
            raise ValueError(f"`hop_size={hop_size}` is invalid, "
                             f"`hop_size={closest_alt}` is closest alt. (Must "
                             "wholly divide padded length length of `x`, meaning "
                             f"`({2**self.J_pad} / hop_size).is_integer()`).")
        elif hop_size > self.max_invertible_hop_size:  # no-cov
            warnings.warn(("`hop_size={}` exceeds max invertible hop size {}! "
                          "Recommended to lower it, or to increase `Q`."
                           ).format(hop_size, self.max_invertible_hop_size))
        _check_runtime_args_common(x)
        x, *_ = _handle_input_and_backend(self, x)

        # transform, return
        Wx = cwt1d(x, hop_size, self.pad_fn, self.backend,
                   self.psi1_f, self.psi1_f_stacked,
                   self.cwt_unpad_indices, self.vectorized)
        return Wx

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dict
            See `help(wavespin.scattering1d.scat_utils.compute_meta_scattering)`.
        """
        return compute_meta_scattering(self.psi1_f, self.psi2_f, self.phi_f,
                                       self.log2_T, self.paths_include_n2n1,
                                       max_order=self.max_order)

    def info(self, specs=True, show=True):
        """Prints relevant info. See `help(wavespin.toolkit.scattering_info)`."""
        return scattering_info(self, specs, show)

    # properties #############################################################
    # Read-only attributes ---------------------------------------------------
    @property
    def compute_graph(self):
        return self._compute_graph

    @property
    def paths_include_n2n1(self):
        return self._paths_include_n2n1

    @property
    def default_kwargs(self):
        return deepcopy(ScatteringBase1D.DEFAULT_KWARGS)

    @property
    def filters_device(self):
        """Returns the current device of filters, if filters were explicitly
        moved to a device (non-NumPy). Checks based on `phi_f`, since it's always
        included in moving.

        Not called `device` to avoid overloading certain backends' attributes
        (e.g. `torch.nn.Module`).
        """
        if self._moved_to_device:
            device = (self.phi_f[0][0].device if hasattr(self, 'scf') else
                      self.phi_f[0].device)
            if hasattr(device, '__call__'):
                device = device()
            return device
        return None

    # Reactive attributes & helpers ------------------------------------------
    @property
    def paths_exclude(self):
        return self._paths_exclude

    @paths_exclude.setter
    def paths_exclude(self, value):
        self.raise_reactive_setter('paths_exclude')

    @property
    def oversampling(self):
        return self._oversampling

    @oversampling.setter
    def oversampling(self, value):
        self.raise_reactive_setter('oversampling')

    @property
    def pad_mode(self):
        return self._pad_mode

    @pad_mode.setter
    def pad_mode(self, value):
        self.raise_reactive_setter('pad_mode')

    def pack_runtime_filters(self):
        return self.pack_runtime_filters_scat1d()

    def pack_runtime_filters_scat1d(self):
        pass

    def deduplicate_stacked_filters(self):
        if self.psi1_f_stacked is not None:
            # replace original arrays so there's not two total copies
            # TODO what's this for?
            for n1 in range(len(self.psi1_f)):
                self.psi1_f[n1][0] = self.psi1_f_stacked[0, n1]

    def raise_reactive_setter(self, name):
        _raise_reactive_setter(name, 'REACTIVE_PARAMETERS')

    def rebuild_for_reactives(self):
        # handle easiest first
        _handle_pad_mode(self)

        self.build_paths_include_n2n1()
        self._compute_graph = build_compute_graph_tm(self)
        self.pack_runtime_filters()
        self.deduplicate_stacked_filters()

    def update(self, **kwargs):
        """The proper way to set "dynamic attributes".

        Specifically, this method is required for "reactive attributes",
        as these attributes require rebuilding certain internal attributes
        in order to take effect. Example:

        ::

            sc = Scattering1D(N=512, oversampling=0)

            # correct syntax
            sc.update(oversampling=1)
            pe = sc.paths_exclude
            pe['n2'].append(2)
            sc.update(paths_exclude=pe)

            # can do many at once
            sc.update(oversampling=1, paths_exclude=pe)

            # incorrect syntax
            sc.oversampling = 1  # will error
            sc.paths_exclude['n2'].append(2)    # won't error, silent failure
            sc.paths_exclude.update({'n2': 2})  # won't error, silent failure

        Note
        ----
        This feature is tested well, but not exhaustively. If in doubt, simply
        make a new scattering object. For assurance that `update()` is correct,
        assert equality between outputs of updated and newly-instantiated objects.
        """
        for name, value in kwargs.items():
            if name not in ScatteringBase1D.DYNAMIC_PARAMETERS:
                warnings.warn(f"`{name}` is not a dynamic parameter, changing "
                              "it may not have intended effects.")
            setattr(self, f'_{name}', value)
        self.rebuild_for_reactives()

    # docs ###################################################################
    _doc_class = \
        r"""
        The 1D scattering transform.

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
        $S_J^{{(0)}} x$ and $S_J^{{(1)}} x$; see `DYNAMIC_PARAMETERS`.
        {frontend_paragraph}
        Given an input `{array}` `x` of shape `(B, N)`, where `B` is the
        number of signals to transform (the batch size) and `N` is the length
        of the signal, we compute its scattering transform by passing it to
        the `scattering` method (or its alias, `__call__`{alias_name}), also
        see its docs.

        Example
        -------
        ::

            # Set the parameters of the scattering transform
            J = 9
            N = 2 ** 13
            Q = 8

            # Generate a sample signal
            x = np.random.randn(N)

            # Define a Scattering1D object
            sc = Scattering1D(N, J, Q)

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
        per octave by default, but can be set like `Q = (8, 2)`.

        {parameters}

        {attributes}

        References
        ----------
        This is a modification of `kymatio/scattering1d/frontend/base_frontend.py`
        in https://github.com/kymatio/kymatio/blob/0.3.0/
        Kymatio, (C) 2018-present. The Kymatio developers.
        """

    _doc_params = \
        r"""
        Parameters
        ----------
        shape : int
            The length of the input signals.

        J : int >= 1 / tuple[int]
            Controls the maximum dyadic scale of the scattering transform, hence
            also the total number of wavelets. The maximum width of any filter is
            `2**J`, not to be confused with maximum support.

            Tuple sets `J1` and `J2` separately, for first-order and second-order
            scattering, respectively.

            Defaults to `ceil(log2(shape)) - 3`. It's recommended to not exceed
            `ceil(log2(shape)) - 2`, as the largest scale wavelets will derive
            most information from padding rather than signal, or be distorted if
            there's not enough padding.

            **Extended description:**

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
                - "width vs support" - see `scattering_filter_factory()` in
                  `wavespin/scattering1d/filter_bank.py`.
                - See "Parameter Sweeps" in `examples/` or
                  https://wavespon.readthedocs.io/en/latest/examples

        Q : int >= 1 / tuple[int]
            Controls the number of wavelets per octave, and their frequency
            resolution. If tuple, sets `Q = (Q1, Q2)`, where `Q1` and `Q2` control
            behaviors in first and second order respectively.

                - Q1: For audio signals, a value of `>= 12` is recommended in
                  order to separate partials.
                - Q2: Recommended `1` for most applications. Higher values are
                  recommended only for `Q1 <= 10`.

            Defaults to `(8, 1)`.

            **Extended description:**

                - Greater `Q` <=> greater frequency resolution <=> greater scale.
                - `Q`, together with `r_psi`, sets the "quality factor" of
                  wavelets, i.e. `(center freq) / bandwidth`. Literature sometimes
                  incorrectly states this same "Q" to be the quality factor.
                - Higher `Q1` is better for multi-component signals (many AM/FM),
                  at expense of few-component signals with rapid AM/FM variations.
                - `Q2` is the `Q1` of amplitude modulations produced by `Q1`'s
                  scalogram. Higher is better if we expect the AMs themselves
                  to be multi-component. Not to be confused with the *signal*
                  having multi-component AM; higher `Q2` only benefits if the
                  *transform* extracts multi-component AM. Otherwise, `Q2=1` is
                  ideal for single-component AM, and likelyhood of more components
                  strongly diminishes with greater `Q1`.
                - See "Parameter Sweeps" in `examples/` or
                  https://wavespon.readthedocs.io/en/latest/examples

        T : float > 0 / str['global']
            Temporal width of the low-pass filter, controlling amount of imposed
            time-shift invariance and maximum subsampling.
            `'global'` for global average pooling (simple arithmetic mean),
            which is faster and eases on padding (ignores `phi_f`'s requirement).

            See `sigma0` for a description on how exactly `T` quantifies
            "invariance". In addition to this "information sense", there's
            invariance in sense of Euclidean distances, that's not
            straightforwardly measured by `T`.

            **`T='global'` note**:

            Recommended to use `max_pad_factor <= 1` for more accurate energy
            normalization, particularly with `pad_mode = 'zero'`. This is because
            outputs in padded regions for such large paddings are generally
            ill-behaved. In zero-padded case, the mean's divisor is inflated
            (`M` in `sum(x) / M`) for more coefficients^1. In non-zero padded
            case, we're including more output points not derived from original
            input (though this can be acceptable, as with `T != 'global'`).

              - 1: it's always "inflated" for non-largest coefficients in that the
                support of smallest-wavelet coefficients is much smaller than that
                of largest-wavelet coefficients, but that's by design, as we seek
                to average by the same filter for all coefficients. But if the
                largest filters are bigger than the input, then the "targetted"
                averaging support exceeds input size (also, we're dividing by a
                greater factor than otherwise only per a few wavelets).

            For detailed discussion, see `_compute_energy_correction_factor` in
            `scat_utils_jtfs.py`

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
            Changing rebuilds `compute_graph` for time scattering.

            Note, "greater than `1` should never be needed" isn't entirely
            correct, due to an unfixed Kymatio mistake. In short, it's accurate
            for `Q1 >= 8`. See
            `wavespin.utils.measures.compute_max_dyadic_subsampling`.

        out_type : str
            Output format:

                - `'array'`: `{array}` of shape `(B, C, N_sub)`, where
                  `N_sub` is coefficient temporal length after subsampling,
                  and `C` is the number of scattering coefficients.
                - `'list'`: list of dictionaries, of length `C`.
                  Each dictionary contains a coefficient, keyed by `'coef'`, and
                  the meta of filter(s) that produced it, e.g. `'j'` (scale)
                  and `'n'` (index, as in `psi1_f[n]`).
                  Each coefficient is shaped `(B, 1, N_sub)`.

            Defaults to `'array'`.

            Can be changed after instantiation. See `DYNAMIC_PARAMETERS` doc.

        pad_mode : str / function
            Name of padding scheme to use, one of (`x = [1, 2, 3]`):

                - `'zero'`:    `[0, 0, 0, 1, 2, 3, 0, 0]`
                - `'reflect'`: `[2, 3, 2, 1, 2, 3, 2, 1]`

            Or, pad function with signature `pad_fn(x, pad_left, pad_right)`.
            This sets `self.pad_mode='custom'` (the name of padding is used
            for some internal logic).
            Defaults to `'reflect'`.

            Can be safely changed after instantiation IF the original `pad_mode`
            wasn't `'zero'`. `'zero'` pads less than any other padding, so
            changing from `'zero'` to anything else risks incurring boundary
            effects.
            See `DYNAMIC_PARAMETERS` and `REACTIVE_PARAMETERS` docs.

        smart_paths : float / tuple[float, int] / str['primitive']
            Threshold controlling maximum energy loss guaranteed by the
            Smart Scattering Paths algorithm. Higher = more coefficients excluded,
            but more information lost.

            The guarantee isn't absolute but empirical. Tuple, like `(.01, 1)`,
            controls the degree of confidence:

                - `0`: liberal. Reasonably safe.
                  Medium-sized survey consisting of audio and seizure iEEG
                  datasets didn't exceed this.
                - `1`: conservative. Very safe.
                  Roughly interprets as 1 in 10 million chance of violating the
                  bound for a general signal. The odds of getting such a signal
                  as a WGN realization are much lower.
                - `2`: adversarial-practical. Extremely safe.
                  Derived by maximizing the energy loss via gradient descent,
                  with very loose constraints to avoid practically impossible
                  edge cases. Not practically useful.
                - `3`: adversarial. 2 but completely unconstrained.
                  Practically useless, but has uses for debugging.

            If we're allowed to fit the entire dataset, we can get "level 4" with
            *more* paths excluded via `wavespin.toolkit.fit_smart_paths()`.

            `'primitive'` excludes coefficients if `j2 <= j1`, a criterion
            used by some libraries (e.g. Kymatio). This is a poor criterion,
            provided only for reference.

            `0` to disable. Will still exclude `j2==0` paths.

            Defaults to `(.01, 1)`.
            See `help(wavespin.smart_paths_exclude)` for an extended description.
            For full control, set to `0` and set desired output of
            `smart_paths_exclude()` to `paths_exclude`.

        max_order : int
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.

        vectorized : bool / int[0, 1, 2] / None
            `True` batches operations (does more at once), which is much faster
            but uses more peak memory (highest amount used at any given time).

            `False` uses for-loops.

            `2` should be tried in case `True` fails. It's `True` without a
            memory-heavy step, which still preserves most of the speedup.

            Defaults to `True` for non-JTFS; does nothing for JTFS (its
            main compute is always vectorized).

        kwargs : dict
            Keyword arguments controlling advanced configurations, passed via
            `**kwargs`.

            See `help(Scattering1D.SUPPORTED_KWARGS)`.
            These args are documented below.

        normalize : str / tuple[str]
            Tuple sets first-order and second-order separately, but only the
            first element sets `normalize` for `phi_f`. str sets both orders same.
            Supported:

                - `'l1'`: bandpass normalization; all filters' amplitude
                  envelopes sum to 1 in time domain (for Morlets makes them peak
                  at 1 in frequency domain). `sum(abs(psi)) == 1`.
                - `'l2'`: energy normalization; all filters' energies are 1
                  in time domain; not suitable for scattering.
                  `sum(abs(psi)**2) == 1`. *Strongly discouraged*.
                - `'l1-energy'`, `'l2-energy'`: additionally renormalizes the
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
            Controls the redundancy of the filters (the larger `r_psi`, the larger
            the overlap between adjacent wavelets), and stability against
            time-warp deformations (larger `r_psi` improves it).
            Defaults to `sqrt(0.5)`. Must be >0 and <1.

            Tuple specifies first and second orders separately. If float,
            only sets first order, and preserves the default for second order.

            Changing default `r_psi` for `Q=1` (for either first or second order)
            is discouraged: greater makes filter bandwidths too great and
            yields aliasing and distortions in both time and frequency domains;
            lower under-tiles the frequency domain; `smart_paths` isn't tuned
            for non-default `r_psi2`.

        max_pad_factor : int >= 0 (default 1) / None
            Will pad by at most `2**max_pad_factor` relative to `nextpow2(shape)`.

            E.g. if input length is 150, then maximum padding with
            `max_pad_factor=3` is `256 * (2**3) = 2048`. `None` means limitless.

            This quantity is best left untouched. If there are boundary effects,
            they should be dealt with by lowering `J` or `T` instead
            (note `T=='global'` ignores `T`'s pad requirement). Otherwise,
            large-scale coefficients are "air-packed" and uninformative.

            The maximum realizable value is `4`: a Morlet wavelet of scale `scale`
            requires `2**(scale + 4)` samples to convolve without boundary
            effects, and with fully decayed wavelets - i.e. x16 the scale,
            and the largest permissible `J` or `log2_T` is `log2(N)`.

            **Notes**:

                - `None`: a limitation with `analytic=True` is,
                  `compute_minimum_support_to_pad` does not account for it.
                - `None`: a general limitation is not accounting for spatial
                  expansiveness of convolution, which is relevant past
                  first-order scattering. See "Limitations" in
                  `compute_minimum_support_to_pad` in
                  `wavespin\scattering1d\scat_utils.py`.
                - non-`None`, with `max_pad_factor=1`, a warning isn't thrown
                  if the ideal threshold is exceeded by less than 3%. This is
                  treated as a special case, as the default `J` exceeds by under
                  1% and the warning isn't worth it.

        analytic : bool (default True)
            `True` enforces strict analyticity by zeroing wavelets' negative
            frequencies. Required for accurate AM/FM extraction.

            `True` worsens time decay for very high and very low frequency
            wavelets. Except for rare cases, `True` is strongly preferred, as
            `False` can extract AM and FM where none exist, or fail to extract,
            and the loss in time localization is mostly drowned out by lowpass
            averaging.

        paths_exclude : dict[str: list[int]] / dict[str: int] / None
            Will exclude coefficients with these paths from computation and
            output.
            Supported keys: `'n2'`, `'j2'`. E.g.:

                - `{'n2': [2, 3, 5], 'j2': 1}`
                - `{'n2': [0, -1], 'j2': [-3, 0]}`

            Negatives wrap around like indexing, e.g. `j2=-1` means `max(j2s)`.
            `dict[str: int]` will convert to `dict[str: list[int]]`.
            `n2=3` means will exclude `self.psi2_f[3]`.
            `j2=3` excludes all `p2f = self.psi2_f` with `p2f['j'] == 3`.

            For speed or memory purposes, it's recommended to tune `smart_paths`
            instead. `paths_exclude` can be used alongside, and doesn't
            affect the results of, `smart_paths`.

            Can be changed after instantiation. However, must be formated
            correctly, i.e. `dict[str: list[int]]`, e.g. `{'n2': [1]}`, so
            `{'n2': 1}` won't work. See `DYNAMIC_PARAMETERS` doc.
            # TODO just formatted correctly or also be an update to existing?
            # i.e. `'n2, n1'`

        precision : str / None
            One of:

                - `'single'`: float32, complex64
                - `'double'`: float64, complex128

            Controls numeric precision at which filters are built and computations
            are done. Will automatically cast input `x` to this precision.

            Defaults to `'double'` for NumPy backend, otherwise to `'single'`.
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
            One of supported padding modes: `'reflect'`, `'zero'` - or `'custom'`
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

        N_scale : int
            `== ceil(log2(N))`. Dyadic scale of the input, and the shortest
            possible filterbank that can be used for temporal scattering.

        log2_T : int
            `== floor(log2(T))`. Dyadic subsampling factor of temporal scattering.

        paths_include_n2n1 : dict[int: list[int]]
            Specifies second-order paths that will be computed, accounting
            for `paths_exclude` and others.
            See `Scattering1D.build_paths_include_n2n1()`.

            `(5 in paths_include_n2n1[3])` means `(n2, n1) = (3, 5)` is computed.

        vectorized_early_U_1 : bool
            == `vectorized and vectorized != 2`.

            Controls whether a memory-intensive speedup is done in `scattering1d`,
            thereby whether `psi1_f_stacked` is constructed.

        psi1_f_stacked : tensor / None
            Concatenation of all first-order filters, shaped
            `(len(psi1_f), 2**J_pad)`. Done if `vectorized` is True.

        cwt_unpad_indices : dict[int: int]
            `ind_start, ind_end = cwt_unpad_indices[hop_size]`.

        max_invertible_hop_size : int
            Max `hop_size` for which `cwt(x, hop_size)` is invertible.
            Note: this follows STFT's NOLA logic, which may not be correct (found
            no source on strided CWT inversion), but it's a good reference.

        max_low_alias_hop_size : int
            Max `hop_size` for which the scalogram, or `abs(cwt(x, hop_size))`,
            has low aliasing. The computation of this quantity isn't ideal but
            still a fair ballpark.

            This quantity will be drastically different from
            `max_invertible_hop_size`. Indeed it's a similar story for STFT.

        compute_graph : list[dict]
            Used in `scattering1d`, specifies which paths to compute and
            how to group them for `vectorized=True`.

        sigma0 : float
            Controls the definition of `'scale'` and `'width'` for all filters.
            Controls alias error in subsampling after lowpass filtering, and
            the number of CQT wavelets.

            Specifically, it's the continuous-time width parameter of the lowpass
            filter when `width = 1`. Any other width has `sigma = sigma0 / width`
            in frequency. For `T`, this means `sigma0 / T`, and the largest scale
            wavelet has `sigma = sigma0 / 2**J`.

            This quantity's default is set such that subsampling a signal
            lowpassed by the lowpass filter with `sigma = sigma0 / T`, then
            subsampled by `T`, is approximately unaliased, i.e. alias is under
            the default `criterion_amplitude`. See
            `wavespin.utils.measures.compute_bandwidth`.

            Configurable via `wavespin.CFG`.
            Defaults to `0.13`.

            **Definition of invariance**:

            In addition to the two aforementioned definitions, reproduced below,
            there's a third sense of invariance, which is an extension or
            restatement of the second:

                1. Euclidean distance is lower with greater `T`, for any given
                   time-shift `dT` of `x`, for `dT < T`. # TODO stability
                   also must have size of deformation bounded?

                   - Misconception: "scale of invariance", in sense of
                     "distance is low as long as shifts are less than `T`".
                     No such thing. What we do have is the warp-stability theorem,
                     which says 1) the distance is upper-bounded for all `x`,
                     2) the bound grows with the size of deformation - in this
                     case, shifts, which are a class of warps.
                     If there were a "scale of invariance", it'd contradict
                     the motivation of the stability theorem: we do not desire
                     invariance to warps.

                2. Variability across a duration lesser than T is killed*.
                   (Greater T means greater lossless subsampling, meaning fewer
                   samples represent the same variation. So x16 subsampling means,
                   pre-subsampling, 16 samples don't represent any variation that
                   1 sample can't)  # TODO

        P_max : int >= 1
            Maximal number of periods to use to make ensure that the Fourier
            transform of the filters is periodic. `P_max = 5` is more than enough
            for double precision.

            The goal's to make the sampling grid long enough for the wavelet to
            decay below machine epsilon (`eps`), yielding an ~unaliased wavelet.
            For example, large `sigma` wavelets need greater `P_max`.

            Configurable via `wavespin.CFG`.
            Defaults to `5`.

        eps : float
            Required machine precision for periodization in filter construction
            (single floating point is enough for deep learning applications).

            Configurable via `wavespin.CFG`.
            Defaults to `1e-7`.

        criterion_amplitude : float
            The overarching precision parameter. Controls

                 - Boundary effects: error due to.
                   See `wavespin.utils.measures.compute_spatial_support`.

                 - Filter decay: sufficiency of.
                   See `wavespin.utils.measures.compute_spatial_support`.

                 - Aliasing: amount of, due to subsampling.
                   See `wavespin.utils.measures.compute_max_dyadic_subsampling`.
                   See `wavespin.utils.measures.compute_bandwidth`.

                 - Filter meta: `'j'`, `'width'`, `'support'`, `'scale'`,
                   `'bw'`, `'bw_idxs'`. See `scattering_filter_factory` in
                   `wavespin.scattering1d.filter_bank`.

            May replace `sigma0`, `P_max`, and `eps` in the future.

            Configurable via `wavespin.CFG`.
            Defaults to `1e-3`.

        halve_zero_pad : bool (default True)
            Whether to halve the pad requirement for `pad_mode='zero'` and
            `pad_mode_fr='zero'` (JTFS). Relevant if seeking to completely
            eliminate (circular) boundary effects, in which case it should be
            `False` and paired with `max_pad_factor=None`.
            See "Limitations" in `compute_minimum_support_to_pad` in
            `wavespin/scattering1d/scat_utils.py`.

            Additionally, for large `J` or `T` (making filters' support exceed
            input's), while boundary effects may be minimal with `False`, filter
            distortion less so. All these effects were judged acceptable in
            practice, hence the default.

        DYNAMIC_PARAMETERS : set[str]
            Names of parameters that can be changed after object creation.
            These only affect the runtime computation of coefficients (no need
            to e.g. rebuild filterbank).

            *Note*, not all listed parameters are always safe to change, refer
            to the individual parameters' docs for exceptions.

        REACTIVE_PARAMETERS : set[str]
            These are `DYNAMIC_PARAMETERS` that can only be updated through
            `self.update()`.
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
        This is a modification of `kymatio/scattering1d/frontend/base_frontend.py`
        in https://github.com/kymatio/kymatio/blob/0.3.0/
        Kymatio, (C) 2018-present. The Kymatio developers.
        """

    _doc_cwt = \
        """
        Apply the Continuous Wavelet Transform, with hop support.

        Reuses the first-order scattering filterbank to do CWT. Note, first-order
        scattering is just CWT with modulus and averaging. As with scattering,
        the lowest frequencies are ~STFT, the rest is CQT - see `self.info()`.

        Some constructor arguments (e.g. `out_type`) have no effect, others are
        only partial, described below.

        Parameters
        ----------
        x : {array}
            An input `{array}` of size `(B, N)` or `(N,)`.
            `B` is integer.

        hop_size : int
            Hop, or convolution stride, or temporal subsampling factor, same as
            in STFT. Identically, `cwt(x)[..., ::hop_size]` (for non-dyadic `N`,
            within an unpad offset/length).

            Must wholly divide `len(x_padded)`, i.e.
            `(2**self.J_pad / hop_size).is_integer()` must be `True`. To see
            all supported values, run `list(self.cwt_unpad_indices)` (`self`
            meaning `sc` in below example).

            Defaults to `1`, i.e. standard CWT.
            See `max_invertible_hop_size` and `max_low_alias_hop_size`.

        Returns
        -------
        S : tensor
            CWT of `x` with `hop_size` stride.

        Example
        -------
        ::

            N = 2 ** 13
            x = np.random.randn(N)
            sc = Scattering1D(N, Q=8, J=6)

            out0 = sc.cwt(x)[..., ::8]
            out1 = sc.cwt(x, 8)
            assert np.allclose(out0, out1)

        Note, above example won't always work due to alignment-related effects
        of stride in unpadding left-right-padding; it works if `N` and `hop_size`
        are powers of 2. This isn't a "problem".

        Ineffective parameters
        ----------------------
        `T`, `average`, `oversampling`, `out_type`, `smart_paths`, `max_order`,
        `paths_exclude`.

        `out_type` is forced to `'array'`. If calling from JTFS, all JTFS-only
        parameters are ineffective.

        Partial parameters
        ------------------
        For `J`, `Q`, `r_psi`, and `normalize`, only their first-order specs
        are effective (`J1`, etc).
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
        cls.cwt.__doc__ = ScatteringBase1D._doc_cwt.format(
            array=cls._doc_array)


class TimeFrequencyScatteringBase1D():
    DEFAULT_KWARGS_JTFS = dict(
        aligned=None,
        out_3D=False,
        sampling_filters_fr=('exclude', 'resample'),
        analytic_fr=True,
        F_kind='average',
        max_pad_factor_fr=1,
        pad_mode_fr='zero',
        normalize_fr='l1-energy',
        r_psi_fr=math.sqrt(.5),
        oversampling_fr=0,
        do_energy_correction=True,
        max_noncqt_fr=None,
        out_exclude=None,
        paths_exclude=None,
    )
    SUPPORTED_KWARGS_JTFS = tuple(DEFAULT_KWARGS_JTFS)
    DYNAMIC_PARAMETERS_JTFS = {
        'oversampling', 'oversampling_fr', 'out_type', 'out_exclude',
        'paths_exclude', 'pad_mode', 'pad_mode_fr',
    }
    REACTIVE_PARAMETERS_JTFS = {
        'oversampling', 'oversampling_fr', 'out_exclude', 'paths_exclude',
        'pad_mode_fr'
    }

    def __init__(self, J_fr=None, Q_fr=1, F=None, average_fr=False,
                 out_type='array', smart_paths=.007, vectorized_fr=True,
                 implementation=None, **kwargs):
        # set args while accounting for special cases
        args = dict(
            J_fr=J_fr,
            Q_fr=Q_fr,
            F=F,
            average_fr=average_fr,
            out_type=out_type,
            smart_paths=smart_paths,
            vectorized_fr=vectorized_fr,
            implementation=implementation
        )
        for name, value in args.items():
            _setattr_and_handle_reactives(
                self, name, value,
                TimeFrequencyScatteringBase1D.REACTIVE_PARAMETERS_JTFS)
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
        if self._no_second_order_filters:  # no-cov
            raise ValueError("configuration yields no second-order filters; "
                             "try increasing `J`")

        # set standardized pair order to be followed by the API
        self._api_pair_order = ('S0', 'S1',
                                'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
                                'psi_t * psi_f_up', 'psi_t * psi_f_dn')

        # handle `kwargs`, `implementation` ##################################
        # validate
        if self.implementation is not None:
            if not (isinstance(self.implementation, int) and
                    self.implementation in range(1, 6)):  # no-cov
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
            _setattr_and_handle_reactives(self, name, I.pop(name),
                                          self.reactives_jtfs)

        # invalid arg check
        if len(I) != 0:  # no-cov
            supported = (
                inspect.getfullargspec(ScatteringBase1D).args +
                list(ScatteringBase1D.SUPPORTED_KWARGS) +
                inspect.getfullargspec(TimeFrequencyScatteringBase1D).args +
                list(TimeFrequencyScatteringBase1D.SUPPORTED_KWARGS_JTFS)
            )
            supported = [nm for nm in supported if nm != 'self']
            raise ValueError("unknown args: {}\n\nSupported are:\n{}".format(
                list(I), supported))

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
                    sampling_filters_fr=('exclude', 'recalibrate'),
                    out_type='dict:list'),
        }
        # override arguments with presets
        if isinstance(self.implementation, int):
            for k, v in self._implementation_presets[self.implementation].items():
                _setattr_and_handle_reactives(self, k, v, self.reactives_jtfs)

        # handle `configs.py` that need handling here ########################
        # N_fr stuff
        if CFG['JTFS']['N_fr_p2up'] is None:
            self.N_fr_p2up = bool(self.out_3D)
        else:  # no-cov
            self.N_fr_p2up = CFG['JTFS']['N_fr_p2up']
        self.N_frs_min_global = CFG['JTFS']['N_frs_min_global']

        # energy correction stuff
        self.do_ec_frac_tm = CFG['JTFS']['do_ec_frac_tm']
        self.do_ec_frac_fr = CFG['JTFS']['do_ec_frac_fr']

        # --------------------------------------------------------------------
        # `out_structure`
        if isinstance(self.implementation, int) and self.implementation in (3, 5):
            self.out_structure = 5
        else:
            self.out_structure = None

        # handle `out_exclude`
        if self.out_exclude is not None:
            if isinstance(self.out_exclude, str):  # no-cov
                self._out_exclude = [self.out_exclude]
            # ensure all names are valid
            supported = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                         'psi_t * phi_f', 'psi_t * psi_f_up', 'psi_t * psi_f_dn')
            for name in self.out_exclude:
                if name not in supported:
                    raise ValueError(("'{}' is an invalid coefficient name; "
                                      "must be one of: {}").format(
                                          name, ', '.join(supported)))

        # handle `F`; this is processed further in `scf`
        if self.F is None:
            # default to one octave (Q wavelets per octave, J octaves,
            # approx Q*J total frequency rows, so averaging scale is `Q/total`)
            # F is processed further in `_FrequencyScatteringBase1D`
            self.F = self.Q[0]

        # handle `vectorized_fr`
        if self.vectorized_fr is None:
            if self.vectorized:
                self.vectorized_fr = 2
        self.vectorized_early_fr = bool(self.vectorized_fr and
                                        self.vectorized_fr != 2)

        # handle `max_noncqt_fr`
        if self.max_noncqt_fr is not None:
            if not isinstance(self.max_noncqt_fr, (str, int)):  # no-cov
                raise TypeError("`max_noncqt_fr` must be str, int, or None, "
                                "got %s" % type(self.max_noncqt_fr))
            if self.max_noncqt_fr == 'Q':
                self.max_noncqt_fr = self.Q[0] // 2

        # `paths_exclude`, `smart_paths` (need earlier for N_frs)
        if not isinstance(self.paths_exclude, dict):
            self._paths_exclude = {}
        _handle_smart_paths(self.smart_paths, self._paths_exclude,
                            self.psi1_f, self.psi2_f, self.r_psi)
        self.handle_paths_exclude_jtfs(n1_fr=False)

        # store info about argument routing ##################################
        not_passed_from_self = ('self', 'paths_include_build', 'N_frs',
                                'max_order_fr')
        scf_args_all = inspect.getfullargspec(_FrequencyScatteringBase1D).args
        # see `args_meta` docs
        self._args_meta = {
            # this is what's fetched from `self` and passed to `scf`'s constructor
            'passed_to_scf': tuple(
                name for name in scf_args_all
                if name not in not_passed_from_self
            ),
            # these are arguments that belong at top level, but `scf` happens to
            # use them, and rerouting from `scf` isn't needed
            'top_level': (
                'precision', 'backend',
            ),
            # these are helpful `scf` extras exposed to top level
            'top_level_scf_extras': (
                'psi1_f_fr_up', 'psi1_f_fr_dn', 'phi_f_fr',
            ),
            # set below
            'only_scf': (),
        }
        self._args_meta['only_scf'] = tuple(
            a for a in self._args_meta['passed_to_scf']
            if a not in self._args_meta['top_level'])
        # validate `fr_attributes`
        assert sorted(self.fr_attributes) == sorted(
            self.args_meta['only_scf'] + self.args_meta['top_level_scf_extras'])

        # frequential scattering object ######################################
        paths_include_build, N_frs = self.build_scattering_paths_fr()
        # number of psi1 filters
        self._n_psi1_f = len(self.psi1_f)
        max_order_fr = 1

        # pack args & create object
        args_not_from_self = dict(paths_include_build=paths_include_build,
                                  N_frs=N_frs, max_order_fr=max_order_fr)
        args_from_self = {
            k: getattr(self,
                       k if (k not in self.reactives_jtfs) else '_' + k
                       )
            for k in self.args_meta['passed_to_scf']
        }
        self.scf = _FrequencyScatteringBase1D(**args_not_from_self,
                                              **args_from_self)

        # further steps
        self.finish_creating_filters()
        self.handle_paths_exclude_jtfs(n1_fr=True)
        # update time graphs
        self.build_paths_include_n2n1()
        self._compute_graph = build_compute_graph_tm(self)

        # finish processing args #############################################
        # detach `__init__` args, instead access `scf`'s via `__getattr__`.
        # this is so that changes in attributes are reflected here
        for init_arg in self.args_meta['only_scf']:
            if init_arg in self.reactives_jtfs:
                init_arg = '_' + init_arg
            delattr(self, init_arg)

        # build compute graph ################################################
        self._compute_graph_fr = build_compute_graph_fr(self)
        # handle `vectorized_fr`
        if self.vectorized_fr:
            # group frequential filters by realized subsampling factors
            self.psi1_f_fr_stacked_dict, self.scf.stack_ids = (
                make_psi1_f_fr_stacked_dict(self.scf, self.paths_exclude))
        _handle_device_non_filters_jtfs(self)
        self.pack_runtime_filters()

        # sanity checks ######################################################
        # meta
        try:
            self.meta()
        except:  # no-cov
            warnings.warn(("Failed to build meta; the implementation may be "
                           "faulty. Try another configuration, or call "
                           "`jtfs.meta()` to debug."))

        # paths
        for n2 in range(len(N_frs)):
            if n2 in self.paths_include_n2n1:
                assert N_frs[n2] == len(self.paths_include_n2n1[n2])
                assert N_frs[n2] == sum(
                    map(len, self.compute_graph['U_12_dict'][n2].values()))

        # args_meta
        for name in not_passed_from_self:
            if name == 'self':
                continue
            assert name in args_not_from_self, (name, args_not_from_self)

    def build_scattering_paths_fr(self):
        """Determines all (n2, n1) pairs that will be scattered, not accounting
        for `paths_exclude` except for `paths_exclude['n2, n1']`.

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
                            "Try raising `J`, changing `Q`, or setting "
                            "`max_noncqt_fr=None`.")

        # n1s increment by 1, and n1==0 is present
        for n2, n1s in paths_include_build.items():
            assert np.all(np.diff(n1s) == 1), n1s
            assert n1s[0] == 0, n1s

        return paths_include_build, N_frs

    def build_paths_include_n2n1(self):
        """Overrides `Scattering1D`'s method, instead working through
        `build_scattering_paths_fr`.

        It's same as `paths_include_build`, except that it accounts for
        `'n2'` and `'j2'` in `paths_exclude`.
        """
        self.build_paths_include_n2n1_jtfs()

    def build_paths_include_n2n1_jtfs(self):
        self._paths_include_n2n1 = deepcopy(self.paths_include_build)

        for n2, p2f in enumerate(self.psi2_f):
            j2 = p2f['j']

            if (n2 in self.paths_exclude['n2'] or
                    j2 in self.paths_exclude['j2']):
                self._paths_include_n2n1.pop(n2, [])

    def handle_paths_exclude_jtfs(self, n1_fr=False):
        supported = {'n2', 'n1_fr', 'j2', 'j1_fr'}

        # n2
        if not n1_fr:
            j_all = [p['j'] for p in self.psi2_f]
            n_psis = len(self.psi2_f)
            self._paths_exclude = _handle_paths_exclude(
                self._paths_exclude, j_all, n_psis, supported, names=('n2', 'j2')
            )
        # n1_fr
        if n1_fr:
            j_all = self.scf.psi1_f_fr_up['j'][0]
            n_psis = len(self.scf.psi1_f_fr_up[0])
            _handle_paths_exclude(
                self._paths_exclude, j_all, n_psis, supported,
                names=('n1_fr', 'j1_fr'))

    def finish_creating_filters(self):
        """Handles necessary adjustments in time scattering filters unaccounted
        for in default construction. Doesn't finish all filter-related builds.
        """
        # ensure phi is subsampled up to log2_T for `phi_t * psi_f` pairs
        max_sub_phi = lambda: max(k for k in self.phi_f if isinstance(k, int))
        while max_sub_phi() < self.log2_T:
            new_sub = max_sub_phi() + 1
            self.phi_f[new_sub] = fold_filter_fourier(
                self.phi_f[0], nperiods=2**new_sub)

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
                # (assuming no aliasing, otherwise we still reproduce the
                # sampling of `gauss_1d`)
                phi_f[trim_tm] = [v[::2**trim_tm] for v in phi_f[0]]
        self.phi_f = phi_f

        # adjust padding (redundant-seeming dict-comp to make a copy)
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

    def scattering(self, x):
        # input checks #######################################################
        _check_runtime_args_common(x)
        _check_runtime_args_jtfs(self.average, self.average_fr, self.out_type,
                                 self.out_3D)
        x, batch_shape, backend_obj = _handle_input_and_backend(self, x)

        # scatter, postprocess, return #######################################
        Scx = timefrequency_scattering1d(
            x, self.compute_graph, self.compute_graph_fr,
            self.scattering1d_kwargs,
            **{arg: getattr(self, arg) for arg in (
                'backend', 'J', 'log2_T', 'psi1_f', 'psi2_f', 'phi_f', 'scf',
                'pad_mode', 'pad_left', 'pad_right',
                'ind_start', 'ind_end',
                'oversampling',
                'average', 'average_global', 'average_global_phi',
                'out_type', 'out_exclude', 'paths_exclude',
                'vectorized', 'api_pair_order', 'do_energy_correction',
            )}
        )
        Scx = self._post_scattering(Scx, batch_shape, backend_obj)
        return Scx

    def _post_scattering(self, Scx, batch_shape, backend_obj):
        """Split from `scattering` for testing purposes."""
        if self.out_structure is not None:
            Scx = pack_coeffs_jtfs(
                Scx, self.meta(), structure=self.out_structure,
                separate_lowpass=True,
                sampling_psi_fr=self.scf.sampling_psi_fr, out_3D=self.out_3D)

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
            See `help(wavespin.scattering1d.scat_utils_jtfs.compute_meta_jtfs)`.
        """
        return compute_meta_jtfs(
            self.scf,
            **{arg: getattr(self, arg) for arg in (
                'psi1_f', 'psi2_f', 'phi_f', 'log2_T', 'sigma0',
                'average', 'average_global', 'average_global_phi', 'oversampling',
                'out_type', 'out_exclude', 'paths_exclude', 'api_pair_order',
            )}
        )

    # Properties #############################################################
    # Read-only attributes ---------------------------------------------------
    @property
    def compute_graph_fr(self):
        return self._compute_graph_fr

    @property
    def paths_include_build(self):
        return self.scf.paths_include_build

    @property
    def fr_attributes(self):
        """Exposes `scf`'s attributes via main object."""
        # this is validated in `build()`
        return ('J_fr', 'Q_fr', 'F', 'average_fr', 'aligned', 'oversampling_fr',
                'sampling_filters_fr', 'out_3D', 'max_pad_factor_fr',
                'pad_mode_fr', 'analytic_fr', 'max_noncqt_fr', 'normalize_fr',
                'F_kind', 'r_psi_fr', 'vectorized_fr', 'vectorized_early_fr',
                '_n_psi1_f',
                'psi1_f_fr_up', 'psi1_f_fr_dn', 'phi_f_fr')

    @property
    def args_meta(self):
        return self._args_meta

    @property
    def default_kwargs_jtfs(self):
        """Shorthand."""
        return deepcopy(TimeFrequencyScatteringBase1D.DEFAULT_KWARGS_JTFS)

    @property
    def reactives_jtfs(self):
        """Shorthand."""
        return TimeFrequencyScatteringBase1D.REACTIVE_PARAMETERS_JTFS

    @property
    def api_pair_order(self):
        return self._api_pair_order

    # Reactive attributes & helpers ------------------------------------------
    @property
    def paths_exclude(self):
        return self._paths_exclude

    @paths_exclude.setter
    def paths_exclude(self, value):
        self.raise_reactive_setter('paths_exclude')

    @property
    def out_exclude(self):
        return self._out_exclude

    @out_exclude.setter
    def out_exclude(self):
        self.raise_reactive_setter('out_exclude')

    def raise_reactive_setter(self, name):
        _raise_reactive_setter(name, 'REACTIVE_PARAMETERS_JTFS')

    def rebuild_for_reactives(self):
        # tm
        _handle_pad_mode(self)
        self.build_paths_include_n2n1()
        self._compute_graph = build_compute_graph_tm(self)

        # fr
        _handle_pad_mode_fr(self.scf)
        self.psi1_f_fr_stacked_dict, self.scf.stack_ids = (
            make_psi1_f_fr_stacked_dict(self.scf, self.paths_exclude))
        self._compute_graph_fr = build_compute_graph_fr(self)
        if self.filters_device is not None:
            self.update_filters('psi1_f_fr_stacked_dict')
            _handle_device_non_filters_jtfs(self)
        self.pack_runtime_filters()
        self.deduplicate_stacked_filters()

    def pack_runtime_filters(self):
        self._compute_graph_fr['spin_data'] = pack_runtime_spinned(
            self.scf, self.compute_graph_fr, self.out_exclude)

    def update(self, **kwargs):
        """See `help(wavespin.Scattering1D.update)`."""
        for name, value in kwargs.items():
            if name not in TimeFrequencyScatteringBase1D.DYNAMIC_PARAMETERS_JTFS:
                warnings.warn(f"`{name}` is not a dynamic parameter, changing "
                              "it may not have intended effects.")
            if name in self.args_meta['only_scf']:
                setattr(self.scf, f'_{name}', value)
            else:
                setattr(self, f'_{name}', value)
        self.rebuild_for_reactives()

    # Rerouted to `scf` attributes -------------------------------------------
    @property
    def psi1_f_fr_stacked_dict(self):
        return self.scf.psi1_f_fr_stacked_dict

    @psi1_f_fr_stacked_dict.setter
    def psi1_f_fr_stacked_dict(self, value):
        self.scf.psi1_f_fr_stacked_dict = value
    # ------------------------------------------------------------------------

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
            terminology=cls._doc_terminology,
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
            \psi_\mu^{{(2)}}(t) \psi_{{l}}(+\lambda)$
            spin up bandpass

            $\Psi_{{\mu, l, -1}}(t, \lambda) =
            \psi_\mu^{{(2)}}(t) \psi_{{l}}(-\lambda)$
            spin down bandpass

            $\Psi_{{\mu, -\infty, 0}}(t, \lambda) =
            \psi_\mu^{{(2)}}(t) \phi_F(\lambda)$
            temporal bandpass, frequential lowpass

            $\Psi_{{-\infty, l, 0}}(t, \lambda) =
            \phi_T(t) \psi_{{l}}(\lambda)$
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
        $\psi_{{l}}(+\lambda)$ is like $\psi_\lambda^{{(1)}}(t)$ but with
        its own parameters (center frequency, support, etc), and an anti-analytic
        complement (spin up is analytic).

        Filters are built at initialization. While the wavelets are fixed, other
        parameters may be changed after the object is created, such as `out_type`;
        see `DYNAMIC_PARAMETERS_JTFS`.

        {frontend_paragraph}

        Example
        -------
        ::

            # Set the parameters of the scattering transform
            J = 9
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
        which increases frequency resolution with higher `Q`. The second-order
        wavelets :math:`\psi_\mu^{{(2)}}(t)` have one wavelet per octave by
        default, but can be set like `Q = (8, 2)`.

        Internally, `J_fr` and `Q_fr`, the frequential variants of `J` and `Q`,
        are defaulted, but can be specified as well.

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

            Greater `J2` lowers the minimum captured slope and the longest
            captured AM/FM geometry along time, useful if FDTS spans over
            long durations.

            Important: while `J1 < J2` has good use cases, `J2 < J1` rarely so,
            as per the basic energy flow criterion `j2 >= j1`, more parts of
            the scalogram will be excluded from joint pairs.

        Q : int / tuple[int]
            (Extended docs for JTFS)

              - `Q1`, together with `J`, determines `N_frs_max` and `N_frs`,
                or length of inputs to frequential scattering.
              - `Q2`, together with `J`, determines `N_frs` (via the `j2 >= j1`
                criterion, also `smart_paths`), and total number of joint slices.
              - Greater `Q2` values better capture temporal AM modulations (AMM)
                of multiple rates. Suited for inputs of multirate or intricate AM.
                `Q2=2` is in close correspondence with the mamallian auditory
                cortex: https://asa.scitation.org/doi/full/10.1121/1.1945807
                Also see note in `Scattering1D` docs.

        r_psi : float / tuple[float]
            (Extended docs for JTFS)

            Greater `r_psi1`, while maintaining the same quality factor
            (so also greater `Q1`, see `Q` docs and `self.info()`), reduces
            aliasing in frequential scattering. The extent of aliasing with
            default `r_psi1` is yet to be studied in detail, but it can be
            measured via `oversampling_fr=99` and appropriate comparisons.

            The extent is suspected to be substantial, and `r_psi1` of 0.9
            to 0.96 is recommended, but development time is needed to test
            these.

            If higher `r_psi1` and `Q1` are used, higher `F` is also due, both
            in physical sense (preserve averaging scale) and output size. So, one
            might try increasing `Q1` and `r_psi1` until size doubles, then
            double `F`, unless only the averaging scale matters (downsampling
            is only done in powers of 2, while averaging scale can be tuned
            to any amount).

        J_fr : int >= 1
            `J` but for frequential scattering; see `J` docs in
            `help(wavespin.Scattering1D())`.

            Greater values increase the maximum captured FDTS slope and the
            longest captured structure along log-frequency, useful if signal
            behavior spans multiple octaves.

            Recommended to not set this above the default; see `J` docs.

            Default is determined at instantiation from longest frequential row
            in frequential scattering, set to `log2(nextpow2(N_frs_max)) - 3`,
            i.e. maximum possible minus 3, but no less than 3, and no more than
            max. For predicting `N_frs_max`, see its docs.

            In general, `J_fr >= 5` is preferred for tiling completeness of
            quefrencies^1, but `J_fr` should still not exceed
            `log2(nextpow2(N_frs_max)) - 3` (see 2 below).

                - 1: `J_fr >= 5` attains better tiling in lowest quefrencies
                  by working around an issue with filter design
                  (see "Near-DC note" in `test_lp_sum` in `test_jtfs.py`).
                - 2: `N_frs_max` can be increased by increasing `Q`, `J`, and/or
                  `r_psi` (only first-order matters). To preserve the physical
                  characteristics of the (first-order) filterbank (max scale and
                  time-frequency resolution), increase `Q` and `r_psi` together,
                  don't change `J`, and check that quality factor is approx.
                  unchanged via `self.info()`.

        Q_fr : int >= 1
            `Q` but for frequential scattering; see `J` docs in
            `help(wavespin.Scattering1D())`.

            Greater values better capture quefrential variations of multiple
            rates - that is, variations and structures along frequency axis of the
            wavelet transform's 2D time-frequency plane. Suited for inputs of many
            frequencies or intricate AM-FM variations. `2` or `1` should work for
            most purposes.

            If compute burden isn't a concern, `Q_fr=2` should work better in
            general, especially for higher-dim convs or high `F`.

            Default is `1` for speed.

        F : float > 0 / str['global'] / None
            Temporal width of frequential low-pass filter, controlling amount of
            imposed frequency transposition invariance and maximum frequential
            subsampling. Defaults to `Q1`, i.e. one octave.

              - If `'global'`, sets to maximum possible `F` based on `N_frs_max`,
                and pads less (ignores `phi_f_fr`'s requirement).
              - For predicting `N_frs_max`, see its docs.
              - With `average_fr=True`, amounts to reducing the total output size
                by `F` (except for `S0` and `S1` coeffs).
              - Used even with `average_fr=False` (see its docs); this is likewise
                true of `T` for `phi_t * phi_f` and `phi_t * psi_f` pairs.

            For `'global'`, it's recommended to use `max_pad_factor_fr <= 1`,
            for more accurate energy normalization, particularly with
            `pad_mode_fr = 'zero'` (see "`T='global'` note" in `T`'s docs).

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
          NOTE: unlike in `Scattering1D`, the batch dimension is not collapsed
          if `len(x.shape) == 1`.

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
            Preset configuration to use. **Overrides** the following parameters:

                - `average_fr, aligned, out_3D, sampling_filters_fr, out_type`

            **Implementations:**

                1: Standard for 1D convs. `(n1_fr * n2 * n1, t)`.
                  - `average_fr = False`
                  - `aligned = False`
                  - `out_3D = False`
                  - `sampling_psi_fr = 'exclude'`
                  - `sampling_phi_fr = 'resample'`

                2: Standard for 2D convs. `(n1_fr * n2, n1, t)`.
                  - `average_fr = True`
                  - `aligned = True`
                  - `out_3D = True`
                  - `sampling_psi_fr = 'exclude'`
                  - `sampling_phi_fr = 'resample'`

                3: Standard for 3D/4D convs. `(n1_fr, n2, n1, t)`. [2] but
                  - `out_structure = 5`

                4: Alternative for 2D convs. [2] but
                  - `sampling_psi_fr = 'recalibrate'`

                5: Alternative for 1D convs. [1] but
                  - `average_fr = True`
                  - `sampling_phi_fr = 'recalibrate'`

            where

                - `n1`: index of first-order wavelet used to produce the output
                - `n2`: `n1` for second order
                - `n1_fr`: `n1` for frequential scattering
                - Above they're rather used to indicate what's packed along what
                  dimension, where `*` means "unrolling". They don't always
                  interpret as said indices; see "1:" in "Extended description".

            For beginners, `1` or `2` is recommended, or simply `None`. Otherwise,

                - `3`: see "JTFS 2D Conv-Net" example.
                - `4`: more coefficients, but more informative.
                  May improve performance.
                - `5`: much fewer coefficients, with adaptive downsampling
                  to maximize coefficient informativeness.
                  May improve performance.

            **Extended description:**

                - `out_structure` refers to packing output coefficients via
                  `pack_coeffs_jtfs(..., out_structure)`. This zero-pads and
                  reshapes coefficients, but does not affect their values or
                  computation in any way. (Thus, 3==2 except for shape).
                  See `help(wavespin.toolkit.pack_coeffs_jtfs)`.

                - `'exclude'` in `sampling_psi_fr` can be replaced with
                  `'resample'`, which yields significantly more coefficients and
                  doesn't lose information (which `'exclude'` strives to
                  minimize), but is slower and the coefficients are mostly
                  "air-packed" (stemming mostly from zeros) and uninformative.

                - `5` also makes sense with `sampling_phi_fr = 'resample'` and
                  small `F` (small enough to let `J_pad_frs` drop below max),
                  but the argument will only set `'recalibrate'`.

                - 1: E.g. `(n1_fr * n2, ...)` means that indexing dim 0 accesses
                  coefficients stemming from different second-order and
                  frequential filters. Also, for `4` & `5`, `n1` is no longer
                  directly an index, per `aligned=False`; it's a function of `n2`
                  (see `aligned` docs).

            **Advanced use:**

            For 2D+ convolutions, at least for the first layer, grouped
            convolutions should be used. (For an explanation, see
            "Note: `not aligned` && `out_3D`" in `aligned` docs).

                - `n_groups` should equal `n_channels_in` (i.e. length of first
                  non-batch dim).
                - If `sampling_psi_fr != 'exclude'`, then it can also equal
                  the length of the `n1_fr` dimension. Without `paths_exclude`,
                  that's `len(sc.psi1_f_fr_up[0])` (i.e. number of frequential
                  filters).

            1-2 such layers should suffice in most cases. To fully mitigate the
            effect of initial `n1` misalignment,


        vectorized : bool (default True) / int
            JTFS reuses `Scattering1D.scattering`'s computation, minus
            second-order modulus and averaging. Controls its behavior.

            See `help(wavespin.Scattering1D())`.

        vectorized_fr : bool / int / None
            Whether to vectorize frequential scattering. `2` is to not fully
            vectorize the most memory-intensive step.

            Defaults to `2` if `vectorized` is True, else `False`. That's
            because on most devices, excess vectorization can be detrimental.
            `True` is worth trying with GPU, on small inputs.

            Note, some part of frequential scattering is always vectorized.
            If `vectorized_fr=False` doesn't cause much memory issues, odds
            are `vectorized=True` will work fine.

        kwargs : dict
            Keyword arguments controlling advanced configurations, passed via
            `**kwargs`.
            See `help(TimeFrequencyScattering1D.SUPPORTED_KWARGS_JTFS)`.
            These args are documented below.

        aligned : bool / None
            Whether to enforce uniformity of the full 4D JTFS tensor.

                - `True`: rows index to same frequency for all joint slices.
                  Enforces same total frequential stride for all slices.
                - `False` uses stride that maximizes information richness &
                  density.
                - `True` example: `S_2[2][4]` & `S_2[3][4]` correspond to same
                  frequency (fifth row of third and fourth joint slices).

            Defaults to `True` if `out_3D=True`.

            **Short version**:

            Changing this parameter is best suited for advanced users. The
            default will select what generally works best.

            `aligned=False` with `out_3D=True` with 2D or higher convolutions
            may require specialized network design to perform well; see
            "Note: `not aligned` && `out_3D`" below.

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

            See "Compute logic: stride, padding" in `core`, specifically
            `'recalibrate'`.

            **Illustration**:

            Intended usage is `aligned=True` && `sampling_filters_fr='resample'`
            and `aligned=False` && `sampling_filters_fr='recalibrate'`. Below
            example assumes these.

            `x` == zero; `0, 4, ...` == indices of actual (nonpadded) data.
            That is, `x` means the convolution kernel (wavelet or lowpass) is
            centered in the padded region and contains less (or no) information,
            whereas `4` centers at `input[4]`. And `input` is `U1`, so the
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

            **Unpadding with `out_3D=True`**:

                - `aligned=True`: we always have fr stride == `log2_F`, with
                  which we index `ind_start_fr_max` and `ind_end_fr_max`
                  (i.e. take maximum of all unpads across `n2` from this factor
                  and reuse it across all other `n2`).
                - `aligned=False`: decided from `N_fr_scales_max` case, where we
                  compute `unpad_len_common_at_max_fr_stride`. Other `N_fr_scales`
                  use that quantity to compute `min_stride_to_unpad_like_max`.
                  See "Compute logic: stride, padding" in `core`, specifically
                  `'recalibrate'`.

            The only exception is with `average_fr_global_phi and not average_fr`:
            spinned pairs will have zero stride, but `phi_f` pairs will have max.

            **Note: `not aligned` && `out_3D`**:

            Should work best with grouped convolutions. It's recommended at least
            at input layer, with `groups = n_in_channels`^1 (PyTorch), such that
            filters "don't talk". This doesn't fully fix the issues with
            `not aligned` if non-grouped convolutions are used, nor does any
            number of grouped layers, so resuming standard convolutions isn't
            advised unless the spatial misalignment can be addressed another way.
            As this inhibits cross-channel interactions, it inhibits learning
            cross-`xi2`-`xi1_fr` dependencies, so this configuration
            (`not aligned` && `out_3D`) might be net-detrimental.

            In further detail:

                - In standard convolutions, to compute one output channel, a
                  distinct filter operates on every input channel, and their
                  contributions are summed. This is done to compute every output
                  point for that channel. To compute another output channel,
                  another set of `n_in_channels` number of filters, with different
                  weights, is used, with the same procedure. So there are a total
                  of `n_out_channels * n_in_channels` filters, each with size
                  `(kernel_size_h, kernel_size_w)`^2.
                - The implication is, with `aligned=False`, standard convolutions
                  will sum contributions from different `xi1`s (frequency axis of
                  2D JTFS slices), which likely isn't desired. Grouped convs
                  avoid this interaction.
                - More grouped layers help with spatial misalignment, as they can
                  learn to account for it, but it's a distracting learning
                  objective, and misalignment persists through any number of conv
                  layers, only question being how much.
                - More grouped layers also inhibit cross-channel learning, as
                  information doesn't flow between groups.^3

            Footnotes:

                - 1: `groups = len(jtfs.psi1_f_fr_up[0])` is likely to work
                  better, per involving some cross-channel interactions as opposed
                  to none. Works since with `out_3D=True` it's always
                  `aligned=True` on per-`n2` basis.
                - 2: Indeed, the weight tensor is shaped
                  `(n_out_channels, n_in_channels, kernel_size_h, kernel_size_w)`.
                - 3: In best case, there's only flow `xi1_fr`-to-`xi1_fr` on
                  per-`xi2` basis (see "1:"), but not `xi2`-to-`xi2`; in worst,
                  there isn't any flow between joint slices. (An exception is at
                  the final aggregation layer, e.g. global average pooling
                  followed by dense, but that's much more indirect.)

            **Note: `not aligned` && `not out_3D`**:

            This is recommended. Without need for 2D+ structure, there likely
            isn't purpose to alignment, and other benefits can be realized
            without harm.

        out_3D : bool (default False)
            `True` adjusts frequential scattering to enable concatenation along
            joint slices dimension, as opposed to flattening (mixing joint slices
            and frequencies).

                - `True` is recommended with 2D, 3D, 4D use cases.
                - `True` requires `average_fr=True`.

            **Extended description**:

                - `False` can still be concatenated into the "true" JTFS
                  4D structure; see `help(wavespin.toolkit.pack_coeffs_jtfs)`
                  for a complete description. The difference is in how values are
                  computed, especially near boundaries. More importantly, `True`
                  enforces `aligned=True` on *per-`n2`* basis, enabling (grouped)
                  3D convs even  with `aligned=False`
                  (see "Note: `not aligned` && `out_3D`" in `aligned` docs).
                - `False` will unpad freq by exact amounts for each joint slice,
                  whereas `True` will unpad by minimum amount common to all
                  slices at a given subsampling factor to enable concatenation.
                  See `scf_compute_padding_fr()`.
                - See `aligned` for its interactions with `out_3D` (also below).
                - `aligned and out_3D` may sometimes be significantly more
                  compute-intensive than just `aligned`. `aligned and not out_3D`
                  is an alternative worth inspecting with `visuals.viz_jtfs_2d`,
                  as the added compute may sometimes not justify the added
                  information.

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
                input (hence why `'conj-reflect-zero'`), and hence, unpadding by
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

              - `'resample'`: preserve physical dimensionality
                (center frequeny, width) at every length (trimming in time
                domain).
                E.g. `psi = psi_fn(N/2) == psi_fn(N)[N/4:-N/4]`.

              - `'recalibrate'`: recalibrate filters to each length.

                - widths (in time): widest filter is halved in width, narrowest
                  is kept unchanged, and other widths are re-distributed from
                  the new minimum to same maximum.

                - center frequencies: all redistribute between new min and max.

                  - New max is set by halving distance between old max and 0.5
                    (greatest possible), e.g. 0.44 -> 0.47, then 0.47 -> 0.485,
                    etc.
                  - New CQT min stems from applying the widths ratio (`s[1]/s[0]`,
                    `s = sigmas`) upon the new max, `n_cqt` number of times.
                  - New overall min follows the same non-CQT design scheme as
                    regular scattering: tile between `0` and last CQT `xi`.

                - The number of CQT and non-CQT wavelets is preserved.

              - `'exclude`': same as `'resample'` except filters wider than
                `widest / 2` are excluded, then `widest / 4` for next
                `N_fr_scale`, and so on.
                 (More precisely, it's `widest * width_exclude_ratio`, where
                 `width_exclude_ratio` defaults to 0.5 and is a configurable
                 via `wavespin.CFG`.)

            Tuple can set separately `(sampling_psi_fr, sampling_phi_fr)`, else
            both set to same value.

            From an information/classification standpoint:

                - `'resample`' enforces freq invariance imposed by `phi_f_fr` and
                  physical scale of extracted modulations by `psi1_f_fr_up`
                  (& down). This is consistent with scattering theory and is the
                  standard used in existing applied literature.
                - `'recalibrate'` remedies a problem with `'resample'`.
                  `'resample'` calibrates all filters relative to longest input;
                  when the shortest input is very different in comparison, it
                  makes most filters appear lowpass-ish. In contrast,
                  recalibration enables better exploitation of fine structure
                  over the smaller interval (which is the main motivation behind
                  wavelets, a "multi-scale zoom".)
                - `'exclude'` circumvents the problem by simply excluding wide
                  filters. `'exclude'` is simply a subset of `'resample'`,
                  preserving all center frequencies and widths - a 3D/4D
                  coefficient packing will zero-pad to compensate
                  (see `help(wavespin.toolkit.pack_coeffs_jtfs)`).
                - Non-`'resample'` are most useful with a large `J_fr`, i.e.
                  close to `log2(N_frs_max)`, and may have little effect otherwise
                  (or even none with small `J_fr`).

            Note: `sampling_phi_fr = 'exclude'` will re-set to `'resample'`, as
            `'exclude'` isn't a valid option (there must exist a lowpass for every
            fr input length).

            # TODO make note on 3D motivation of `'recalibrate'`

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

                - `'average'`: Gaussian, standard for scattering. Imposes
                  frequency transposition invariance.

                - `'decimate'`: Hamming-windowed sinc (~brickwall in freq domain).
                  Decimates coefficients: used for ~unaliased downsampling,
                  without imposing invariance in Euclidean distance sense.

                  Experimental feature. See "`'decimate'` overview"
                  (or "Short version) below.

            **Short version**:

            Main use cases:

                - May benefit 1D & 2D conv-nets. Reduces output dimensionality
                  while approximately preserving its frequency unaveraging.
                - Visualization: for seeing how spinned coefficients look
                  (approx.) without frequential averaging, but without using
                  `F=1` (e.g. if it's computationally prohibitive, or a low
                  aliasing view on downsampled coefficients is desired).

            Notes:

                - The net effect of `'decimate'` depends on signal profile and
                  task (including subsequent steps, e.g. conv-nets), and needs
                  further study.
                - For both purposes, good to pair with higher `r_psi1` (see
                  `r_psi` docs).
                - For visualization, good to pair with `'average'`, to check
                  for "undue tails" along frequency that `'decimate'` can produce.
                - Must have `oversampling_fr=0`, and cannot change device after
                  first call to scattering.

            **`'decimate'` overview**:

                   - Preserves more information along frequency than `'average'`
                     (see "Info preservation" below).
                   - Corrects negative outputs via absolute value; the negatives
                     are possible since the kernel contains negatives, but are in
                     minority and are small in magnitude.
                   - Invariance is still imposed in an "information sense", but
                     not Euclidean distances sense; see `sigma0` docs.
                   - May benefit 1D & 2D convolutions, but be detrimental in
                     higher dimensions per loss of smooth structure; this wasn't
                     sufficiently studied.

            Does not interact with other parameters in any way - that is, won't
            affect stride, padding, etc - only changes the lowpass filter for
            spinned pairs. `phi_f` pairs will still use Gaussian, and `phi_f_fr`
            remains Gaussian but is used only for `phi_f` pairs. Has no effect
            with `average_fr=False`.

            `'decimate'` is also experimental in technical sense (i.e. not fully
            developed), but it is tested:

                - is differentiable
                - filters are automatically converted to input's device,
                  precision, and backend
                - `'torch'` backend lacks `register_filters` support, so filters
                  are invisible to `nn.Module`
                - filters are built dynamically, on per-requested basis. The first
                  run is slower than the rest as a result
                - `oversampling_fr != 0` is not supported
                - changing device after first call to scattering is not supported

            **Info preservation**:

            `'decimate'`

              - 1) Increases amount of information preserved.

                  - Its cutoff spills over the alias threshold, and there's
                    notable amount of aliasing (subject to future improvement).
                  - Its main lobe is narrower than Gauss's, hence better
                    preserving component separation along frequency.

                      - Usually, narrower main lobe would be at expense of longer
                        tails, but that assumes same effective bandwidth; the
                        design here uses much greater effective bandwidth for
                        `'decimate'`, so the "expense" is aliasing, see point 3.

                  - Limited reconstruction experiments did not reveal a definitive
                    advantage over Gaussian: either won depending on transform
                    and optimizer configurations. Further study is required.

              - 2) Reduces distortion of preserved information.

                  - The Gaussian changes relative scalings of bins, progressively
                    attenuating higher frequencies, whereas windowed sinc is ~flat
                    in frequency until reaching cutoff (i.e. it copies input's
                    spectrum). As a result, Gaussian blurs, while windowed sinc
                    faithfully captures the original, including shape and spatial
                    localization.
                  - At the same time, sinc increases distortion per aliasing, but
                    the net effect is a benefit.

              - 3) Increases distortion of preserved information.

                  - Due to maybe notable aliasing. Amount of energy aliased is
                    4-5% of total energy, while for the Gaussian it's 0.0001%.
                  - Due to the time-domain kernel having negatives, which
                    sometimes outputs negatives for a non-negative input,
                    requiring correction.
                  - Due to "undue tails". This effect is signal-dependent, and is
                    generally very small, being worst for pulse-like shapes along
                    frequency axis in scalogram. The Gaussian also has tails,
                    which are in fact longer than the sinc's by `'decimate'`'s
                    design, but its tails are consistent and well-behaved
                    (e.g. sinc has tails for an echirp along frequency, with
                    oscillating amplitudes along time; Gauss doesn't oscillate).
                    The sinc was designed to have low support to minimize this
                    effect.

            Likely for most signals, `2)` far outweighs `3)`. `2)` is the main
            advantage and is the main motivation for `'decimate'`: we want
            a smaller unaveraged output, that resembles the full original.

        max_pad_factor_fr : int >= 0 (default 1) / None / list[int]
            `max_pad_factor` for frequential axis in frequential scattering.

                - `None`: unrestricted; will pad as much as needed.

                - `list[int]`: controls max padding for each `N_fr_scale`
                  separately, in reverse order (max to min).

                    - Values may not be such that they yield increasing
                      `J_pad_frs`
                    - If the list is insufficiently long (less than number of
                      scales), will extend list with the last provided value
                      (e.g. `[1, 2] -> [1, 2, 2, 2]`).
                    - Indexed by `scale_diff == N_fr_scales_max - N_fr_scales`

                - `int`: will convert to list[int] in an adaptive manner to
                  balance quality and compute constraints; see "`int` behavior".

                - `list[tuple[int]]`: despite non-`None` converting to this
                  structure internally, it's not permitted as input.

            **`int` behavior**:

            Pad amounts are controlled separately for each `N_fr_scale`, and
            for spinned and non-spinned pairs. Let `mpf` be the original user
            spec, and `scale_diff = N_fr_scales_max - N_fr_scale`. Then, for
            `'resample'` we have

                `max_pad_factor_fr[scale_diff] = mpf + scale_diff`

            else (non-`'resample'`)

                `max_pad_factor_fr[scale_diff] = min(mpf + scale_diff, ideal - 1)`

            and this is handled separately in a size 2 tuple, where
            `max_pad_factor_fr[scale_diff][0]` stores max pad factor for spinned
            pairs, conditioned by `sampling_psi_fr`, and `[1]` for non-spinned,
            conditioned by `sampling_phi_fr`. The tuple is for the separate
            handling in case of `average_fr=False`.

            The idea is, non-`'resample'` lowers pad requirements for lower
            `N_fr_scale` by design, so `mpf + scale_diff` will eventually pad
            every `N_fr_scale` ideally, yet the user elected non-`None`, so we
            go with one less than ideal. We could've reused `1` for all, but the
            gains in compute speed are small, and harms to feature quality large.
            `'resample'`, however, lacks such cleanly predictable wiggle room.

            Note, "ideal" is `log2(width) + 4`, following the default `sigma0`.
            Hence, it's set as `4 + ceil(log2(width_exclude_ratio))` for
            non-`'resample'` (see `sampling_filters_fr`).

            If this behavior is undesired, the same value can be set for all
            `scale_diff` by passing the integer as one-sized list, e.g.
            `max_pad_factor_fr = [1]` (`[1]` not to be confused with earlier
            shorthand notation), but the overriding notes in
            "Realization behavior" still apply.

            **Realization behavior**:

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

            **Other notes**:

            A limitation of `None` with `analytic=True` is,
            `compute_minimum_support_to_pad` does not account for it.

        pad_mode_fr : str['zero', 'conj-reflect-zero'] / function
            Name of frequential padding mode to use:

                - `'zero`': zero-padding. Standard and recommended padding,
                  matching the true extension for a bandlimited signal.
                - `'conj-reflect-zero'`: zero-pad lower frequency portion, and
                  conjugate + `'reflect'` all else. Aids energy conservation,
                  especially for large `J_fr` (`J_fr` > `N_fr_scales_max - 3`),
                  but is prone to introducing distortions, and is slower. Useful
                  for synthetic tests and debugging, and can sometimes improve
                  (on-task) performance.

            Can also be a function with signature `pad_fn_fr(x, pad_fr, scf, B)`;
            see `_right_pad` in
            `wavespin.scattering1d.core.timefrequency_scattering1d`.

            If using `pad_mode = 'reflect'` and `average = True`, reflected
            portions will be automatically conjugated before frequential
            scattering to avoid spin cancellation. For same reason, there isn't
            `pad_mode_fr = 'reflect'`.

            Demo & detailed explanation on `'conj-reflect-zero'`:
            https://github.com/kymatio/kymatio/discussions/
            752#discussioncomment-864234

            Can be safely changed after instantiation IF the original
            `pad_mode_fr` wasn't `'zero'`. See `pad_mode` in
            `help(wavespin.Scattering1D())`.

        normalize_fr : str
            See `normalize` in `help(wavespin.Scattering1D())`.
            Applies to `psi1_f_fr_up`, `psi1_f_fr_dn`, `phi_f_fr`.

        r_psi_fr : float
            See `r_psi` in `help(wavespin.Scattering1D())`.

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

        out_exclude : list[str] / tuple[str] / None
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
            Supported keys: `'n2'`, `'n1_fr'`, `'j2'`, `'j1_fr'`. E.g.:

                - `{'n2': [2, 3, 5], 'n1_fr': [0, -1]}`
                - `{'j2': 1, 'j1_fr': [3, 1]}`
                - `{'n2': [0, 1], 'j2': [-1]}`

            Negatives wrap around like indexing, e.g. `j2=-1` means `max(j2s)`.
            `dict[str: int]` will convert to `dict[str: list[int]]`.
            `n2=3` means will exclude `self.psi2_f[3]`.
            `j2=3` excludes all `p2f = self.psi2_f` with `p2f['j'] == 3`.
            `n1_fr=3` excludes `self.psi1_f_fr_up[psi_id][3]` for all `psi_id`.

            Excluding `j2==1` paths yields greatest speedup, and is recommended
            in compute-restricted settings, as they're the lowest energy paths
            (i.e. generally least informative). However, it's recommended to
            tune `smart_paths` instead.

            Note, `n2` and `n1_fr` only affect `psi_t *` pairs. To exclude
            `phi_t *` pairs, use `out_exclude`.

            Can be changed after instantiation, *except for* the `'n2, n1'` key
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

            **Predicting `N_frs_max`**:

            The determination of this quantity is extremely complicated. The
            safest way to do so is to simply instantiate a JTFS object, and print
            `jtfs.scf.N_frs_max`, which will never exceed `len(jtfs.psi1_f)`, and
            is independent of frequential specs (`J_fr`, `F`, etc). Determination
            of `J_fr` and `F` however should be based on time-frequency design or
            output size; latter is yet more complicated and best obtained by just
            doing scattering.

            Simple example:

            ::

                configs = dict(shape=2048, Q=8, J=8)
                dummy = TimeFrequencyScattering1D(**configs)
                print("N_frs_max", dummy.scf.N_frs_max)

                # As of writing, it's 53; may change in future, and that's OK!
                # This means `out.shape[-2]` (on joint pairs) is 53 or less.
                # Say we want to make it 8 or less. So, `F = ceil(53/8) = 7`,
                # just take `8`. Note, need `average_fr=True` to have effect
                # on spinned pairs.
                jtfs = TimeFrequencyScattering1D(**configs, F=8, average_fr=True)

            Example with output sizes:

            ::

                x = np.random.randn(2048)
                configs = dict(shape=len(x), Q=8, J=8, average_fr=True,
                               out_3D=True, out_type='dict:array')
                # use `F=1` to see max output shape
                dummy = TimeFrequencyScattering1D(**configs, F=1)
                out_d = dummy(x)

                print("N_frs_max", dummy.scf.N_frs_max)
                print("dummy_spinned.shape =", out_d['psi_t * psi_f_up'].shape)

                # As expected, `out.shape[-2] == N_frs_max`. Now we reduce it
                # with higher `F`.
                jtfs = TimeFrequencyScattering1D(**configs, F=8)
                out = jtfs(x)
                print("spinned.shape =", out['psi_t * psi_f_up'].shape)

            Also see `examples/more/jtfs_out_shapes.py`, concerning other
            configurations.

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
            (for `'exclude'` & `'recalibrate'` `sampling_psi_fr`).

        N_fr_scales_max : int
            `== max(N_fr_scales)`. Used to set `J_pad_frs_max` and
            `J_pad_frs_max_init`.

                - `J_fr` default is set using this value, and `J_fr` cannot
                  exceed it. If `F == 2**J_fr`, then `average_fr_global=True`.
                - Used in `compute_J_pad_fr()` and `psi_fr_factory()`.

        N_fr_scales_min : int
            `== min(N_fr_scales)`.

            Used in `jtfs.scf._compute_J_pad_frs_min_limit_due_to_psi`.

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
            See `help(jtfs.scf.compute_stride_fr)`.
            See "Compute logic: stride, padding" in
            `wavespin.scattering1d.core.timefrequency_scattering1d`.

            `over_U1` seeks to emphasize that it is the stride over first order
            coefficients.

        total_conv_stride_over_U1s_phi : dict[int: int]
            Stores strides for frequential scattering (`phi_f` pairs):

                {scale_diff: stride}

            Derives from `total_conv_stride_over_U1s`, differently depending on
            `average_fr`, `aligned`, and `sampling_phi_fr`.
            See "Stride, padding: `phi_f` pairs" in
            `wavespin.scattering1d.core.timefrequency_scattering1d`.

        n1_fr_subsamples : dict[str: dict[int: list[int]]]
            Stores strides for frequential scattering (`psi_f` pairs).
            Accounts for both `j1_fr` and `log2_F_phi`, so subsampling won't
            alias the lowpass.

                {'spinned: {scale_diff: [...]},
                 'phi':    {scale_diff: [...]}}

            See `jtfs.scf._compute_scale_and_stride_logic`.

        log2_F_phis : dict[str: dict[int: list[int]]]
            `log2_F`-equivalent - that is, maximum permitted subsampling, and
            dyadic scale of invariance - of lowpass filters used for a given
            pair, `N_fr_scale`, and `n1_fr` -

                {'spinned: {scale_diff: [...]},
                 'phi':    {scale_diff: [...]}}

            Equals `log2_F` everywhere with `sampling_phi_fr='resample'`.
            Is `None` for `'spinned'` (for each `scale_diff`) if
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
            `wavespin.scattering1d.core.timefrequency_scattering1d`,
            specifically `'recalibrate'`.

        phi_f : dict
            Is structurally different from that of `Scattering1D`: it's now
            `phi_f[trim_tm][sub]`, where `phi_f[0][sub]` is same as `phi_f[sub]`
            of `Scattering1D`.

            `trim_tm` controls the unsubsampled length of the filter, for
            different amounts of intermediate temporal unpadding that JTFS does
            in joint scattering.

        psi1_f : dict
            Same as that of `Scattering1D`.

        psi2_f : dict
            Same as that of `Scattering1D`.

        phi_f_fr : dict[int: dict, str: dict]
            Contains the frequential lowpass filter at all resolutions.
            See `help(wavespin.scattering1d.filter_bank_jtfs.phi_fr_factory)`.

            Full type spec:

                dict[int: dict[int: list[tensor[float]]],
                     str: dict[int: dict[int: list[int]], float]]

        psi1_f_fr_up : dict[int: dict, str: dict]
            List of dictionaries containing all frequential scattering filters
            with "up" spin.
            See `help(wavespin.scattering1d.filter_bank_jtfs.psi_fr_factory)`.

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
            See `help(jtfs.scf._compute_psi_fr_params)` and
            `help(wavespin.scattering1d.filter_bank_jtfs.psi_fr_factory)`.

        pad_fn_fr : function
            A backend padding function, or user function (as passed
            to `pad_mode_fr`), with signature

                pad_fn_fr(Y_2, pad_fr, vectorized, scf, B)

            where

                - `Y_2` is the list or tensor of coefficients to frequentially pad
                - `2**pad_fr` is the amount to pad `Y_2.shape[-2]` to
                - `B` is the backend object

        average_fr_global_phi : bool
            `True` if `F == nextpow2(N_frs_max)`, i.e. `F` is maximum possible
            and equivalent to global averaging, in which case lowpassing is
            replaced by simple arithmetic mean.

            If `True`, `sampling_phi_fr` has no effect.

            In case of `average_fr==False`, controls scattering logic for
            `phi_f` pairs.

        average_fr_global : bool
            `True` if `average_fr_global_phi and average_fr`. Same as
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
            (column lengths given by `N_frs`).
            See `jtfs.scf.compute_padding_fr()`.

        J_pad_frs_max_init : int
            Set as reference for computing other `J_pad_fr`.

            Serves to create the initial frequential filterbank, and equates to
            `J_pad_frs_max` with `sampling_psi_fr='resample'` &
            `sampling_phi_fr='resample'`. Namely, it is the maximum padding under
            "standard" frequential scattering configurations.

        J_pad_frs_max : int
            `== max(J_pad_frs)`. Realized maximum padding, may differ from
            `J_pad_frs_max_init`.

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
            Used in computing `J_pad_fr`. See `jtfs.scf.compute_J_pad_fr()`.

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
            See `help(jtfs.scf.compute_padding_fr)`.

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
              - With `'recalibrate'`, `scale_diff_max_to_build=None` if build
                didn't terminate per `sigma_max_to_min_max_ratio`.
              - With `'resample'`, it's `None` until padding finishes computation,
                then it is set to the first `scale_diff` that indexes into a
                padding that matches the minimum padding.

        sigma_max_to_min_max_ratio : float >= 1
            Largest permitted `max(sigma) / min(sigma)`. Used with `'recalibrate'`
            `sampling_psi_fr` to restrict how large the smallest sigma can get.

            Worst cases (high `subsample_equiv_due_to_pad`):

              - A value of `< 1` means a lower center frequency will have
                the narrowest temporal width, which is undesired.
              - A value of `1` means all center frequencies will have the same
                temporal width, which is undesired.
              - The `2` default was chosen as a compromise between not overly
                restricting sigma, and closeness to `1`. Originally it was chosen
                to be `1.2`, but that was too liberal; it prevented complete
                tiling per CQT, and caused excessive redundancy.

            Configurable via `wavespin.CFG`.

        width_exclude_ratio : float > 0
            Ratio to use in `sampling_psi_fr = 'exclude'` and `'recalibrate'`.

            - `'exclude'`: a frequential scattering filter is excluded if

                  `width > 2**N_fr_scale * width_exclude_ratio`

              As the default is `0.5`, this means we keep filters with width
              being at most half the (p2up of) frequential input length.

            - `'recalibrate'`: the original (max input length) filterbank is
              reused as long as its highest width wavelet satisfies

                  `width < 2**N_fr_scale * width_exclude_ratio`

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
            Enforces `min(N_frs) >= N_frs_min_global`. Is used to exclude `n2`'s
            that derive from very few `n1`'s, which are uninformative and heavy
            with transform artifacts. The algorithm is, for any `n2`:

                1. `N_frs_min_global//2 < n_n1s < N_frs_min_global`
                   Appends n1s until `n_n1s == N_frs_min_global`.
                2. `n_n1s <= N_frs_min_global//2`
                   Discards the `n2`. Per diminishing scattering energies,
                   appending is ineffective.
                3. `n_n1s >= N_frs_min_global`
                   Do nothing.

            Set to `0` to disable.
            Configurable via `wavespin.CFG`.

        do_energy_correction : bool (default True)
            Whether to do energy correction. Enables

                ||jtfs(x)|| ~= ||x||
                ||jtfs_subsampled(x)|| ~= ||jtfs(x)||

            by scaling coefficients by their subsampling factors, also potentially
            via fractional unpad correction (see `do_ec_frac_tm`). These may
            or may not be desired.

            Only partial results if used without `'l1-energy'` normalizations.
            May have a minor performance overhead.
            See `_compute_energy_correction_factor` in `scat_utils_jtfs.py`.
            See `do_ec_frac_tm` and `do_ec_frac_fr`.

            **Short version**:

            Use `False` if

                - `average_fr` && `aligned`
                - `average_fr` && `out_3D` && `sampling_phi_fr`

            or if coefficient scalings should be independent of subsampling, as
            one might wish for |STFT| or split-up |CWT| (more subsampling for
            lower freqs, with separate 2D outputs).

            **When to use `False`**:

                1. If correction amounts to a global rescaling (see below) and we
                wish to save compute, or we apply some normalization on outputs.

                2. If coefficients should accurately reflect intensities
                (original coefficient magnitudes), rather than energies.
                (Coefficient values are still always magnitudes, but correction
                makes it so energies are conserved.)

            Example by comparison:

                - Multi-rate |CWT|. One can split up the scalogram into different
                subsampling factors. For example, use 2 for high freqs, and 8
                for low freqs. Feed the resulting separate 2D arrays to whichever
                algorithm. If in this case we wouldn't rescale the two outputs
                based on subsampling factor, then `False` should be used.

            **When correction is global rescaling**:

            `average = True` for all:

                - average_fr = True
                  aligned = True
                - average_fr = True
                  out_3D = True
                  sampling_filters_fr = (*, 'resample')
                - average_fr = False
                  aligned = False
                  out_3D = False (forced)

            These force shared stride in time and frequency.

            *Exception*: `S0` and `S1` by a different factor than all other
            pairs, and the `phi_t * psi_f` pair is always rescaled by an
            additional `sqrt(2)`.


        do_ec_frac_tm : bool / None
            Whether to do fractional unpad index energy correction for temporal
            scattering. Default is determined at build time.
            See `_compute_energy_correction_factor` in `scat_utils_jtfs.py`.

        do_ec_frac_fr : bool / None
            `do_ec_frac_tm` for frequential scattering.

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

            See `help(jtfs.scf.compute_padding_fr)` and
            `jtfs.scf.compute_unpadding_fr()`.

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
            Additionally internally sets the `'n2, n1'` pair via `smart_paths`.

        paths_include_build : dict
            "Set in stone" paths computed at build time, see
            `build_scattering_paths_fr()`.

        paths_include_n2n1 : dict
            Has differences relative to `Scattering1D`, see
            `build_paths_include_n2n1()`.

        out_structure : int / None
            See `implementation` docs.

        api_pair_order : set[str]
            Standardized order of JTFS coefficient pairs for internal
            consistency.

        fr_attributes : tuple
            Names of attributes that can be accessed from `self` via `self.scf`.

        args_meta : dict
            We desire to control temporal and frequential scattering with a single
            frontend, yet have separate classes for each. This requires argument
            resolution, hence `args_meta`.

              - `'passed_to_scf'` is what's fetched from `self` (i.e. top level)
                and passed to `self.scf`'s constructor.

                  - Enables `self.name = value` and `setattr(self, name, value)`
                    syntax during build.
                  - Used for sanity check that we accounted for all arguments.

              - `'top_level'` is what belongs in `self`, but `self.scf` happens
                to use them, and rerouting from `self.scf` isn't needed.

              - `'only_scf'` is all that's in `'passed_to_scf'` but not in
                `'top_level'`.

                  - Removed from `self` after `self.scf` is built.
                  - Used to reroute `self.scf` args to `self` via `fr_attributes`.

        DYNAMIC_PARAMETERS_JTFS : set[str]
            See `DYNAMIC_PARAMETERS` in `help(wavespin.Scattering1D())`.

            *Note*, not all listed parameters are always safe to change, refer
            to the individual parameters' docs for exceptions.
        """

    _doc_terminology = \
        r"""
        Terminology
        -----------
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

        CQT :
            Constant-Q Transform. `Q = (center frequency) / (bandwidth)`, is the
            "quality factor", *not* to be confused with `Q` in this library.
            Enables important properties:

                1. Time-warp stability of the scattering transform;
                illustrated in https://dsp.stackexchange.com/a/78513/50076 .

                2. Constant redundancy of the Continuous Wavelet Transform (and
                hence of scattering). However, CQT is necessary but not
                sufficient: additionally the scales/frequencies must be
                exponentially distributed - that is, `xi / sigma = const.`, and
                `xi_{i + 1} / xi_{i} = const.`.

                3. Time-warp equivariance, within a time stretch, of the CWT.
                Equivalently, scale-equivariance: features of contracted input
                match stretched features of original input.
                Derivation above eq. (2.13) in
                https://theses.hal.science/tel-01559667/document (V. Lostanlen)

                4. Enables constant time-frequency resolution of the CWT, in
                log sense: see https://dsp.stackexchange.com/a/84746/50076

            `2.` is critical for JTFS, since we spatially operate along
            frequency, and convolutions assume spatial uniformity, which (due to
            log-scaling of bandwidths) is only achievable in log-frequency,
            meaning uniform spacing along log-frequency.

            See "Parameter Sweeps" example for a visual.

        support :
            These (support, width, length) are well-defined and not used
            interchangeably, see "Meta" in
            `help(wavespin.scattering1d.filter_bank.scattering_filter_factory)`.
            "Length" exclusively refers to 1D array size / number of samples.

        width :
            See "support".

        length :
            See "support".

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
        NOTE: unlike in `Scattering1D`, the batch dimension is not collapsed
        if `len(x.shape) == 1`.

        Parameters
        ----------
        x : {array}
            An input `{array}` of size `(B, N)` or `(N,)`.

        Returns
        -------
        S : dict[tensor/list] / tensor/list / two-tuple of tensor/list
            See above.
        """


__all__ = ['ScatteringBase1D', 'TimeFrequencyScatteringBase1D']
