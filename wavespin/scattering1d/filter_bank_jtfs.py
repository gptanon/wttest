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
from types import FunctionType
from copy import deepcopy

from .filter_bank import (morlet_1d, gauss_1d, fold_filter_fourier,
                          calibrate_scattering_filters, width_2_scale,
                          N_and_pad_2_J_pad, make_strictly_analytic)
from ..utils.measures import (compute_spatial_support, compute_spatial_width,
                              compute_bandwidth, compute_bw_idxs,
                              compute_minimum_required_length,
                              compute_max_dyadic_subsampling)
from .refining import energy_norm_filterbank_fr
from .scat_utils import compute_minimum_support_to_pad
from ..frontend.base_frontend import ScatteringBase
from .. import CFG


class _FrequencyScatteringBase1D(ScatteringBase):
    """Attribute object for TimeFrequencyScatteringBase1D for frequential
    scattering part of JTFS.
    """
    # note: defaults are set in `wavespin.scattering1d.frontend.base_frontend.py`,
    # `DEFAULT_KWARGS`
    def __init__(self, paths_include_build, N_frs, J_fr=None, Q_fr=None, F=None,
                 max_order_fr=None, average_fr=None, aligned=None,
                 oversampling_fr=None, sampling_filters_fr=None,
                 out_3D=None, max_pad_factor_fr=None,
                 pad_mode_fr=None, analytic_fr=None,
                 max_noncqt_fr=None, normalize_fr=None, F_kind=None,
                 r_psi_fr=None, _n_psi1_f=None, backend=None):
        super(_FrequencyScatteringBase1D, self).__init__()
        self._paths_include_build = paths_include_build
        self._N_frs = N_frs
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.max_order_fr = max_order_fr
        self.average_fr = average_fr
        self.aligned = aligned
        self.max_noncqt_fr = max_noncqt_fr
        self.oversampling_fr = oversampling_fr
        self.sampling_filters_fr = sampling_filters_fr
        self.sampling_psi_fr = None  # set in build()
        self.sampling_phi_fr = None  # set in build()
        self.out_3D = out_3D
        self.max_pad_factor_fr = max_pad_factor_fr
        self.pad_mode_fr = pad_mode_fr
        self.analytic_fr = analytic_fr
        self.normalize_fr = normalize_fr
        self.F_kind = F_kind
        self.r_psi_fr = r_psi_fr
        self._n_psi1_f = _n_psi1_f
        self.backend = backend

        self.build()
        self.create_init_psi_filters()
        self.compute_unpadding_fr()
        self.compute_stride_fr()
        self.compute_scale_and_stride_logic()
        self.compute_padding_fr()
        self.create_psi_filters()
        self.compute_scale_and_stride_logic(for_validation=True)
        self.create_phi_filters()
        self.adjust_padding_and_filters()

        # TODO chk all docs
        # TODO nuke kymatio discussions everywhere
        # TODO remove `Tx

    # forbid modifying these #################################################
    @property
    def paths_include_build(self):
        return self._paths_include_build

    @property
    def N_frs(self):
        return self._N_frs
    # ------------------------------------------------------------------------

    def build(self):
        """Mainly handles input arguments. For a description of the complete
        build pipeline, see `compute_padding_fr`.
        """
        self.sigma0 = CFG['JTFS']['sigma0']
        self.P_max = CFG['JTFS']['P_max']
        self.eps = CFG['JTFS']['eps']
        self.criterion_amplitude = CFG['JTFS']['criterion_amplitude']
        self.sigma_max_to_min_max_ratio = CFG['JTFS'][
            'sigma_max_to_min_max_ratio']
        self.width_exclude_ratio = CFG['JTFS']['width_exclude_ratio']

        # `N_frs` used in scattering, == realized `psi2_f`s
        self.N_frs_realized = [s for s in self.N_frs if s > 0]
        # longest & shortest obtainable frequency row w.r.t. which we
        # calibrate filters
        self.N_frs_max = max(self.N_frs_realized)
        self.N_frs_min = min(self.N_frs_realized)
        # above is for `psi_t *` pairs, below is actual max, which
        # occurs for `phi_t *` pairs
        self.N_frs_max_all = self._n_psi1_f
        # compute corresponding scales
        self.N_fr_scales = [(math.ceil(math.log2(s)) if s != 0 else -1)
                            for s in self.N_frs]
        self.N_fr_scales_max = max(self.N_fr_scales)
        # smallest scale is also smallest possible maximum padding
        # (cannot be overridden by `max_pad_factor_fr`)
        self.N_fr_scales_min = min(s for s in self.N_fr_scales if s != -1)
        # scale differences
        self.scale_diffs = [(self.N_fr_scales_max - N_fr_scale
                             if N_fr_scale != -1 else -1)
                            for N_fr_scale in self.N_fr_scales]
        # make unique variants
        self.N_fr_scales_unique = np.unique([N_fr_scale
                                             for N_fr_scale in self.N_fr_scales
                                             if N_fr_scale != -1])
        self.scale_diffs_unique = np.unique([scale_diff
                                             for scale_diff in self.scale_diffs
                                             if scale_diff != -1])
        # store number of unique scales
        self.n_scales_fr = len(self.scale_diffs_unique)

        # ensure 2**J_fr <= nextpow2(N_frs_max)
        if self.J_fr is None:
            # default to `max - 2` if possible, but no less than `3`,
            # and no more than `max`
            self.J_fr = min(max(self.N_fr_scales_max - 2, 3),
                            self.N_fr_scales_max)
        elif self.J_fr > self.N_fr_scales_max:
            raise ValueError(("2**J_fr cannot exceed maximum number of frequency "
                              "rows (rounded up to pow2) in joint scattering "
                              "(got {} > {})".format(
                                  2**(self.J_fr), 2**self.N_fr_scales_max)))

        # check F or set default
        if self.F == 'global':
            self.F = 2**self.N_fr_scales_max
        elif self.F > 2**self.N_fr_scales_max:
            raise ValueError("The temporal support F of the low-pass filter "
                             "cannot exceed maximum number of frequency rows "
                             "(rounded up to pow2) in joint scattering "
                             "(got {} > {})".format(
                                 self.F, 2**self.N_fr_scales_max))
        self.log2_F = math.floor(math.log2(self.F))
        self.average_fr_global_phi = bool(self.F == 2**self.N_fr_scales_max)
        self.average_fr_global = bool(self.average_fr_global_phi and
                                      self.average_fr)

        # handle F_kind
        if self.F_kind == 'decimate':
            from ..toolkit import Decimate
            self.decimate = Decimate(backend=self.backend, sign_correction='abs')
        elif self.F_kind == 'average':
            self.decimate = None
        else:
            raise ValueError(
                "`F_kind` must be 'average' or 'decimate', got %s" % self.F_kind)

        # restrict `J_pad_frs_max` (and `J_pad_frs_max_init`) if specified by user
        if isinstance(self.max_pad_factor_fr, int):
            self.max_pad_factor_fr = {scale_diff: self.max_pad_factor_fr
                                      for scale_diff in self.scale_diffs_unique}

        elif isinstance(self.max_pad_factor_fr, (list, tuple)):
            _max_pad_factor_fr = {}
            for i, scale_diff in enumerate(self.scale_diffs_unique):
                if i > len(self.max_pad_factor_fr) - 1:
                    _max_pad_factor_fr[scale_diff] = self.max_pad_factor_fr[-1]
                else:
                    _max_pad_factor_fr[scale_diff] = self.max_pad_factor_fr[i]
            self.max_pad_factor_fr = _max_pad_factor_fr

            # guarantee that `J_pad_fr > J_pad_frs_max_init` cannot occur
            if max(self.max_pad_factor_fr.values()) > self.max_pad_factor_fr[0]:
                J_pad_frs_max = max(s + p for s, p in
                                    zip(self.N_fr_scales_unique[::-1],
                                        self.max_pad_factor_fr.values()))
                first_max_pf_min = J_pad_frs_max - self.N_fr_scales_max
                # ensures `J_pad_frs[0] >= J_pad_frs[1:]`
                self.max_pad_factor_fr[0] = first_max_pf_min

        elif self.max_pad_factor_fr is None:
            pass

        else:
            raise ValueError("`max_pad_factor_fr` mus be int>0, "
                             "list/tuple[int>0], or None (got %s)" % str(
                                 self.max_pad_factor_fr))
        self.unrestricted_pad_fr = bool(self.max_pad_factor_fr is None)

        # validate `max_pad_factor_fr`
        # 1/2**J < 1/Np2up so impossible to create wavelet without padding
        if (self.J_fr == self.N_fr_scales_max and
                self.max_pad_factor_fr is not None and
                self.max_pad_factor_fr[0] == 0):
            raise ValueError("`max_pad_factor_fr` can't be 0 if "
                             "J_fr == log2(nextpow2(N_frs_max)). Got, "
                             "respectively, %s\n%s\n%s" % (
                                 self.mad_pad_factor_fr, self.J_fr,
                                 self.N_fr_scales_max))

        # validate `out_3D`
        if self.out_3D and not self.average_fr:
            raise ValueError("`out_3D=True` requires `average_fr=True`. "
                             "`F=1` with `average_fr=True` will yield coeffs "
                             "close to unaveraged.")

        # check `pad_mode_fr`, set `pad_fn_fr`
        supported = ('conj-reflect-zero', 'zero')
        if isinstance(self.pad_mode_fr, FunctionType):
            fn = self.pad_mode_fr

            def pad_fn_fr(x, pad_fr, scf, B):
                return fn(x, pad_fr, scf, B)

            self.pad_mode_fr = 'custom'

        elif self.pad_mode_fr not in supported:
            raise ValueError(("unsupported `pad_mode_fr` '{}';\nmust be a "
                              "function, or string, one of: {}").format(
                                  self.pad_mode_fr, ', '.join(supported)))

        else:
            pad_fn_fr = None  # handled in `core`
        self.pad_fn_fr = pad_fn_fr

        # unpack `sampling_` args
        if isinstance(self.sampling_filters_fr, tuple):
            self.sampling_psi_fr, self.sampling_phi_fr = self.sampling_filters_fr
            if self.sampling_phi_fr == 'exclude':
                # if user explicitly passed 'exclude' for `_phi`
                warnings.warn("`sampling_phi_fr = 'exclude'` has no effect, "
                              "will use 'resample' instead.")
                self.sampling_phi_fr = 'resample'
        else:
            self.sampling_psi_fr = self.sampling_phi_fr = self.sampling_filters_fr
            if self.sampling_phi_fr == 'exclude':
                self.sampling_phi_fr = 'resample'
        self.sampling_filters_fr = (self.sampling_psi_fr, self.sampling_phi_fr)

        # validate `sampling_*` args
        psi_supported = ('resample', 'recalibrate', 'exclude')
        phi_supported = ('resample', 'recalibrate')
        if self.sampling_psi_fr not in psi_supported:
            raise ValueError(("unsupported `sampling_psi_fr` ({}), must be one "
                              "of: {}").format(self.sampling_psi_fr,
                                               ', '.join(psi_supported)))
        elif self.sampling_phi_fr not in phi_supported:
            raise ValueError(("unsupported `sampling_phi_fr` ({}), must be one "
                              "of: {}").format(self.sampling_phi_fr,
                                               ', '.join(phi_supported)))
        elif self.sampling_phi_fr == 'recalibrate' and self.average_fr_global_phi:
            raise ValueError("`F='global'` && `sampling_phi_fr='recalibrate'` "
                             "is unsupported.")
        elif self.sampling_phi_fr == 'recalibrate' and self.aligned:
            raise ValueError("`aligned=True` && `sampling_phi_fr='recalibrate'` "
                             "is unsupported.")

        # compute maximum amount of padding.
        # we do this at max possible `N_fr` per each dyadic scale to guarantee
        # pad uniqueness across scales; see `compute_padding_fr` docs
        (self.J_pad_frs_max_init, self.min_to_pad_fr_max, self._pad_fr_phi,
         self._pad_fr_psi) = self._compute_J_pad_fr(2**self.N_fr_scales_max,
                                                    Q=(self.Q_fr, 0))

        # track internally for edge case testing
        N_fr_scale_fo = math.ceil(math.log2(self._n_psi1_f))
        self._J_pad_fr_fo, *_ = self._compute_J_pad_fr(2**N_fr_scale_fo,
                                                       Q=(self.Q_fr, 0))

        # warn of edge case; see `_J_pad_fr_fo` in docs (base_frontend)
        if (self.max_pad_factor_fr is not None and
            self.max_pad_factor_fr[0] == 0):
            if self.J_pad_frs_max_init < self._J_pad_fr_fo:
                warnings.warn(("Edge case: due to `max_pad_factor_fr=0`, "
                               "the `phi_t * phi_f` pair cannot be padded, "
                               "and instead some first-order coefficients "
                               "will be excluded from computation."))

    def create_phi_filters(self):
        """See `filter_bank.phi_fr_factory`."""
        self.phi_f_fr = phi_fr_factory(
            self.J_pad_frs_max_init, self.J_pad_frs, self.F, self.log2_F,
            **self.get_params('unrestricted_pad_fr', 'pad_mode_fr',
                              'sampling_phi_fr', 'average_fr',
                              'average_fr_global_phi', 'aligned',
                              'criterion_amplitude', 'normalize_fr', 'sigma0',
                              'P_max', 'eps'))

    def create_psi_filters(self):
        """See `filter_bank.psi_fr_factory`."""
        (self.psi1_f_fr_up, self.psi1_f_fr_dn, self.psi_ids
         ) = psi_fr_factory(
            self.psi_fr_params, self.N_fr_scales_unique, self.N_fr_scales_max,
            self.J_pad_frs, **self.get_params(
                'sampling_psi_fr', 'scale_diff_max_to_build', 'normalize_fr',
                'criterion_amplitude', 'sigma0', 'P_max', 'eps'))

        # cannot do energy norm with 3 filters, and generally filterbank
        # isn't well-behaved
        n_psi_frs = len(self.psi1_f_fr_up[0])
        if n_psi_frs <= 3:
            raise Exception(("configuration yielded %s wavelets for frequential "
                             "scattering, need a minimum of 4; try increasing "
                             "J, Q, J_fr, or Q_fr." % n_psi_frs))

        # analyticity
        if self.analytic_fr:
            psi_fs_all = (self.psi1_f_fr_up, self.psi1_f_fr_dn)
            for s1_fr, psi_fs in enumerate(psi_fs_all):
                anti_analytic = bool(s1_fr == 1)
                for psi_id in psi_fs:
                    if isinstance(psi_id, int):
                        for n1_fr in range(len(psi_fs[psi_id])):
                            pf = psi_fs[psi_id][n1_fr]
                            make_strictly_analytic(pf, anti_analytic)

    def adjust_padding_and_filters(self):
        # realized minimum & maximum
        self.J_pad_frs_min = min(self.J_pad_frs.values())
        self.J_pad_frs_max = max(self.J_pad_frs.values())

        if not self.unrestricted_pad_fr:
            # adjust phi_fr
            pad_diff_max_realized = self.J_pad_frs_max_init - self.J_pad_frs_min
            log2_F_phi_diffs = [k for k in self.phi_f_fr if isinstance(k, int)]
            for log2_F_phi_diff in log2_F_phi_diffs:
                for pad_diff in self.phi_f_fr[log2_F_phi_diff]:
                    if pad_diff > pad_diff_max_realized:
                        del self.phi_f_fr[log2_F_phi_diff][pad_diff]
                # shouldn't completely empty a scale
                assert len(self.phi_f_fr[log2_F_phi_diff]) != 0

        # energy norm
        if 'energy' in self.normalize_fr:
            energy_norm_filterbank_fr(self.psi1_f_fr_up, self.psi1_f_fr_dn,
                                      self.phi_f_fr, self.J_fr, self.log2_F,
                                      self.sampling_psi_fr)

    def create_init_psi_filters(self):
        T = 1  # for computing `sigma_low`, unused
        (_, xi1_frs, sigma1_frs, is_cqt1_frs, *_
         ) = calibrate_scattering_filters(self.J_fr, self.Q_fr, T=T,
                                          r_psi=self.r_psi_fr, sigma0=self.sigma0,
                                          J_pad=self.J_pad_frs_max_init)

        # instantiate filter
        psi1_f_fr_up = {}
        scale_diff0 = -1  # since this filterbank is removed later
        psi1_f_fr_up[scale_diff0] = []
        # initialize meta
        for field in ('width', 'xi', 'sigma', 'j', 'is_cqt'):
            psi1_f_fr_up[field] = {scale_diff0: []}

        N_fr_scale = self.N_fr_scales_max
        for n1_fr in range(len(xi1_frs)):
            #### Compute wavelet #############################################
            # fetch wavelet params, sample wavelet
            xi, sigma = xi1_frs[n1_fr], sigma1_frs[n1_fr]
            padded_len = 2**self.J_pad_frs_max_init

            # expand dim to multiply along freq like (2, 32, 4) * (32, 1)
            pf = morlet_1d(padded_len, xi, sigma, normalize=self.normalize_fr,
                           P_max=self.P_max, eps=self.eps)[:, None]
            psi1_f_fr_up[scale_diff0].append(pf)

            # embed meta (only what we'll use for now) #######################
            peak_idx = np.argmax(np.abs(pf))
            bw_idxs = compute_bw_idxs(pf, self.criterion_amplitude, c=peak_idx)
            j = compute_max_dyadic_subsampling(pf, bw_idxs)

            width = compute_spatial_width(
                pf, N=2**N_fr_scale, sigma0=self.sigma0,
                criterion_amplitude=self.criterion_amplitude)
            psi1_f_fr_up['width' ][scale_diff0].append(width)
            psi1_f_fr_up['xi'    ][scale_diff0].append(xi)
            psi1_f_fr_up['sigma' ][scale_diff0].append(sigma)
            psi1_f_fr_up['j'     ][scale_diff0].append(j)
            psi1_f_fr_up['is_cqt'][scale_diff0].append(is_cqt1_frs[n1_fr])

        self.psi1_f_fr_up_init = psi1_f_fr_up

    def _compute_psi_fr_params(self):
        psis = self.psi1_f_fr_up_init
        scale_diff0 = -1

        params_init = {field: [p for n1_fr, p in enumerate(psis[field][-1])]
                       for field in ('xi', 'sigma', 'j', 'is_cqt')}
        params = {}

        if self.sampling_psi_fr in ('resample', 'exclude'):
            for N_fr_scale in self.N_fr_scales[::-1]:
                if N_fr_scale == -1:
                    continue
                scale_diff = self.N_fr_scales_max - N_fr_scale

                if self.sampling_psi_fr == 'resample' or scale_diff == 0:
                    # all `sampling_psi_fr` should agree on `N_fr_scales_max`;
                    # 'exclude' may omit some filters if `not unrestricted_pad_fr`
                    # per insufficient padding amplifying 'width'
                    params[scale_diff] = deepcopy(params_init)

                elif self.sampling_psi_fr == 'exclude':
                    params[scale_diff] = {field: [] for field in params_init}

                    for n1_fr in range(len(params_init['j'])):
                        width = psis['width'][scale_diff0][n1_fr]
                        if width > width_threshold(
                                N_fr_scale, self.width_exclude_ratio):
                            # subsequent `width` are only greater
                            break

                        for field in params_init:
                            params[scale_diff][field].append(
                                params_init[field][n1_fr])

        elif self.sampling_psi_fr == 'recalibrate':
            max_original_width = max(psis['width'][scale_diff0])
            (xi1_frs_new, sigma1_frs_new, j1_frs_new, is_cqt1_frs_new,
             self.scale_diff_max_recalibrate) = _recalibrate_psi_fr(
                 max_original_width,
                 *[params_init[field] for field in params_init],
                 **{k: getattr(self, k) for k in
                    ('N_fr_scales_max', 'N_fr_scales_unique',
                     'sigma_max_to_min_max_ratio', 'width_exclude_ratio',
                     'J_pad_frs_max_init', 'criterion_amplitude',
                     'normalize_fr', 'P_max', 'eps')})

            # pack as `params[scale_diff][field][n1_fr]`
            for scale_diff in j1_frs_new:
                params[scale_diff] = {field: [] for field in params_init}
                for field, p in zip(params_init, (xi1_frs_new, sigma1_frs_new,
                                                  j1_frs_new, is_cqt1_frs_new)):
                    params[scale_diff][field] = p[scale_diff]

        # ensure no empty scales
        for scale_diff in params:
            assert all(len(params[scale_diff][field]) > 0
                       for field in params[scale_diff]
                       ), (params, self.sampling_filters_fr)

        self.psi_fr_params = params

        if self.sampling_psi_fr != 'recalibrate':
            self.scale_diff_max_recalibrate = None

        # remove unused
        self.psi1_f_fr_up_init = {}
        del self.psi1_f_fr_up_init

        # compute needed quantity
        self._compute_J_pad_frs_min_limit_due_to_psi()

    def _compute_J_pad_frs_min_limit_due_to_psi(self):
        """
        `J_pad_frs_min_limit_due_to_psi` determined by:

          [1] 'resample' and unrestricted_pad_fr: smallest padding such that
              all wavelets still fully decay (reach `criterion_amplitude`)

          [2] 'exclude': smallest padding is padding that occurs at
              smallest `N_fr_scale` (i.e. largest `scale_diff`) that computes a
              filterbank (i.e. at least one wavelet with
              'width' > 2**N_fr_scale * width_exclude_ratio);
              i.e. we reuse that padding for lesser `N_fr_scale`

          [3] 'recalibrate': we reuse same as 'exclude', except now determined
              by `scale_diff_max_recalibrate` returned by
              `filter_bank_jtfs._recalibrate_psi_fr`

          [4] all: smallest padding such that all wavelets are wavelets
              (i.e. not a pure tone, one DFT bin, which throws ValueError)
              however we only exit the program if this occurs for non-'resample',
              as 'exclude' & 'recalibrate' must automatically satisfy this
              per shrinking support with smaller `N_fr_scale`

          not [5] phi: phi computes such that it's available at all given
              paddings, even if it distorts severely. However, worst case
              distortion is `scale == pad`, i.e. `log2_F_phi == J_pad_fr`,
              guaranteed by `J_pad_fr = max(, total_conv_stride_over_U1)`.

              Note phi construction can never fail via ValueError in [4],
              but it can become a plain global average against intent. We agree
              to potential distortions with `max_pad_factor_fr != None`,
              while still promising "no extreme distortions" (i.e. ValueError);
              here it's uncertain what's "extreme", as even a fully decayed phi
              with `log2_F_phi == N_fr_scale` will approximate a direct
              global averaging.

        [1] and [4] directly set `J_pad_frs_min_limit_due_to_psi`.
        [2] and [3] set `scale_diff_max_to_build`, which in turn
        sets `J_pad_frs_min_limit_due_to_psi` in `compute_padding_fr()`.
        """
        params = self.psi_fr_params
        scale_diff_max_to_set_pad_min = None
        pad_diff_max = None

        if self.sampling_psi_fr in ('exclude', 'recalibrate'):
            scale_diff_prev = 0
            for N_fr_scale in self.N_fr_scales_unique[::-1]:
                scale_diff = self.N_fr_scales_max - N_fr_scale
                if scale_diff not in params:
                    scale_diff_max_to_set_pad_min = scale_diff_prev
                    break
                scale_diff_prev = scale_diff

            if self.sampling_psi_fr == 'recalibrate':
                a = scale_diff_max_to_set_pad_min
                b = self.scale_diff_max_recalibrate
                assert a == b, (a, b)

        elif self.sampling_psi_fr == 'resample':
            # 'resample''s `else` is also applicable to 'exclude' & 'recalibrate',
            # but it's expected to hold automatically - and if doesn't, will
            # raise Exception in `filter_bank.psi_fr_factory`

            # unpack params
            xi1_frs, sigma1_frs, j1_frs, is_cqt1_frs = [
                params[0][field] for field in ('xi', 'sigma', 'j', 'is_cqt')]

            if self.unrestricted_pad_fr:
                # in this case filter temporal behavior is preserved across all
                # lengths, so we must restrict lowest length such that longest
                # filter still decays
                pad_diff = 0
                while True:
                    # `scale_diff = 0` <=> `J_pad_fr = J_pad_frs_max_init` here
                    J_pad_fr = self.J_pad_frs_max_init - pad_diff

                    psi_longest = morlet_1d(2**J_pad_fr, xi1_frs[-1],
                                            sigma1_frs[-1], P_max=self.P_max,
                                            normalize=self.normalize_fr,
                                            eps=self.eps)[:, None]

                    psi_longest_support = compute_spatial_support(
                        psi_longest,
                        criterion_amplitude=self.criterion_amplitude)
                    if psi_longest_support == len(psi_longest):
                        # in zero padding we cut padding in half, which distorts
                        # the wavelet but negligibly relative to the
                        # scattering scale
                        if pad_diff == 0:
                            if self.pad_mode_fr != 'zero':
                                raise Exception(
                                    "got `pad_diff_max == 0` with"
                                    "`pad_mode_fr != 'zero'`, meaning "
                                    "`J_pad_frs_max_init` computed incorrectly.")
                            pad_diff_max = 0
                        else:
                            pad_diff_max = pad_diff - 1
                        break
                    elif len(psi_longest) == 2**self.N_fr_scales_min:
                        # smaller pad length is impossible
                        break
                    pad_diff += 1
            else:
                # this `else` isn't exclusive with the `if`
                # (i.e. in case `unrestricted_pad_fr==True`), but if `if` holds,
                # so will `else` (former's more restrictive)
                pad_diff = 0
                while True:
                    # in submaximal padding, a non-last wavelet may have longer
                    # support, so check all wavelets (still won't be near Nyquist
                    # but simplify logic)
                    for n1_fr in range(len(xi1_frs)):
                        #### Compute wavelet #################################
                        # fetch wavelet params, sample wavelet
                        xi, sigma = xi1_frs[n1_fr], sigma1_frs[n1_fr]
                        J_pad_fr = self.J_pad_frs_max_init - pad_diff

                        try:
                            _ = morlet_1d(2**J_pad_fr, xi, sigma,
                                          normalize=self.normalize_fr,
                                          P_max=self.P_max, eps=self.eps)
                        except ValueError as e:
                            if "division" not in str(e):
                                raise e
                            pad_diff_max = max(pad_diff - 1, 0)
                            break
                    if pad_diff_max is not None:
                        break
                    pad_diff += 1

        # only one must be set
        assert not (pad_diff_max is not None and
                    scale_diff_max_to_set_pad_min is not None)

        if pad_diff_max is not None:
            self.J_pad_frs_min_limit_due_to_psi = (self.J_pad_frs_max_init -
                                                   pad_diff_max)
            self.scale_diff_max_to_build = None
        elif scale_diff_max_to_set_pad_min is not None:
            self.J_pad_frs_min_limit_due_to_psi = None  # None for now
            self.scale_diff_max_to_build = (
                scale_diff_max_to_set_pad_min)
        else:
            # no limits (i.e. naturally computed J_pad_frs_min is correct)
            self.J_pad_frs_min_limit_due_to_psi = None
            self.scale_diff_max_to_build = None

    def compute_stride_fr(self):
        """See "Compute logic: stride, padding" in `core`."""
        self._compute_psi_fr_params()

        if self.out_3D:
            stride_at_max_fr = self._get_stride(
                scale_diff=0, N_fr_scale=self.N_fr_scales_max)[0]
            self.unpad_len_common_at_max_fr_stride = (
                self.ind_end_fr_max[stride_at_max_fr] -
                self.ind_start_fr_max[stride_at_max_fr])
        else:
            self.unpad_len_common_at_max_fr_stride = None

        # spinned stride -- main stride ######################################
        self.total_conv_stride_over_U1s = {}
        for N_fr_scale in self.N_fr_scales_unique[::-1]:
            scale_diff = self.N_fr_scales_max - N_fr_scale

            s = self._get_stride(scale_diff, N_fr_scale,
                                 self.unpad_len_common_at_max_fr_stride)
            self.total_conv_stride_over_U1s[scale_diff] = s

        # now for phi_f pairs ################################################
        self.total_conv_stride_over_U1s_phi = {}
        for scale_diff in self.total_conv_stride_over_U1s:
            if self.average_fr or not self.aligned:
                if self.aligned or self.sampling_phi_fr == 'resample':
                    s = self.log2_F
                else:
                    s = min(self.log2_F,
                            max(self.total_conv_stride_over_U1s[scale_diff]))
            else:
                s = 0
            self.total_conv_stride_over_U1s_phi[scale_diff] = s

        # sanity checks ######################################################
        for nm in ('total_conv_stride_over_U1s',
                   'total_conv_stride_over_U1s_phi'):
            for scale_diff, strides in getattr(self, nm).items():
                assert np.min(strides) >= 0, "{}, scale_diff={}\n{}".format(
                    nm, scale_diff, strides)

        # clarity assertions #################################################
        # phi stride <= spinned stride
        if self.average_fr:
            for scale_diff in self.total_conv_stride_over_U1s:
                s_spinned = max(self.total_conv_stride_over_U1s[scale_diff])
                s_phi     = self.total_conv_stride_over_U1s_phi[scale_diff]
                # must hold to not potentially require padding phi pairs
                # separately per J_pad_fr >= total_conv_stride_over_U1
                # (or padding spinned more just for sake of `phi_f` pairs).
                # no choice with `average_fr=False`, via e.g. `log2_F > J_fr`
                assert s_phi <= s_spinned, (s_phi, s_spinned)

        # for out_3D, stride is same on per-`scale_diff` basis, and spinned and
        # phi strides must match for scale_diff==0 (for `stride_ref` in unpadding)
        if self.out_3D:
            for scale_diff in self.total_conv_stride_over_U1s:
                assert len(set(self.total_conv_stride_over_U1s[scale_diff])
                           ) == 1, self.total_conv_stride_over_U1s
            s0 = self.total_conv_stride_over_U1s[0][0]
            s1 = self.total_conv_stride_over_U1s_phi[0]
            assert s0 == s1, (self.total_conv_stride_over_U1s,
                              self.total_conv_stride_over_U1s_phi)
            assert s0 <= self.log2_F

    def _get_stride(self, scale_diff, N_fr_scale,
                    unpad_len_common_at_max_fr_stride=None):
        assert N_fr_scale != -1

        # prefetch params ################################################
        if scale_diff not in self.psi_fr_params:
            # shouldn't occur otherwise
            assert self.sampling_psi_fr in ('exclude', 'recalibrate'), (
                scale_diff, self.psi_fr_params)
            # don't need to adjust `N_fr_scale`, since in the one place it's
            # used, it must refer to the actual scale, while the point of
            # `scale_diff` is to refer to the actual filterbank, which
            # this `scale_diff_ref` will do
            scale_diff_ref = max(self.psi_fr_params)
            # clarity assertion
            assert scale_diff_ref == self.scale_diff_max_to_build
        else:
            scale_diff_ref = scale_diff
        j1_frs_scale = self.psi_fr_params[scale_diff_ref]['j']
        n_n1_frs = len(j1_frs_scale)
        assert n_n1_frs > 0, (scale_diff_ref, self.psi_fr_params)

        # handle edge cases (cont'd) #####################################
        # resample_psi = bool(sampling_psi_fr in ('resample', 'exclude'))
        resample_phi = bool(self.sampling_phi_fr == 'resample')

        if (unpad_len_common_at_max_fr_stride is None and
                (self.out_3D and not self.aligned and not resample_phi)):
            return [self.log2_F] * n_n1_frs

        # get stride #####################################################
        if self.average_fr:
            if self.aligned:
                s = self.log2_F
                assert resample_phi
            else:
                if resample_phi:
                    if self.out_3D:
                        s = self.log2_F
                    else:
                        s = []
                        for n1_fr in range(n_n1_frs):
                            max_filter_stride = max(self.log2_F,
                                                    j1_frs_scale[n1_fr])
                            _s = max_filter_stride
                            s.append(_s)
                else:
                    if self.out_3D:
                        # avoid N_fr dependence
                        N_fr_max_at_scale = 2**N_fr_scale
                        # except at max scale case
                        N_fr = min(N_fr_max_at_scale, self.N_frs_max)
                        min_stride_to_unpad_like_max = max(0, math.ceil(math.log2(
                            N_fr / unpad_len_common_at_max_fr_stride)))
                        s = min_stride_to_unpad_like_max
                    else:
                        s = []
                        for n1_fr in range(n_n1_frs):
                            # min: nonzero unpad
                            # max: not oversampled
                            _s = max(min(self.log2_F, N_fr_scale),
                                     j1_frs_scale[n1_fr])
                            nonzero_unpad_but_not_oversampled = _s
                            s.append(nonzero_unpad_but_not_oversampled)
        else:
            if self.aligned:
                s = 0
            else:
                s = j1_frs_scale
            assert not self.out_3D

        if not isinstance(s, list):
            s = [s] * n_n1_frs
        assert len(s) == n_n1_frs, (len(s), n_n1_frs)
        return s

    def compute_scale_and_stride_logic(self, for_validation=False):
        if not for_validation:
            self.n1_fr_subsamples, self.log2_F_phis, self.log2_F_phi_diffs = (
                self._compute_scale_and_stride_logic(self.psi_fr_params))
            return

        # ensure params match the actual filterbank's ########################
        # this ensures `psi_fr_factory` built as predicted, i.e. didn't change
        # `n1_fr_subsamples`. These params compute before the filterbank because
        # they're needed to compute padding, which is needed for the filterbank.
        # unpack from the filters
        psi_fr_params = {}
        # ensure filters built every scale_diff requested by psi_fr_params
        for scale_diff in self.psi_fr_params:
            psi_fr_params[scale_diff] = {}
            psi_id = self.psi_ids[scale_diff]
            for field in self.psi_fr_params[scale_diff]:
                psi_fr_params[scale_diff][
                    field] = self.psi1_f_fr_up[field][psi_id]

        n1_fr_subsamples, log2_F_phis, log2_F_phi_diffs = (
            self._compute_scale_and_stride_logic(psi_fr_params))

        for pair in n1_fr_subsamples:
            for scale_diff in self.n1_fr_subsamples[pair]:
                a = n1_fr_subsamples[pair][scale_diff]
                b = self.n1_fr_subsamples[pair][scale_diff]
                assert a == b, (a, b)

    def _compute_scale_and_stride_logic(self, psi_fr_params):
        n1_fr_subsamples = {}
        log2_F_phis = {}
        log2_F_phi_diffs = {}

        # spinned ############################################################
        n1_fr_subsamples['spinned'] = {}
        log2_F_phis['spinned'] = {}
        log2_F_phi_diffs['spinned'] = {}

        for scale_diff in self.scale_diffs_unique:
            n1_fr_subsamples['spinned'][scale_diff] = []
            log2_F_phis['spinned'][scale_diff] = []
            log2_F_phi_diffs['spinned'][scale_diff] = []

            scale_diff_ref = (scale_diff if scale_diff in psi_fr_params else
                              max(psi_fr_params))
            for n1_fr, j1_fr in enumerate(psi_fr_params[scale_diff_ref]['j']):
                total_conv_stride_over_U1 = self.total_conv_stride_over_U1s[
                    scale_diff][n1_fr]

                # n1_fr_subsample & log2_F_phi
                if self.average_fr and not self.average_fr_global:
                    if self.sampling_phi_fr == 'resample':
                        log2_F_phi_diff = 0
                    elif self.sampling_phi_fr == 'recalibrate':
                        log2_F_phi_diff = max(self.log2_F -
                                              total_conv_stride_over_U1, 0)
                    log2_F_phi = self.log2_F - log2_F_phi_diff

                    # Maximum permitted subsampling before conv w/ `phi_f_fr`.
                    # This avoids distorting `phi` (aliasing), or for
                    # 'recalibrate', requesting one that doesn't exist.
                    max_subsample_before_phi_fr = log2_F_phi
                    sub_adj = min(j1_fr, total_conv_stride_over_U1,
                                  max_subsample_before_phi_fr)
                else:
                    log2_F_phi, log2_F_phi_diff = None, None
                    sub_adj = (j1_fr if self.average_fr_global else
                               min(j1_fr, total_conv_stride_over_U1))
                n1_fr_subsamples['spinned'][scale_diff].append(sub_adj)
                log2_F_phis['spinned'][scale_diff].append(log2_F_phi)
                log2_F_phi_diffs['spinned'][scale_diff].append(log2_F_phi_diff)

        # phi ################################################################
        n1_fr_subsamples['phi'] = {}
        log2_F_phis['phi'] = {}
        log2_F_phi_diffs['phi'] = {}

        if self.average_fr_global_phi:
            # not accounted; compute at runtime
            pass
        else:
            for scale_diff in self.scale_diffs_unique:
                total_conv_stride_over_U1_phi = (
                    self.total_conv_stride_over_U1s_phi[scale_diff])
                n1_fr_subsample = total_conv_stride_over_U1_phi

                log2_F_phi = (self.log2_F
                              if (not self.average_fr and self.aligned) else
                              total_conv_stride_over_U1_phi)
                log2_F_phi_diff = self.log2_F - log2_F_phi

                n1_fr_subsamples['phi'][scale_diff] = n1_fr_subsample
                log2_F_phis['phi'][scale_diff] = log2_F_phi
                log2_F_phi_diffs['phi'][scale_diff] = log2_F_phi_diff

        return n1_fr_subsamples, log2_F_phis, log2_F_phi_diffs

    def compute_unpadding_fr(self):
        """See `help(scf.compute_padding_fr)`.

        This is JTFS's equivalent, repeated for each N_fr, as JTFS is a
        multi-input network (along freq). We then compute 3D unpad indices by
        taking worst case for each subsampling - longest unpad as opposed to
        shortest, as latter tosses out coeffs.
        """
        self.ind_start_fr, self.ind_end_fr = [], []
        for n2, N_fr in enumerate(self.N_frs):
            if N_fr != 0:
                _ind_start, _ind_end = self._compute_unpadding_params(N_fr)
            else:
                _ind_start, _ind_end = [], []

            self.ind_start_fr.append(_ind_start)
            self.ind_end_fr.append(_ind_end)

        # compute out_3D params
        self.ind_start_fr_max, self.ind_end_fr_max = [], []
        for sub in range(self.log2_F + 1):
            start_max, end_max = [max(idxs[n2][sub]
                                      for n2 in range(len(self.N_frs))
                                      if len(idxs[n2]) != 0)
                                  for idxs in
                                  (self.ind_start_fr, self.ind_end_fr)]
            self.ind_start_fr_max.append(start_max)
            self.ind_end_fr_max.append(end_max)

    def _compute_unpadding_params(self, N_fr):
        # compute unpad indices for all possible subsamplings
        ind_start, ind_end = [0], [N_fr]
        for j in range(1, max(self.J_fr, self.log2_F) + 1):
            ind_start.append(0)
            ind_end.append(math.ceil(ind_end[-1] / 2))
        return ind_start, ind_end

    def compute_padding_fr(self):
        """Built around stride. The pipeline is as follows:

          1. Compute `J_pad_frs_max_init`, which is max padding under
            "standard" scattering configuration (all 'resample').
          2. Sample frequential filterbank at `2**J_pad_frs_max_init`,
             store in `psi1_f_fr_up_init`. (`scf.create_init_psi_filters`)
          3. Compute `psi_fr_params` from `psi1_f_fr_up_init`, in accords
             with `sampling_psi_fr`. (`scf._compute_psi_fr_params`)
          4. Compute `J_pad_frs_min_limit_due_to_psi` to restrict `J_pad_frs_min`
             as filterbank quality assurance.
             (`scf._compute_J_pad_frs_min_limit_due_to_psi`)
          5. Maybe delete keys in `psi_fr_params`

        Entering this method, for each `N_fr_scale`,

          - Compute minimum required padding to avoid boundary effects (which
            also ensures all wavelets fully decay), `min_to_pad_bound_effs`
          - Accounting for `max_pad_factor_fr`, we get `pad_boundeffs`
          - Accounting for `out_3D`, we get `pad_3D`, which overrides
            `pad_boundeffs`
          - Accounting for stride overrides all above quantities
          - Accounting for `J_pad_frs_min_limit_due_to_psi` overrides all above
            quantities

        We then assert that the resulting `J_pad_frs` is non-increasing, which
        subsequent methods assume to avoid complications.

        Padding (also stride) is computed on per-`2**N_fr_scale` basis rather than
        per-`N_fr` as latter greatly complicates implementation for little gain.
        Different `N_fr` within same `N_fr_scale` can yield different `J_pad_fr`,
        which requires additional filter indexing to track

        Relevant attributes
        -------------------

          - `pad_left_fr, ind_start_fr`: always zero since we right-pad
          - `pad_right_fr`: computed to avoid boundary effects *for each* `N_frs`.
          - `ind_end_fr`: computed to undo `pad_right_fr`
          - `ind_end_fr_max`: maximum unpad index across all `n2` for a given
            subsampling factor. E.g.:

            ::

                n2 = (0, 1, 2)
                J_fr = 3 --> j_fr = (0, 1, 2, 3)
                ind_end_fr = [[32, 16, 8, 4],
                              [29, 14, 7, 3],
                              [33, 16, 8, 4]]
                ind_end_fr_max = [33, 16, 8, 4]

            Ensures same unpadded freq length for `out_3D=True` without losing
            information. Unused for `out_3D=False`.
        """
        self.J_pad_frs = {}
        for N_fr_scale in self.N_fr_scales_unique[::-1]:
            # check for reuse
            scale_diff = self.N_fr_scales_max - N_fr_scale
            if (self.scale_diff_max_to_build is not None and
                scale_diff > self.scale_diff_max_to_build):
                # account for `scale_diff_max_to_build`
                # reuse max `scale_diff`'s
                self.J_pad_frs[scale_diff] = self.J_pad_frs[
                    self.scale_diff_max_to_build]
                if self.J_pad_frs_min_limit_due_to_psi is None:
                    self.J_pad_frs_min_limit_due_to_psi = self.J_pad_frs[
                        scale_diff]
                continue

            # compute padding ################################################
            # compute pad for bound effs
            min_to_pad_bound_effs = self._get_min_to_pad_bound_effs(
                N_fr_scale, scale_diff)
            if self.unrestricted_pad_fr:
                pad_boundeffs = min_to_pad_bound_effs
            else:
                pad_boundeffs = min(min_to_pad_bound_effs,
                                    N_fr_scale +
                                    self.max_pad_factor_fr[scale_diff])

            # account for out_3D
            if not self.out_3D:
                J_pad_fr = pad_boundeffs
            else:
                # smallest `J_pad_fr` such that
                #   2**J_pad_fr / s >= unpad_len_common_at_max_fr_stride;
                #   s = 2**total_conv_stride_over_U1s[n2][0]
                # i.e.
                #   2**J_pad_fr >= unpad_len_common_at_max_fr_stride * 2**s [*1]
                #
                # for `aligned and out_3D`, this will always end up being near
                # `J_pad_frs_max`, since `s = log2_F` always, and `J_pad_fr`
                # in [*1] is forced to the max. "Near" as in, this is the
                # `min_to_pad_stride` criterion, < `min_to_pad_bound_effs`.
                #
                # the `[0]` isn't special, all indices are same for a given `n2`,
                # due to `out_3D`
                pad_3D = math.ceil(math.log2(
                    self.unpad_len_common_at_max_fr_stride *
                    2**self.total_conv_stride_over_U1s[scale_diff][0]
                ))
                # `pad_3D` overrides `max_pad_factor_fr`
                J_pad_fr = max(pad_boundeffs, pad_3D)
                # but not `min_to_pad_bound_effs`
                # this would save compute - often small, sometimes significant -
                # but require implementing re-padding and alternative intermediate
                # frequential unpadding.
                # J_pad_fr = min(J_pad_fr, min_to_pad_bound_effs)

            # account for stride
            s_spinned = max(self.total_conv_stride_over_U1s[scale_diff])
            s_phi     = self.total_conv_stride_over_U1s_phi[scale_diff]
            min_to_pad_stride = max(s_spinned, s_phi)
            J_pad_fr = max(J_pad_fr, min_to_pad_stride)

            # account for phi 'resample'
            if self.sampling_phi_fr == 'resample':
                # This isn't necessary but would require handling phi construction
                # with subsampling > phi's length, or doing global averaging for
                # log2_F_phi > J_pad_fr, both doable but currently not done.
                # This is automatically satisfied by `max(, stride)` except in the
                # `not average_fr and aligned` case
                J_pad_fr = max(J_pad_fr, self.log2_F)

            # account for `J_pad_frs_min_limit_due_to_psi`
            if self.J_pad_frs_min_limit_due_to_psi is not None:
                J_pad_fr = max(J_pad_fr, self.J_pad_frs_min_limit_due_to_psi)

            # insert
            self.J_pad_frs[scale_diff] = J_pad_fr

        # ensure integer type (non-numpy to not confuse backends)
        for k, v in self.J_pad_frs.items():
            self.J_pad_frs[k] = int(v)

        # validate padding computed so far, as `psi_fr_factory` relies on it
        self._assert_nonincreasing_J_pad_frs()

        # compute related params #############################################
        self.pad_left_fr, self.pad_right_fr = [], []
        for n2, N_fr in enumerate(self.N_frs):
            if N_fr != 0:
                scale_diff = self.N_fr_scales_max - math.ceil(math.log2(N_fr))
                J_pad_fr = self.J_pad_frs[scale_diff]
                (_pad_left, _pad_right
                 ) = self._compute_padding_params(J_pad_fr, N_fr)
            else:
                _pad_left, _pad_right = -1, -1

            self.pad_left_fr.append(_pad_left)
            self.pad_right_fr.append(_pad_right)

        # compute `scale_diff_max_to_build`
        if self.scale_diff_max_to_build is None:
            # clarity assertion
            assert (self.sampling_psi_fr == 'resample' or

                    (self.sampling_psi_fr == 'exclude' and
                     max(self.scale_diffs) in self.psi_fr_params) or

                    (self.sampling_psi_fr == 'recalibrate' and
                     self.scale_diff_max_recalibrate is None)

                    ), (self.sampling_psi_fr, self.scale_diffs,
                        list(self.psi_fr_params),
                        self.scale_diff_max_recalibrate)

            if self.sampling_psi_fr == 'resample':
                # scale before the first scale to drop below minimum
                # is the limiting scale
                J_pad_frs_min = min(self.J_pad_frs.values())
                for scale_diff, J_pad_fr in self.J_pad_frs.items():
                    if J_pad_fr == J_pad_frs_min:
                        self.scale_diff_max_to_build = scale_diff
                        break

                # clear `psi_fr_params` of scales we won't build
                scale_diffs = list(self.psi_fr_params)
                for scale_diff in scale_diffs:
                    if scale_diff > self.scale_diff_max_to_build:
                        del self.psi_fr_params[scale_diff]

    def _get_min_to_pad_bound_effs(self, N_fr_scale, scale_diff):
        common_kw = dict(normalize=self.normalize_fr, P_max=self.P_max,
                         eps=self.eps)
        ca = dict(criterion_amplitude=self.criterion_amplitude)

        # psi ################################################################
        if self.sampling_psi_fr == 'resample':
            # note, for `average_fr=False`, `min_to_pad_phi` can be `0` for
            # spinned pairs, but this may necessiate two FFTs on |U1 * psi2|, one
            # on larger padded (for the phi_f pairs) and other on its trimming.
            # Not a concern for `J_fr >= log2_F and sampling_psi_fr == 'resample'`
            min_to_pad_psi = self._pad_fr_psi
        elif self.sampling_psi_fr in ('exclude', 'recalibrate'):
            # fetch params
            xis, sigmas = [self.psi_fr_params[scale_diff][field]
                           for field in ('xi', 'sigma')]

            psi_fn = lambda N: morlet_1d(N, xis[-1], sigmas[-1], **common_kw)
            N_min_psi = compute_minimum_required_length(
                psi_fn, N_init=2**N_fr_scale, **ca)
            min_to_pad_psi = compute_spatial_support(psi_fn(N_min_psi), **ca)
            if self.pad_mode_fr == 'zero':
                min_to_pad_psi //= 2

        # phi ################################################################
        if self.sampling_phi_fr == 'resample':
            min_to_pad_phi = self._pad_fr_phi
        elif self.sampling_phi_fr == 'recalibrate':
            spinned_diffs = self.log2_F_phi_diffs['spinned'][scale_diff]
            phi_diff      = self.log2_F_phi_diffs['phi'][scale_diff]
            if None in spinned_diffs:
                assert not self.average_fr
                log2_F_phi_diff = phi_diff
            else:
                # lower -> greater log2_F -> greater pad, take worst case
                log2_F_phi_diff = min(min(spinned_diffs), phi_diff)

            sigma_low = self.sigma0 / self.F
            sigma_low_F = sigma_low * 2**log2_F_phi_diff

            phi_fn = lambda N: gauss_1d(N, sigma_low_F, **common_kw)
            N_min_phi = compute_minimum_required_length(
                phi_fn, N_init=2**N_fr_scale, **ca)
            min_to_pad_phi = compute_spatial_support(phi_fn(N_min_phi), **ca)
            if self.pad_mode_fr == 'zero':
                min_to_pad_phi //= 2

        # final ##############################################################
        min_to_pad = max(min_to_pad_phi, min_to_pad_psi)
        min_to_pad_bound_effs = N_and_pad_2_J_pad(2**N_fr_scale, min_to_pad)
        return min_to_pad_bound_effs

    def _compute_padding_params(self, J_pad, N_fr):
        pad_left = 0
        pad_right = 2**J_pad - pad_left - N_fr

        # sanity check
        pad_diff = self.J_pad_frs_max_init - J_pad
        assert pad_diff >= 0, "%s > %s | %s" % (
            J_pad, self.J_pad_frs_max_init, N_fr)
        # return
        return pad_left, pad_right

    def _compute_J_pad_fr(self, N_fr, Q):
        """Depends on `N_frs`, `max_pad_factor_fr`, `sampling_phi_fr`,
        `sampling_psi_fr`, and common filterbank params.

        `min_to_pad` is computed for both `phi` and `psi` in case latter has
        greater time-domain support (stored as `_pad_fr_phi` and `_pad_fr_psi`).

          - 'resample': will use original `_pad_fr_phi` and/or `_pad_fr_psi`
          - 'recalibrate' / 'exclude': will divide by difference in dyadic scale,
            e.g. `_pad_fr_phi / 2`.

        `recompute=True` will force computation from `N_frs` alone, independent
        of `J_pad_frs_max` and `min_to_pad_fr_max`, and per
        `sampling_* = 'resample'`.
        """
        min_to_pad, pad_phi, pad_psi1, _ = compute_minimum_support_to_pad(
            N_fr, self.J_fr, Q, self.F, pad_mode=self.pad_mode_fr,
            normalize=self.normalize_fr, r_psi=self.r_psi_fr,
            **self.get_params('sigma0', 'P_max', 'eps', 'criterion_amplitude'))
        if self.average_fr_global_phi:
            min_to_pad = pad_psi1  # ignore phi's padding
            pad_phi = 0
        J_pad_ideal = N_and_pad_2_J_pad(N_fr, min_to_pad)

        # adjust per `max_pad_factor_fr` and warn if needed
        # must do this to determine `xi_min` later. if "ideal pad" amount is
        # of interest, it should be another variable
        if not self.unrestricted_pad_fr:
            N_fr_scale = math.ceil(math.log2(N_fr))
            scale_diff = self.N_fr_scales_max - N_fr_scale

            # edge case: `phi_t` pair (_J_pad_fr_fo)
            if scale_diff < 0:
                mx = math.ceil(math.log2(self._n_psi1_f))
                assert mx > self.N_fr_scales_max, (mx, self.N_fr_scales_max)
                scale_diff = 0  # reuse `max_pad_factor_fr` for max spinned scale

            J_pad = min(J_pad_ideal,
                        N_fr_scale + self.max_pad_factor_fr[scale_diff])
            if J_pad_ideal - J_pad > 1:
                extent_txt = ('Severe boundary'
                              if J_pad_ideal - J_pad > 2 else
                              'Boundary')
                warnings.warn(f"{extent_txt} effects and filter distortion "
                              "expected per insufficient temporal padding; "
                              "recommended higher `max_pad_factor_fr` or lower "
                              "`J_fr` or `F`.")
        else:
            J_pad = J_pad_ideal

        return J_pad, min_to_pad, pad_phi, pad_psi1

    def get_params(self, *args):
        return {k: getattr(self, k) for k in args}

    def _assert_nonincreasing_J_pad_frs(self):
        prev_pad = 999
        for pad in self.J_pad_frs.values():
            if pad > prev_pad:
                raise Exception("w yielded padding that's "
                                "greater for lesser `N_fr_scale`; this is "
                                "likely to yield incorrect or undefined behavior."
                                "\nJ_pad_frs=%s" % self.J_pad_frs)
            prev_pad = pad
        # additionally assert we didn't exceed J_pad_frs_max_init
        assert all(p <= self.J_pad_frs_max_init for p in self.J_pad_frs.values()
                   ), (self.J_pad_frs, self.J_pad_frs_max_init)


# filterbank builders ########################################################

def psi_fr_factory(psi_fr_params, N_fr_scales_unique, N_fr_scales_max, J_pad_frs,
                   sampling_psi_fr='resample', scale_diff_max_to_build=None,
                   normalize_fr='l1', criterion_amplitude=1e-3, sigma0=0.13,
                   P_max=5, eps=1e-7):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Every filter is provided as a dictionary with "meta" information; see
    `help(wavespin.scattering1d.filter_bank.scattering_filter_factory)`.

    Parameters
    ----------
    psi_fr_params : dict[int:dict[str:list]]
        Filterbank parameters, structured as

            `scale_diff: {field: [*values]}`

        e.g.

            {0: {'xi': [0.4, 0.2, 0.1],
                 'sigma': [.1, .05, .025],
                 ...},
             1: {...},
             ...}

        Alongside `J_pad_frs`, will determine `psi_ids`.

    N_fr_scales_unique : list[int]
        Used for iterating over `scale_diff`, and a sanity check.

    N_fr_scales_max : int
        Used to compute `scale_diff` from `N_fr_scale` from `N_fr_scales_unique`.

    J_pad_frs : dict[int: int]
        Used to compute `padded_len`s, the lengths at which `morlet_1d`
        is sampled.

    sampling_psi_fr : str['resample', 'recalibrate', 'exclude']
        Used for a sanity check in case of 'exclude'.
        See `help(TimeFrequencyScattering1D)`.
        In terms of effect on maximum `j` per `n1_fr`:

            - 'resample': no variation (by design, all temporal properties are
              preserved, including subsampling factor).

            - 'recalibrate': `j1_fr_max` is (likely) lesser with greater
              `scale_diff` (by design, temporal width is tailored to
              `2**N_fr_scale`). The limit, however, is set by
              `sigma_max_to_min_max_ratio` (see its docs).

            - 'exclude': approximately same as 'recalibrate'. By design, excludes
              temporal widths above `2**N_fr_scale * width_exclude_ratio`, which
              is likely to reduce `j1_fr_max` with greater `scale_diff`.
                - It's "approximately" same because center frequencies and
                  widths are different; depending how loose our alias tolerance
                  (`criterion_amplitude`), they're exactly the same.

    scale_diff_max_to_build : int
        Controls max built `scale_diff`, and is used for a sanity check.

    normalize_fr, criterion_amplitude, sigma0, P_max, eps:
        Parameters for building `morlet_1d` or its meta.

    Returns
    -------
    psi1_f_fr_up : dict[int: list[tensor[float]],
                        str: dict[int: list[int/float]]]
        Contains band-pass filters of frequential scattering with "up" spin,
        and their meta - keyed by `meta` fields, and `psi_id`:

            {psi_id: [psi_arr0, psi_arr1, ...],
             meta0: {psi_id0: [...], ...}}

        This differs from `Scattering1D.psi1_f`. See `psi_id` below.

        These filters are not subsampled, as they do not receive subsampled
        inputs (but their *outputs*, i.e. convolutions, can be subsampled).
        Difference in filter length is never a time-domain subsampling, but is
        rather *trimming* (`sampling_psi_fr='resample'`), not being included at
        all ('exclude'), or a change in `xi` and `sigma` to match `N_fr_scale`
        ('recalibrate').
        Different filterbank lengths are indexed by `psi_id` (see `psi_id` below).

        Example: `J_fr = 2`, lists hold permitted subsampling factors for
                 respective filters (i.e. for after convolving):

            - 'resample':
                0: [2, 1, 0]  # `psi_id=0: [psis[-1], psis[-2], psis[-3]]`
                1: [2, 1, 0]
                2: [2, 1, 0]
            - 'recalibrate':
                0: [2, 1, 0]
                1: [1, 1, 0]
                2: [0, 0, 0]
            - 'exclude':
                0: [2, 1, 0]
                1: [1, 0]
                2: [0]

    psi1_f_fr_dn : dict[int: list[tensor[float]],
                        str: dict[int: list[int/float]]]
        Same as `psi1_f_fr_up` but with "down" spin (anti-analytic, whereas "up"
        is analytic wavelet).

    psi_ids : dict[int: int]
        See `psi_id` below.

    psi_id
    ------

    Indexes frequential filterbanks, returning list of filters. Is a function of
    `scale_diff`, that maps to `params, J_pad_fr`:

       `(xis, sigmas), J_pad_fr = psi_ids_fn(scale_diff)`

    The idea is, we may desire different filterbanks for different `N_fr_scale`s.
    We cannot index them through

      - `J_pad_fr`, since different `scale_diff` may yield same `J_pad_fr`.
      - `scale_diff`, since different `scale_diff` may have the same
        `params, J_pad_fr`, which yields duplication.
      - `params`, since same `params` may apply to different `scale_diff`.

    `psi_id` resolves these conflicts.

      - Higher `psi_id` always corresponds to higher `scale_diff`.
      - One `psi_id` may correspond to multiple `scale_diff`, but never
        multiple `params, J_pad_fr`.
      - Same `params` *or* `J_pad_fr` may correspond to one `psi_id`, but never
        both.

    Logic summary, in pseudocode:

    ::

        psi_ids = {0: 0}
        for scale_diff in scale_diffs:
            params, J_pad_fr = compute_stuff()
            if (params, J_pad_fr) not in built_params_and_J_pad_frs:
                psi_ids[scale_diff] = max(psi_ids.values()) + 1
            else:
                # this `else` may also execute per other conditionals
                psi_ids[scale_diff] = max(psi_ids.values())
    """
    # compute the spectral parameters of the filters
    xi1_frs, sigma1_frs, j1_frs, is_cqt1_frs = [
        {scale_diff: psi_fr_params[scale_diff][field]
         for scale_diff in psi_fr_params}
        for field in ('xi', 'sigma', 'j', 'is_cqt')]

    ###########################################################################
    def repeat_last_built_id(scale_diff, scale_diff_last_built):
        psi_ids[scale_diff] = psi_ids[scale_diff_last_built]

    # instantiate the dictionaries which will contain the filters
    psi1_f_fr_up, psi1_f_fr_dn = {}, {}

    # needed to handle different `N_fr_scale` having same `J_pad_fr` & params,
    # and different params having same `J_pad_fr`. Hence,
    #     params, J_pad_fr = psi_id(N_fr_scale)
    psi_ids = {}

    params_built = []
    scale_diff_last_built = -1
    for N_fr_scale in N_fr_scales_unique[::-1]:
        scale_diff = N_fr_scales_max - N_fr_scale
        first_scale = bool(scale_diff == 0)

        # ensure we compute at valid `N_fr_scale`
        if scale_diff in psi_ids:
            # already built
            continue
        elif first_scale:
            # always built
            pass
        elif N_fr_scale == -1:
            # invalid scale
            continue
        elif scale_diff not in j1_frs:
            # universal cue to reuse last built filterbank.
            # varied `padded_len` as func of `J_pad_frs` & `scale_diff` (hence
            # reuse not sufficing) is ruled out by later assertion, `pads_built`.
            # Frontend also assures this in `compute_padding_fr`, and/or via
            # `scale_diff_max_to_build`
            repeat_last_built_id(scale_diff, scale_diff_last_built)
            continue
        elif (scale_diff_max_to_build is not None and
                scale_diff > scale_diff_max_to_build):
            assert scale_diff not in j1_frs, j1_frs
            # ensure `scale_diff` didn't exceed a global maximum.
            # subsequent `scale_diff` are only greater, so
            # we could `break`, but still need to `repeat_last_built_id`
            repeat_last_built_id(scale_diff, scale_diff_last_built)
            continue

        # extract params to iterate
        n_psi = len(j1_frs[scale_diff])
        params = []
        for n1_fr in range(n_psi):
            xi    = xi1_frs[   scale_diff][n1_fr]
            sigma = sigma1_frs[scale_diff][n1_fr]
            padded_len = 2**J_pad_frs[scale_diff]  # repeat for all n1_fr
            params.append((xi, sigma, padded_len))

        # if already built, point to it and don't rebuild
        if params in params_built:
            repeat_last_built_id(scale_diff, scale_diff_last_built)
            continue

        # build wavelets #####################################################
        psis_up = []
        for n1_fr in range(n_psi):
            # ensure we compute at valid `N_fr_scale`, `n1_fr`
            if first_scale:
                # always built
                pass
            elif (sampling_psi_fr == 'exclude' and
                    # this means the wavelet (sampled at J_pad_frs_max_init)
                    # exceeded max permitted width at this scale,
                    # i.e. `width > 2**N_fr_scale * width_exclude_ratio`
                    (scale_diff not in j1_frs or
                     n1_fr > len(j1_frs[scale_diff]) - 1)):
                # subsequent `scale_diff` are only greater, and
                # hence `len(j1_frs[scale_diff])` only lesser

                # above conditional shouldn't be possible to satisfy but is
                # kept for clarity
                raise Exception("impossible iteration")
                break  # would happen if condition was met; kept for clarity

            #### Compute wavelet #############################################
            # fetch wavelet params, sample wavelet
            xi, sigma, padded_len = params[n1_fr]

            # expand dim to multiply along freq like (2, 32, 4) * (32, 1)
            psi = morlet_1d(padded_len, xi, sigma, normalize=normalize_fr,
                            P_max=P_max, eps=eps)[:, None]
            psis_up.append(psi)

        # if all `n1_fr` built, register & append to filterbank ##############
        if first_scale:
            psi_ids[0] = 0
        else:
            psi_ids[scale_diff] = psi_ids[scale_diff_last_built] + 1
        params_built.append(params)
        scale_diff_last_built = scale_diff

        # append to filterbank
        psi_id = psi_ids[scale_diff]
        psi1_f_fr_up[psi_id] = psis_up

        # compute spin down by time-reversing spin up in frequency domain
        psi1_f_fr_dn[psi_id] = [time_reverse_fr(p) for p in psis_up]

    ##########################################################################

    # ensure every unique `N_fr_scale` has a filterbank
    n_scales = len(N_fr_scales_unique)
    assert len(psi_ids) == n_scales, (psi_ids, N_fr_scales_unique)

    # validate `scale_diff_max_to_build`
    if scale_diff_max_to_build is not None:
        assert scale_diff_last_built <= scale_diff_max_to_build

    # guaranteed by "J_pad_frs is non-increasing", which was already asserted
    # in `compute_padding_fr()` (base_frontend), but include here for clarity;
    # much of above logic assumes this
    pads_built = [math.log2(params[0][2]) for params in params_built]
    assert min(pads_built) == J_pad_frs[scale_diff_last_built], (
        pads_built, J_pad_frs)

    # assert "higher psi_id -> lower scale"
    # since `params, J_pad_fr = psi_ids_fn(scale_diff)`,
    # and `params` and `J_pad_fr` are always same for same `scale_diff`,
    # then `params` or `J_pad_fr` (and hence `psi_id`) only change if
    # `scale_diff` changes. Namely,
    #  - if `psi_id` changed, then either `params` or `J_pad_fr` changed
    #    (which can only happen if `scale_diff` changed),
    #  - if `scale_diff` changed, then `psi_id` doesn't necessarily change,
    #    since neither of `params` or `J_pad_fr` necessarily change.
    # Thus, "psi_id changed => scale_diff changed", but not conversely.
    prev_scale_diff, prev_psi_id = -1, -1
    for scale_diff in psi_ids:
        if psi_id == prev_psi_id:
            # only check against changing `psi_id`, but still track `scale_diff`
            prev_scale_diff = scale_diff
            continue
        assert scale_diff > prev_scale_diff, (scale_diff, prev_scale_diff)
        prev_scale_diff, prev_psi_id = scale_diff, psi_id

    # instantiate for-later params and reusable kwargs
    ca   = dict(criterion_amplitude=criterion_amplitude)
    s0ca = dict(criterion_amplitude=criterion_amplitude, sigma0=sigma0)

    # Embed meta information within the filters ##############################
    for psi_f in (psi1_f_fr_dn, psi1_f_fr_up):
        for field in ('xi', 'sigma', 'j', 'is_cqt', 'width', 'support',
                      'scale', 'bw', 'bw_idxs', 'peak_idx'):
            if field not in psi_f:
                psi_f[field] = {}
            for scale_diff, psi_id in psi_ids.items():
                if psi_id in psi_f[field]:
                    continue

                psi_f[field][psi_id] = []
                for n1_fr in range(len(psi_f[psi_id])):
                    pf = psi_f[psi_id][n1_fr]
                    if field == 'width':
                        N_fr_scale = N_fr_scales_max - scale_diff
                        v = compute_spatial_width(pf, N=2**N_fr_scale, **s0ca)
                    elif field == 'support':
                        v = compute_spatial_support(pf, **ca)
                    elif field == 'scale':
                        v = width_2_scale(psi_f['width'][psi_id][n1_fr])
                    elif field == 'bw':
                        v = compute_bandwidth(pf, **ca)
                    elif field == 'bw_idxs':
                        v = compute_bw_idxs(pf, **ca)
                    elif field == 'peak_idx':
                        v = np.argmax(np.abs(pf))
                    else:
                        v = psi_fr_params[scale_diff][field][n1_fr]
                    psi_f[field][psi_id].append(v)

    # return results
    return psi1_f_fr_up, psi1_f_fr_dn, psi_ids


def phi_fr_factory(J_pad_frs_max_init, J_pad_frs, F, log2_F, unrestricted_pad_fr,
                   pad_mode_fr, sampling_phi_fr='resample', average_fr=None,
                   average_fr_global_phi=None, aligned=None,
                   criterion_amplitude=1e-3, normalize_fr='l1', sigma0=0.13,
                   P_max=5, eps=1e-7):
    """
    Builds in Fourier the lowpass Gaussian filters used for JTFS.

    Every filter is provided as a dictionary with "meta" information; see
    `help(wavespin.scattering1d.filter_bank.scattering_filter_factory)`.

    Parameters
    ----------
    J_pad_frs_max_init : int
        `2**J_pad_frs_max_init` is the largest length of the filters.

    J_pad_frs : dict[int: int]
        Lengths at which to sample `gauss_1d`. For 'recalibrate', also
        controls time-domain widths (see "Build logic").

    F : int
        Scale of invariance (in linear units). Controls `sigma` of `phi`
        via `sigma0 / F`. For 'recalibrate', `log2_F_phi_diff` means
        `sigma0 / F / 2**log2_F_phi_diff` (see "Build logic").

    log2_F : int
        Scale of invariance (log2(prevpow2(F))). Controls maximum dyadic scale
        and subsampling factor for all `phi`.

    unrestricted_pad_fr : bool
        Used for a quality check; `True` ensures `phi` decays sufficiently
        (but not necessarily fully if `pad_mode_fr=='zero'`; see code).
        Including steps outside this function, `max_pad_factor != None`
        such filter distortion considerations.

    pad_mode_fr : str
        Used for a quality check.

    sampling_phi_fr : str['resample', 'recalibrate']
        See "Build logic" below.

    average_fr : bool
        Used for a sanity check.

    average_fr_global_phi : bool
        Used for a quality check.

    aligned : bool
        Used for a sanity check.

    criterion_amplitude : float
        Used to compute `phi` meta.

    sigma0 : float
        Together with `F`, determines width (sigma) of `phi`: `sigma = sigma0/F`.

    normalize_fr : str
        `gauss_1d` parameter `normalize`.

    P_max, eps : float, float
        `gauss_1d` parameters.

    Returns
    -------
    phi_f_fr : dict[int: dict[int: list[tensor[float]]],
                    str: dict[int: dict[int: list[int]], float]]
        Contains the low-pass filter at all possible lengths, scales of
        invariance, and subsampling factors:

            phi_f_fr[invariance_scale][input_length][input_subsampling]
            <= e.g. =>
            phi_f_fr[~log2_F][~J_pad_fr_max][n1_fr_subsample]

        and corresponding meta:

            phi_f_fr['support'][log2_F_phi_diff][pad_diff][n1_fr_subsample]
            phi_f_fr['sigma'][log2_F_phi_diff']  # doesn't vary w/ other params

        This differs from `Scattering1D.phi_f`. See "Build logic" for details.

    Build logic
    -----------
    We build `phi` for every possible input length (`2**J_pad_fr`), input
    subsampling factor (`n1_fr_subsample1`), and ('recalibrate' only) scale
    of invariance. Structured as

        `phi_f_fr[log2_F_phi_diff][pad_diff][sub]`

    `log2_F_diff == log2_F - log2_F_phi`. Hence,
                      higher `log2_F_phi_diff`
                                 <=>
            greater *contraction* (time-domain) of original phi
                                 <=>
            lower `log2_F_phi`, lower permitted max subsampling

    Higher `pad_diff` is a greater *trimming* (time-domain) of the corresponding
    lowpass.

      - 'resample': `log2_F_diff == 0`, always.
      - 'recalibrate': `log2_F_diff` spans from `min(log2_F, J_pad_fr)` to
        `log2_F`. Not all of these will be used, but we compute every possible
        combination to avoid figuring out which will be.

    'resample' enforces global scale of invariance (`==F` for all coefficients).
    'recalibrate' "follows" the scale of `psi`, as controlled by
    `total_conv_stride_over_U1`, averaging less for finer filterbanks.
    """
    # compute the spectral parameters of the filters
    sigma_low = sigma0 / F
    N_init = 2**J_pad_frs_max_init
    zero_stride_globally = bool(not average_fr and aligned)

    def compute_all_subsamplings(phi_f_fr, pad_diff, log2_F_phi, log2_F_phi_diff):
        for sub in range(1, 1 + log2_F_phi):
            ps = fold_filter_fourier(phi_f_fr[log2_F_phi_diff][pad_diff][0],
                                          nperiods=2**sub)
            phi_f_fr[log2_F_phi_diff][pad_diff].append(ps)

    # initial lowpass
    phi_f_fr = {0: {}}
    # expand dim to multiply along freq like (2, 32, 4) * (32, 1)
    phi_f_fr[0][0] = [gauss_1d(N_init, sigma_low, P_max=P_max, eps=eps)[:, None]]
    compute_all_subsamplings(phi_f_fr, pad_diff=0, log2_F_phi=log2_F,
                             log2_F_phi_diff=0)
    # reusable
    common_kw = dict(normalize=normalize_fr, P_max=P_max, eps=eps)

    # lowpass filters at all possible input lengths ##########################
    pads_iterated = []
    for J_pad_fr in list(J_pad_frs.values())[::-1]:
        if J_pad_fr == -1:
            continue
        # avoid recomputation
        if J_pad_fr in pads_iterated:
            continue
        pads_iterated.append(J_pad_fr)

        # validate J_pad_fr
        if sampling_phi_fr == 'resample' and not zero_stride_globally:
            # guaranteed by design:
            #  - 'resample': total_conv_stride_over_U1 >= log2_F
            #  - J_pad_fr = max(, total_conv_stride_over_U1)
            # exception is `not average_fr and aligned`, but we force
            # `max(, log2_F)` in frontend
            assert J_pad_fr >= log2_F, (J_pad_fr, log2_F)

        pad_diff = J_pad_frs_max_init - J_pad_fr

        if sampling_phi_fr == 'resample':
            phi_f_fr[0][pad_diff] = [
                gauss_1d(2**J_pad_fr, sigma_low, **common_kw)[:, None]]
            # dedicate separate filters for *subsampled* as opposed to *trimmed*
            # inputs (i.e. `n1_fr_subsample` vs `J_pad_frs_max_init - J_pad_fr`)
            compute_all_subsamplings(phi_f_fr, pad_diff, log2_F_phi=log2_F,
                                     log2_F_phi_diff=0)

        elif sampling_phi_fr == 'recalibrate':
            # These won't differ from plain subsampling but we still
            # build each `log2_F_phi_diff` separately with its own subsampling
            # to avoid excessive bookkeeping.
            # `phi[::factor] == gauss_1d(N // factor, sigma_low * factor)`
            # when not aliased.

            # by design, (J_pad_frs[scale_diff] >=
            #             total_conv_stride_over_U1s[scale_diff])
            max_log2_F_phi = min(log2_F, J_pad_fr)
            min_log2_F_phi_diff = log2_F - max_log2_F_phi
            # `== log2_F` means `log2_F_phi == 0`
            max_log2_F_phi_diff = log2_F

            # not all these filters will be used, and some will be severely
            # under-padded (e.g. log2_F_phi == J_pad_fr), but we compute them
            # anyway to avoid having to determine what will and won't be used
            for log2_F_phi_diff in range(min_log2_F_phi_diff,
                                         max_log2_F_phi_diff + 1):
                log2_F_phi = log2_F - log2_F_phi_diff
                sigma_low_F = sigma_low * 2**log2_F_phi_diff
                if log2_F_phi_diff not in phi_f_fr:
                    phi_f_fr[log2_F_phi_diff] = {}
                if pad_diff in phi_f_fr[log2_F_phi_diff]:
                    # already computed
                    continue

                phi_f_fr[log2_F_phi_diff][pad_diff] = [
                    gauss_1d(2**J_pad_fr, sigma_low_F, **common_kw)[:, None]]
                compute_all_subsamplings(phi_f_fr, pad_diff, log2_F_phi,
                                         log2_F_phi_diff)

        # validate phi
        if (sampling_phi_fr == 'resample' and unrestricted_pad_fr and
                pad_mode_fr != 'zero' and not average_fr_global_phi):
            # `==` means width is already too great for own length,
            # so lesser length will distort lowpass.
            # This is automatically averted with `max_pad_factor=None`.
            # However, since the zero-padded case takes min_to_pad // 2,
            # this won't hold any longer (but we permit it anyway since
            # the difference is tolerable).
            phi_fr = phi_f_fr[0][pad_diff][0]
            phi_support = compute_spatial_support(
                phi_fr, criterion_amplitude=criterion_amplitude)
            assert phi_support < phi_fr.size, (phi_support, phi_fr.size)

    # reorder keys as ascending
    if sampling_phi_fr == 'recalibrate':
        _phi_f_fr = phi_f_fr
        phi_f_fr = {}
        log2_F_phi_diffs = sorted(list(_phi_f_fr))
        for log2_F_phi_diff in log2_F_phi_diffs:
            phi_f_fr[log2_F_phi_diff] = {}
            pad_diffs = sorted(list(_phi_f_fr[log2_F_phi_diff]))
            for pad_diff in pad_diffs:
                phi_f_fr[log2_F_phi_diff][pad_diff] = _phi_f_fr[
                    log2_F_phi_diff][pad_diff]

    # embed meta info in filters #############################################
    meta_fields = ('xi', 'sigma', 'j', 'width', 'support', 'scale', 'bw',
                   'bw_idxs', 'peak_idx')
    for field in meta_fields:
        phi_f_fr[field] = {}
    ca = dict(criterion_amplitude=criterion_amplitude)
    s0ca = dict(sigma0=sigma0, **ca)

    for log2_F_phi_diff in phi_f_fr:
        if not isinstance(log2_F_phi_diff, int):
            continue
        log2_F_phi = log2_F - log2_F_phi_diff

        xi_fr_0 = 0.
        sigma_fr_0 = sigma_low * 2**log2_F_phi_diff
        j_0 = log2_F_phi

        for field in meta_fields:
            phi_f_fr[field][log2_F_phi_diff] = []

        phi_f_fr['xi'   ][log2_F_phi_diff] = xi_fr_0
        phi_f_fr['sigma'][log2_F_phi_diff] = sigma_fr_0
        phi_f_fr['j'    ][log2_F_phi_diff] = j_0
        for field in ('width', 'support', 'scale', 'bw', 'bw_idxs', 'peak_idx'):
            phi_f_fr[field][log2_F_phi_diff] = {}

        for pad_diff in phi_f_fr[log2_F_phi_diff]:
            for field in ('width', 'support', 'scale', 'bw',
                          'bw_idxs', 'peak_idx'):
                phi_f_fr[field][log2_F_phi_diff][pad_diff] = []

            for sub in range(len(phi_f_fr[log2_F_phi_diff][pad_diff])):
                # should halve with subsequent `sub`, but compute exactly.
                phi = phi_f_fr[log2_F_phi_diff][pad_diff][sub]
                width   = compute_spatial_width(phi, N=phi.size, **s0ca)
                support = compute_spatial_support(phi, **ca)
                scale   = width_2_scale(width)
                bw      = compute_bandwidth(phi, **ca, c=0)
                bw_idxs = compute_bw_idxs(phi, **ca, c=0)

                phi_f_fr['width'   ][log2_F_phi_diff][pad_diff].append(width)
                phi_f_fr['support' ][log2_F_phi_diff][pad_diff].append(support)
                phi_f_fr['scale'   ][log2_F_phi_diff][pad_diff].append(scale)
                phi_f_fr['bw'      ][log2_F_phi_diff][pad_diff].append(bw)
                phi_f_fr['bw_idxs' ][log2_F_phi_diff][pad_diff].append(bw_idxs)
                phi_f_fr['peak_idx'][log2_F_phi_diff][pad_diff].append(0)

    # return results
    return phi_f_fr


def _recalibrate_psi_fr(max_original_width, xi1_frs, sigma1_frs, j1_frs,
                        is_cqt1_frs, N_fr_scales_max, N_fr_scales_unique,
                        sigma_max_to_min_max_ratio, width_exclude_ratio,
                        J_pad_frs_max_init, criterion_amplitude,
                        normalize_fr, P_max, eps):
    # recalibrate filterbank to each `scale_diff`
    # `scale_diff=0` is the original input length, no change needed
    xi1_frs_new, sigma1_frs_new, j1_frs_new, is_cqt1_frs_new = (
        {0: xi1_frs}, {0: sigma1_frs}, {0: j1_frs}, {0: is_cqt1_frs})

    # set largest scale wavelet's scale as reference, but don't allow exceeding
    # largest input's scale (which is possible)
    max_original_scale_thresholded = width_threshold(
        math.ceil(math.log2(max_original_width)), width_exclude_ratio)
    recalibrate_scale_ref = min(max_original_scale_thresholded,
                                N_fr_scales_max)

    sigma_max = max(sigma1_frs)
    sigma_min = min(sigma1_frs)
    xi_min    = min(xi1_frs)
    sigma_min_max = sigma_max / sigma_max_to_min_max_ratio
    scale_diff_max = None
    scale_diff_prev = 0

    def set_params(scale_diff, empty=False, reuse_original=False):
        for param in (xi1_frs_new, sigma1_frs_new, j1_frs_new, is_cqt1_frs_new):
            if empty:
                param[scale_diff] = []
            elif reuse_original:
                param[scale_diff] = param[0].copy()

    for N_fr_scale in N_fr_scales_unique[::-1]:
        scale_diff = N_fr_scales_max - N_fr_scale
        if scale_diff == 0:
            # leave original unchanged
            continue
        elif (max_original_width <
              width_threshold(N_fr_scale, width_exclude_ratio)):
            # no need to recalibrate, reuse 'resample' just like 'exclude' does.
            # psi_fr_factory will skip this via `if params in params_built`
            set_params(scale_diff, reuse_original=True)
            continue
        # else: now the original scale exceeds our threshold, so recalibrate
        # to create finer filters

        recalibrate_scale_diff = recalibrate_scale_ref - N_fr_scale
        assert recalibrate_scale_diff != 0
        factor = 2**recalibrate_scale_diff

        # contract largest temporal width of any wavelet by 2**scale_diff,
        # but not above sigma_max/sigma_max_to_min_max_ratio
        new_sigma_min = sigma_min * factor
        if new_sigma_min > sigma_min_max:
            scale_diff_max = scale_diff_prev
            break

        # init for this scale
        set_params(scale_diff, empty=True)

        # halve distance from existing xi_max to .5 (max possible)
        new_xi_max = .5 - (.5 - max(xi1_frs)) / factor
        # new xi_min scale by scale diff
        new_xi_min = xi_min * np.sqrt(factor)
        # logarithmically distribute
        new_xi = np.logspace(np.log10(new_xi_min), np.log10(new_xi_max),
                             len(xi1_frs), endpoint=True)[::-1]

        xi1_frs_new[scale_diff].extend(new_xi)
        new_sigma = np.logspace(np.log10(new_sigma_min),
                                np.log10(sigma_max),
                                len(xi1_frs), endpoint=True)[::-1]
        sigma1_frs_new[scale_diff].extend(new_sigma)
        for xi, sigma in zip(new_xi, new_sigma):
            # here our j computation is on filter lengths potentially different
            # from what'll be used, as we don't yet know `J_pad_frs`; that's also
            # the case for higher-`scale_diff` 'exclude'.
            # this is fine however as longer's `j`s are conservative relative to
            # shorter's, and the bound's difference is within one sample
            pf = morlet_1d(2**J_pad_frs_max_init, xi, sigma,
                           normalize_fr, P_max, eps)
            bw_idxs = compute_bw_idxs(pf, criterion_amplitude)
            new_j = compute_max_dyadic_subsampling(pf, bw_idxs)

            j1_frs_new[scale_diff].append(new_j)
            is_cqt1_frs_new[scale_diff].append(False)
        scale_diff_prev = scale_diff

    return (xi1_frs_new, sigma1_frs_new, j1_frs_new, is_cqt1_frs_new,
            scale_diff_max)

#### helpers #################################################################
def width_threshold(N_fr_scale, width_exclude_ratio):
    return 2**N_fr_scale * width_exclude_ratio


def time_reverse_fr(x):
    """Time-reverse in frequency domain by swapping all bins (except dc);
    assumes frequency is along first axis. x(-t) <=> X(-w).
    """
    out = np.zeros_like(x)
    out[0] = x[0]
    out[1:] = x[:0:-1]
    return out
