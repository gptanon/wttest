# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import numpy as np
import math

from .filter_bank import scattering_filter_factory
from ..frontend.base_frontend import ScatteringBase


class _FrequencyScatteringBase1D(ScatteringBase):
    """Attribute object for TimeFrequencyScatteringBase1D for frequential
    scattering part of JTFS.
    """
    # note: defaults are set in `wavespin.scattering1d.frontend.base_frontend.py`,
    # `DEFAULT_KWARGS`
    def __init__(self, N_frs, J_fr=None, Q_fr=None, F=None,
                 average_fr=None, oversampling_fr=None, _n_psi1_f=None,
                 backend=None):
        super(_FrequencyScatteringBase1D, self).__init__()
        self.N_frs = N_frs
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.average_fr = average_fr
        self.oversampling_fr = oversampling_fr
        self._n_psi1_f = _n_psi1_f
        self.backend = backend

        self.build()
        self.compute_unpadding_fr()
        self.create_psi_filters()

    def build(self):
        self.sigma0 = .1
        self.P_max = 5
        self.eps = 1e-7
        self.criterion_amplitude = 1e-3

        self.N_frs_max = max(self.N_frs)
        self.N_fr_scales_max = math.ceil(math.log2(self.N_frs_max))
        self.J_pad_fr = self.N_fr_scales_max + 1

        # ensure `2**J_fr <= nextpow2(N_frs_max)`
        if self.J_fr is None:
            # default to `max - 2` if possible, but no less than `3`,
            # and no more than `max`
            self.J_fr = min(max(self.N_fr_scales_max - 2, 3),
                            self.N_fr_scales_max)
        elif self.J_fr > self.N_fr_scales_max:  # no-cov
            raise ValueError(("2**J_fr cannot exceed maximum number of frequency "
                              "rows (rounded up to pow2) in joint scattering "
                              "(got {} > {})".format(
                                  2**(self.J_fr), 2**self.N_fr_scales_max)))

        # check `F`
        if self.F > 2**self.N_fr_scales_max:
            raise ValueError("The temporal support F of the low-pass filter "
                             "cannot exceed maximum number of frequency rows "
                             "(rounded up to pow2) in joint scattering "
                             "(got {} > {})".format(
                                 self.F, 2**self.N_fr_scales_max))
        self.log2_F = math.floor(math.log2(self.F))

    def create_psi_filters(self):
        """Reuses Scattering1D's constructors to build filters that'll
        scatter along frequency.
        """
        self.phi_f_fr, self.psi1_f_fr_up, _ = scattering_filter_factory(
            self.N_frs_max, self.J_pad_fr, self.J_fr, self.Q_fr, self.F,
            criterion_amplitude=self.criterion_amplitude,
            sigma0=self.sigma0, P_max=self.P_max, eps=self.eps)

        # make down by time-reversing up, also expand dims for freq scat
        self.psi1_f_fr_dn = []
        for n1_fr in range(len(self.psi1_f_fr_up)):
            self.psi1_f_fr_dn.append({})

            p1up = self.psi1_f_fr_up[n1_fr]
            p1dn = self.psi1_f_fr_dn[n1_fr]
            p1up[0] = p1up[0][:, None]
            p1dn[0] = time_reverse_fr(p1up[0])

            # copy meta
            for key in p1up:
                if not isinstance(key, int):
                    p1dn[key] = p1up[key]

        # expand lowpass dims
        for key in self.phi_f_fr:
            if isinstance(key, int):
                self.phi_f_fr[key] = self.phi_f_fr[key][:, None]

    def compute_unpadding_fr(self):
        """This is JTFS's equivalent, repeated for each N_fr, as JTFS
        is a multi-input network (along freq). We then compute 3D unpad
        indices by taking worst case for each subsampling - longest unpad
        as opposed to shortest, as latter tosses out coeffs.
        """
        self.ind_start_fr, self.ind_end_fr = [], []
        for n2 in range(len(self.N_frs)):
            N_fr = self.N_frs[n2]
            if N_fr != 0:
                _ind_start, _ind_end = self._compute_unpadding_params(N_fr)
            else:
                _ind_start, _ind_end = [], []

            self.ind_start_fr.append(_ind_start)
            self.ind_end_fr.append(_ind_end)

        # compute out_3D params
        self.ind_start_fr_max, self.ind_end_fr_max = [], []
        for sub in range(self.log2_F + 1):
            if self.ind_start_fr[sub] != []:
                # take `max` across all `n2` that have this `sub`
                start_max = max(self.ind_start_fr[sub].values())
                end_max = max(self.ind_end_fr[sub].values())
            else:
                start_max, end_max = -1, -1

            self.ind_start_fr_max.append(start_max)
            self.ind_end_fr_max.append(end_max)

    def _compute_unpadding_params(self, N_fr):
        # compute unpad indices for all possible subsamplings
        ind_start = {0: 0}
        ind_end = {0: N_fr}
        for j in range(1, max(self.J_fr, self.log2_F) + 1):
            ind_start[j] = 0
            ind_end[j] = math.ceil(ind_end[j - 1] / 2)
        return ind_start, ind_end


def time_reverse_fr(x):
    """Time-reverse in frequency domain by swapping all bins (except dc);
    assumes frequency is along first axis. x(-t) <=> X(-w).
    """
    out = np.zeros_like(x)
    out[0] = x[0]
    out[1:] = x[:0:-1]
    return out
