# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ...frontend.base_frontend import ScatteringBase
import math

from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering1d import timefrequency_scattering1d
from ..filter_bank import scattering_filter_factory
from ..filter_bank_jtfs import _FrequencyScatteringBase1D
from ..scat_utils import (compute_border_indices, compute_padding,
                          compute_minimum_support_to_pad, compute_meta_scattering)


class ScatteringBase1D(ScatteringBase):
    def __init__(self, shape, J=None, Q=8, T=None, average=True, oversampling=0,
                 backend=None):
        super(ScatteringBase1D, self).__init__()
        self.shape = shape
        self.J = J
        self.Q = Q if isinstance(Q, tuple) else (Q, 1)
        self.T = T
        self.average = average
        self.oversampling = oversampling
        self.backend = backend

        self.out_type = "array"
        self.max_order = 2

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """
        self.sigma0 = 0.1
        self.P_max = 5
        self.eps = 1e-7
        self.criterion_amplitude = 1e-3

        # handle `shape`
        assert isinstance(self.shape, int)
        self.N = self.shape

        # dyadic scale of N, also min possible padding
        self.N_scale = math.ceil(math.log2(self.N))

        # handle `J`
        if self.J is None:
            self.J = self.N_scale - 2

        # ensure 2**max(J) <= nextpow2(N)
        Np2up = 2**self.N_scale
        if 2**self.J > Np2up:
            raise ValueError(("2**J cannot exceed input length (rounded up to "
                              "pow2) (got {} > {})".format(
                                  2**max(self.J), Np2up)))

        # check T or set default
        if self.T is None:
            self.T = 2**self.J
        elif self.T > Np2up:
            raise ValueError(("The temporal support T of the low-pass filter "
                              "cannot exceed (nextpow2 of) input length. "
                              "Got {} > {})").format(self.T, self.N))
        # log2_T, global averaging
        self.log2_T = math.floor(math.log2(self.T))
        self.average_global = bool(self.T == Np2up and self.average)

        # Compute the minimum support to pad (ideally)
        min_to_pad, pad_phi, pad_psi1, pad_psi2 = compute_minimum_support_to_pad(
            self.N, self.J, self.Q, self.T,
            sigma0=self.sigma0, P_max=self.P_max, eps=self.eps,
            criterion_amplitude=self.criterion_amplitude)

        J_pad_ideal = math.ceil(math.log2(self.N + min_to_pad))
        self.J_pad = J_pad_ideal

        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.N)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.log2_T, self.J, self.pad_left, 2**self.J_pad - self.pad_right)

    def create_filters(self):
        # Create the filters
        self.phi_f, self.psi1_f, self.psi2_f = scattering_filter_factory(
            self.N, self.J_pad, self.J, self.Q, self.T,
            criterion_amplitude=self.criterion_amplitude,
            sigma0=self.sigma0, P_max=self.P_max, eps=self.eps)

    def scattering(self, x):
        x = self._handle_input(x)
        Scx = scattering1d(
            x, self.backend, self.log2_T,
            self.psi1_f, self.psi2_f, self.phi_f,
            self.pad_left, self.pad_right, self.ind_start, self.ind_end,
            self.oversampling, self.max_order,
            self.average, self.out_type, self.average_global
        )
        return Scx

    def _handle_input(self, x):
        if len(x.shape) < 1:
            raise ValueError("x should be at least 1D. Got %s" % str(x))
        if 'array' in self.out_type and not self.average:
            raise ValueError("Options `average=False` and `'array' in out_type` "
                             "are mutually incompatible. "
                             "Please set out_type='list' or 'dict:list'")
        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)
        return x

    def meta(self):
        return compute_meta_scattering(self.J_pad, self.J, self.Q, self.T,
                                       self.max_order)


class TimeFrequencyScatteringBase1D():
    def __init__(self, J_fr=None, Q_fr=2, F=None, oversampling_fr=0):
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.oversampling_fr = oversampling_fr

        self.average_fr = True

    def build(self):
        """Check args and instantiate `_FrequencyScatteringBase1D` object
        (which builds filters).
        """
        # handle `F`; this is processed further in `scf`
        if self.F is None:
            # default to one octave (Q wavelets per octave, J octaves,
            # approx Q*J total frequency rows, so averaging scale is `Q/total`)
            # F is processed further in `_FrequencyScatteringBase1D`
            self.F = self.Q[0]

        # compute primitive N_frs; note this mirrors Scattering1D's second-order
        N_frs = []
        for n2 in range(len(self.psi2_f)):
            j2 = self.psi2_f[n2]['j']
            N_frs.append(0)
            for n1 in range(len(self.psi1_f)):
                j1 = self.psi1_f[n1]['j']
                if j2 > j1:
                    N_frs[-1] += 1

        # frequential scattering object ######################################
        # number of psi1 filters
        self._n_psi1_f = len(self.psi1_f)

        self.scf = _FrequencyScatteringBase1D(
            N_frs, self.J_fr, self.Q_fr, self.F,
            average_fr=self.average_fr, oversampling_fr=self.oversampling_fr,
            _n_psi1_f=self._n_psi1_f, backend=self.backend)

    def scattering(self, x):
        x = self._handle_input(x)
        Scx = timefrequency_scattering1d(
            x, self.backend.unpad, self.backend,
            self.J, self.log2_T, self.psi1_f, self.psi2_f, self.phi_f, self.scf,
            self.pad_left, self.pad_right, self.ind_start, self.ind_end,
            self.oversampling, self.oversampling_fr, self.average, self.out_type
        )
        return Scx


__all__ = ['ScatteringBase1D', 'TimeFrequencyScatteringBase1D']
