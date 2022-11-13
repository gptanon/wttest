# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Coefficient and filterbank postprocessing, introspection, and manipulation."""

from .modules._toolkit import postprocessing
from .modules._toolkit import introspection
from .modules._toolkit import filterbank
from .modules._toolkit import signals
from .modules._toolkit import misc

from .modules._toolkit.postprocessing import (
   normalize,
   pack_coeffs_jtfs,
   drop_batch_dim_jtfs,
   jtfs_to_numpy,
)
from .modules._toolkit.introspection import (
    coeff_energy,
    coeff_distance,
    coeff_energy_ratios,
    est_energy_conservation,
    top_spinned,
    coeff2meta_jtfs,
    meta2coeff_jtfs,
    scattering_info,
)
from .modules._toolkit.filterbank import (
    fit_smart_paths,
    validate_filterbank_tm,
    validate_filterbank_fr,
    validate_filterbank,
    compute_lp_sum,
    fold_lp_sum,
    Decimate,
    _compute_e_fulls,
)
from .modules._toolkit.signals import (
    echirp,
    fdts,
    bag_o_waves,
    adversarial_am,
    adversarial_impulse_train,
    colored_noise,
)
from .modules._toolkit.misc import (
    fft_upsample,
    tensor_padded,
    find_shape,
    energy,
    l2,
    rel_l2,
    l1,
    rel_l1,
    rel_ae,
    make_eps,
)
