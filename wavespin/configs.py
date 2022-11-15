# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Global configurations for internal parameters.

Editable directly via `wavespin.CFG`, e.g. `wavespin.CFG['S1D']['sigma0'] = 0.1`.

Supported
---------

    - 'S1D':  see `help(wavespin.Scattering1D)`
    - 'JTFS': see `help(wavespin.TimeFrequencyScattering1D)`
    - 'VIZ':  see `help(wavespin.visuals)`

Handling configs
----------------

    - Change defaults for all sessions: edit `CFG` in `wavespin/configs.py`.
    - Restore defaults: `wavespin.configs.restore_defaults()` (see its docs).
    - Get defaults: `wavespin.configs.get_defaults()` (see its docs).
"""
from copy import deepcopy

_README = "See `help(wavespin.configs)`. (This key-value pair does nothing.)"""
CFG = {'README': _README}

# Visuals ####################################################################
SMALL_GLOBAL_SCALE = 0.6
CFG['VIZ'] = dict(
    figsize=(12., 7.),
    dpi=72,
    title=dict(loc='left', fontsize=17, weight='bold'),
    xlabel=dict(fontsize=15, weight='bold'),
    ylabel=dict(fontsize=15, weight='bold'),
    tick_params=dict(labelsize=11),

    # alsp specify fallback fonts
    long_title_fontfamily=(50, 'arial'),
    global_scale=1.,
)

# Scattering objects #########################################################
# See docs in `wavespin.scattering1d.frontend.base_frontend`.
CFG['S1D'] = dict(
    sigma0=.13,
    P_max=5,
    eps=1e-7,
    criterion_amplitude=1e-3,
)
CFG['JTFS'] = dict(
    sigma_max_to_min_max_ratio=1.2,
    width_exclude_ratio=0.5,
    N_fr_p2up=None,
    N_frs_min_global=8,
    **CFG['S1D'],
)

# Captured initial user defaults, DO NOT EDIT! ###############################
_USER_DEFAULTS = deepcopy(CFG)

# Library defaults, DO NOT EDIT! #############################################
_LIB_DEFAULTS = {
    'VIZ': dict(
        figsize=(12, 7),
        dpi=72
    ),
    'S1D': dict(
        sigma0=0.13,
        P_max=5,
        eps=1e-7,
        criterion_amplitude=1e-3,
    ),
}
_LIB_DEFAULTS.update({
    'JTFS': dict(
        sigma_max_to_min_max_ratio=1.2,
        width_exclude_ratio=0.5,
        N_fr_p2up=None,
        N_frs_min_global=8,
        **_LIB_DEFAULTS['S1D'],
    ),
})
_LIB_DEFAULTS['README'] = _README

# Methods ####################################################################
def restore_defaults(library=False):
    """Restores all configurations to their *user* defaults (i.e. what's set
    in `wavespin.configs.py`).

    `library=True` restores to library's defaults (doesn't change user defaults).
    """
    # first clear all keys
    names = list(CFG)
    for name in names:
        if name != 'README':
            keys = list(CFG[name])
            for k in keys:
                del CFG[name][k]
        del CFG[name]

    # now restore
    restorer = _LIB_DEFAULTS if library else _USER_DEFAULTS
    CFG.update(restorer)


def get_defaults(library=False):
    """Fetches copy of *user* configuration defaults.
    `library=True` fetches copy of library's defaults.
    """
    return deepcopy(_LIB_DEFAULTS if library else
                    _USER_DEFAULTS)
