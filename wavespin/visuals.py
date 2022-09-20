# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Convenience visual methods."""

from .modules._visuals import primitives
from .modules._visuals import static
from .modules._visuals import animated

from .modules._visuals.primitives import (
    plot,
    imshow,
    scat,
    plotscat,
    hist,
    plot_box,
)
from .modules._visuals.static import (
    filterbank_scattering,
    filterbank_jtfs_1d,
    filterbank_heatmap,
    viz_jtfs_2d,
    scalogram,
    energy_profile_jtfs,
    coeff_distance_jtfs,
    compare_distances_jtfs,
)
from .modules._visuals.animated import (
    gif_jtfs_2d,
    gif_jtfs_3d,
    viz_top_fdts,
    make_gif,
    viz_spin_2d,
    viz_spin_1d,
)
