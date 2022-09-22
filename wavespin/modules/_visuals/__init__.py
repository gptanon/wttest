# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    plt = None
    # make class so imports don't fail since classes inherit this
    class animation():
        TimedAnimation = object

    import warnings
    warnings.warn("`wavespin.visuals` requires `matplotlib` installed.")
