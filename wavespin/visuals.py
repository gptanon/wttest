# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Convenience visual methods.

Some figure properties are globally configurable via `wavespin.CFG['VIZ']`:

    - 'figsize': matplotlib kwarg controlling width and height in inches.
        - Affects most visuals.
        - For primitives (`plot`, `scat`, `imshow`, etc), only has effect if
          `fig` or `ax` isn't passed (so defaults don't override existing values).
        - If no effect on a visual, try `matplotlib.rcParams['figure.figsize']`.

    - 'dpi': matplotlib kwarg controlling dots per inch of plot (more -> HD).
        - If no effect on a visual, try `matplotlib.rcParams['figure.dpi']`.
        - Second bullet of 'figsize' applies.

    - 'title': dict containing matplotlib kwargs passed to `plt.title()`
        - e.g.: `CFG['VIZ']['title']['fontsize'] = 15`

    - 'long_title_fontfamily': tuple controlling font family for long titles
        - first element specifies "long" threshold
        - second element specifies font family to use above "long" threshold

    - 'global_scale': float, will scale figure and label sizes by this amount
      (e.g. `0.5` makes everything half as big). Not all visuals are supported,
      and may not work perfectly. Default is `1`.
"""

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

# handle runtime configs -----------------------------------------------------
def _set_small_global_scale():
    from . import configs
    from .configs import CFG
    CFG['VIZ']['global_scale'] = configs.SMALL_GLOBAL_SCALE


def adjust_configs_based_on_runtime_type():
    """Adjusts visuals configurations based on runtime type (e.g. Jupyter).

        - Jupyter: runs magic commands to trigger matplotlib's inline backend,
          and switch to 'retina' display for higher quality
        - Spyder: uses scaling of `1`, as the library was developed in Spyder.
          For all else, uses a smaller scaling.
    """
    try:
        # if we failed to import IPython, it means we're not in Spyder, so
        # still use smaller scaling
        _adjust_configs_based_on_runtime_type()
    except ImportError:  # no-cov
        _set_small_global_scale()


def _adjust_configs_based_on_runtime_type():
    import IPython

    def type_of_script():  # no-cov
        try:
            ipy_str = str(type(IPython.get_ipython())).lower()
            if 'spyder' in ipy_str:
                return 'spyder'
            elif 'zmqshell' in ipy_str:
                return 'jupyter'
            elif 'terminal' in ipy_str:
                return 'ipython'
            else:
                return 'terminal'
        except:
            return 'terminal'


    if type_of_script() == 'spyder':
        return
    else:  # no-cov
        _set_small_global_scale()

        if type_of_script() == 'jupyter':
            commands = [
                "matplotlib inline",
                "config InlineBackend.figure_format = 'retina'",
            ]
            ipython = IPython.get_ipython()
            for cmd in commands:
                magic_name, *line = cmd.split(' ')
                line = ' '.join(line)
                ipython.run_line_magic(magic_name, line)


def setup_fonts():
    """Adds the "Arial" font family to matplotlib, if matplotlib is installed
    and doesn't already support it.
    """
    try:  # no-cov
        import matplotlib.font_manager as fm
        from pathlib import Path

        supported = {nm.lower() for nm in fm.get_font_names()}
        if 'arial' not in supported:
            path = Path(Path(__file__).parent, 'utils', '_fonts',
                        'arialbd.ttf').resolve()
            fm.fontManager.addfont(path)

    except ImportError:
        pass
