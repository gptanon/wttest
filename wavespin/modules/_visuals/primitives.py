# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Visuals primitives / messy code."""
import os
import numpy as np
from copy import deepcopy

from . import mpl, plt
from ... import CFG, configs
from ...utils.gen_utils import fill_default_args


def imshow(x, title=None, show=True, cmap=None, norm=None, abs=0,
           w=None, h=None, ticks=True, borders=True, aspect='auto',
           ax=None, fig=None, yticks=None, xticks=None, tick_params=None,
           xlabel=None, ylabel=None, newfig=False, do_gscale=True, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    ticks: False to not plot x & y ticks
    borders: False to not display plot borders
    w, h: rescale width & height
    kw: passed to `plt.imshow()`
    """
    fig, ax, got_fig_or_ax = _handle_fig_ax(fig, ax, newfig)

    if norm is None:
        mx = np.max(np.abs(x))
        vmin, vmax = ((-mx, mx) if not abs else
                      (0, mx))
    else:
        vmin, vmax = norm
    if cmap == 'none':  # no-cov
        cmap = None
    elif cmap is None:
        cmap = 'turbo' if abs else 'bwr'
    _kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect, **kw)

    if abs:
        ax.imshow(np.abs(x), **_kw)
    else:
        ax.imshow(x.real, **_kw)

    _handle_ticks(ticks, xticks, yticks, tick_params, ax)

    _title(title, ax, do_gscale=do_gscale)
    _scale_plot(fig, ax, got_fig_or_ax, show=False, w=w, h=h,
                xlabel=xlabel, ylabel=ylabel, auto_xlims=False,
                do_gscale=do_gscale)

    if not borders:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)

    if show:
        plt.show()


def plot(x, y=None, title=None, show=0, complex=0, abs=0, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None,
         xlabel=None, ylabel=None, xticks=None, yticks=None, ticks=True,
         tick_params=None, ax=None, fig=None, squeeze=True, auto_xlims=None,
         newfig=False, do_gscale=True, logx=False, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    complex: plot `x.real` & `x.imag`; 2=True & abs val envelope
    ticks: False to not plot x & y ticks
    w, h: rescale width & height
    logx: logscale x axis
    kw: passed to `plt.imshow()`
    """
    fig, ax, got_fig_or_ax = _handle_fig_ax(fig, ax, newfig)

    auto_xlims = _handle_auto_xlims(auto_xlims, x, y, logx)
    x, y = _handle_xy(x, y, squeeze)

    if complex:
        ax.plot(x, y.real, color='tab:blue', **kw)
        ax.plot(x, y.imag, color='tab:orange', **kw)
        if complex == 2:
            ax.plot(x, np.abs(y), color='k', linestyle='--', **kw)
    else:
        if abs:
            y = np.abs(y)
        ax.plot(x, y, **kw)

    # styling
    if vlines:
        _vhlines(vlines, kind='v', ax=ax)
    if hlines:
        _vhlines(hlines, kind='h', ax=ax)
    if abs and ylims is None:
        ylims = (0, y.max()*1.03)

    _handle_ticks(ticks, xticks, yticks, tick_params, ax, do_gscale=do_gscale)

    _title(title, ax, do_gscale=do_gscale)
    _scale_plot(fig, ax, got_fig_or_ax, show=show, w=w, h=h, xlims=xlims,
                ylims=ylims, xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims,
                do_gscale=do_gscale, logx=logx)


def scat(x, y=None, title=None, show=0, s=18, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None,
         complex=False, abs=False, xlabel=None, ylabel=None, xticks=None,
         yticks=None, ticks=1, tick_params=None, ax=None, fig=None,
         auto_xlims=None, newfig=False, do_gscale=True, **kw):
    fig, ax, got_fig_or_ax = _handle_fig_ax(fig, ax, newfig)

    auto_xlims = _handle_auto_xlims(auto_xlims, x, y)
    x, y = _handle_xy(x, y)

    def do_scatter(x, y):
        if complex:
            ax.scatter(x, y.real, s=s, **kw)
            ax.scatter(x, y.imag, s=s, **kw)
        else:
            if abs:
                y = np.abs(y)
            ax.scatter(x, y, s=s, **kw)

    x_2d = bool(hasattr(x, 'ndim') and x.ndim == 2)
    y_2d = bool(hasattr(y, 'ndim') and y.ndim == 2)
    if x_2d or y_2d:
        var = x if x_2d else y
        for v in var.T:
            varg = (v, y) if x_2d else (x, v)
            do_scatter(*varg)
    else:
        do_scatter(x, y)

    # styling
    if vlines:
        _vhlines(vlines, kind='v', ax=ax)
    if hlines:
        _vhlines(hlines, kind='h', ax=ax)

    _handle_ticks(ticks, xticks, yticks, tick_params, ax, do_gscale=do_gscale)

    _title(title, ax=ax)
    _scale_plot(fig, ax, got_fig_or_ax, show=show, w=w, h=h, xlims=xlims,
                ylims=ylims, xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims,
                do_gscale=do_gscale)


def plotscat(*args, **kw):
    show = kw.pop('show', False)
    plot(*args, **kw)
    scat(*args, **kw)
    if show:
        plt.show()


def hist(x, bins=500, title=None, show=0, stats=0, ax=None, fig=None, w=1, h=1,
         xlims=None, ylims=None, xlabel=None, ylabel=None, newfig=False,
         do_gscale=True):
    """Histogram. `stats=True` to print mean, std, min, max of `x`."""
    def _fmt(*nums):
        return [(("%.3e" % n) if (abs(n) > 1e3 or abs(n) < 1e-3) else
                 ("%.3f" % n)) for n in nums]

    fig, ax, got_fig_or_ax = _handle_fig_ax(fig, ax, newfig)

    x = np.asarray(x)
    _ = ax.hist(x.ravel(), bins=bins)
    _title(title, ax, do_gscale=do_gscale)
    _scale_plot(fig, ax, got_fig_or_ax, show=show, w=w, h=h, xlims=xlims,
                ylims=ylims, xlabel=xlabel, ylabel=ylabel, do_gscale=do_gscale)
    if show:
        plt.show()

    if stats:
        mu, std, mn, mx = (x.mean(), x.std(), x.min(), x.max())
        print("(mean, std, min, max) = ({}, {}, {}, {})".format(
            *_fmt(mu, std, mn, mx)))
        return mu, std, mn, mx


def plot_box(ctr, w, M=100, fig=None, ax=None, ymax=None, xmax=None, **pkw):
    x0_y01, x1_y01, x01_y0, x01_y1 = _get_box_data(ctr, w, M, ymax, xmax)

    ckw = dict(fig=fig, ax=ax, color='tab:red', linewidth=2,
               auto_xlims=0)
    for x_y in (x0_y01, x1_y01, x01_y0, x01_y1):
        plot(*x_y, **ckw, **pkw)


def _get_box_data(ctr, w, M, ymax=None, xmax=None):
    cy, cx = ctr
    wy, wx = w

    x0, x1, y0, y1 = [loc for loc in
                      (cx - wx, cx + wx, cy - wy, cy + wy)]
    # ensure in-bounds
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    if xmax is not None:
        x1 = min(x1, xmax)
    if ymax is not None:
        y1 = min(y1, ymax)

    M = 100
    x0_y01 = np.ones(M) * x0, np.linspace(y0, y1, M)
    x1_y01 = np.ones(M) * x1, np.linspace(y0, y1, M)
    x01_y0 = np.linspace(x0, x1, M), np.ones(M) * y0
    x01_y1 = np.linspace(x0, x1, M), np.ones(M) * y1

    return x0_y01, x1_y01, x01_y0, x01_y1


#### misc / utils ############################################################
def _vhlines(lines, kind='v', ax=None):
    lfn = getattr(plt if ax is None else ax, f'ax{kind}line')

    if not isinstance(lines, (list, tuple)):
        lines, lkw = [lines], {}
    elif isinstance(lines, (list, np.ndarray)):
        lkw = {}
    elif isinstance(lines, tuple):
        lines, lkw = lines
        lines = lines if isinstance(lines, (list, np.ndarray)) else [lines]
    else:  # no-cov
        raise ValueError("`lines` must be list or (list, dict) "
                         "(got %s)" % lines)
    if 'linewidth' not in lkw:
        lkw['linewidth'] = 1
    if 'color' not in lkw:  # no-cov
        lkw['color'] = 'tab:red'

    for line in lines:
        lfn(line, **lkw)


def _ticks(xticks, yticks, ax):
    def make_fmt(ticks):
        if all(isinstance(h, str) for h in ticks):
            return "%s"
        return ("%.d" if all(float(h).is_integer() for h in ticks) else
                "%.3g")

    targs = {'x': xticks, 'y': yticks}
    for k, ticks in targs.items():
        ax_fn = getattr(ax, f'set_{k}ticks')

        if not hasattr(ticks, '__len__') and not ticks:  # no-cov
            ax_fn([])
        else:
            if isinstance(ticks, tuple):
                ticks, kw = ticks
            else:
                kw = {}

            idxs = np.unique(np.linspace(0, len(ticks)-1, 8).astype('int32'))
            fmt = make_fmt(ticks)
            tl = [fmt % h for h in np.asarray(ticks)[idxs]]
            ax_fn(idxs, tl, **kw)


def _handle_ticks(ticks, xticks, yticks, tick_params, ax, do_gscale=True):
    ticks = ticks if isinstance(ticks, (list, tuple)) else (ticks, ticks)
    if not ticks[0]:
        ax.set_xticks([])
    if not ticks[1]:
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks, ax)

    if do_gscale:
        if tick_params is None:
            tick_params = _handle_tick_params({})
    if tick_params is not None:
        ax.tick_params(**tick_params)


def _title(title, ax=None, do_gscale=True):
    if title is None:
        return
    title, kw = (title if isinstance(title, tuple) else
                 (title, {}))
    kw = fill_default_args(kw, CFG['VIZ']['title'])
    if do_gscale:
        _handle_global_scale(kw)

    # handle long title
    len_th, long_family = CFG['VIZ']['long_title_fontfamily']
    if len(title) > len_th and 'fontfamily' not in kw:
        kw['fontfamily'] = long_family

    if ax:
        ax.set_title(str(title), **kw)
    else:
        plt.title(str(title), **kw)


def _scale_plot(fig, ax, got_fig_or_ax, show=False, ax_equal=False,
                w=None, h=None, xlims=None, ylims=None, xlabel=None, ylabel=None,
                auto_xlims=True, do_gscale=True, logx=False):
    # xlims, ylims
    if xlims:
        ax.set_xlim(*xlims)
    elif auto_xlims:
        ax.autoscale(tight=True, axis='x')
        xmin, xmax = ax.get_xlim()
        rng = xmax - xmin
        ax.set_xlim(xmin - .01 * rng, xmax + .01 * rng)
    if ylims:
        ax.set_ylim(*ylims)

    # height, width
    if got_fig_or_ax:
        width, height = fig.get_size_inches()  # if `not fig` it's `gcf()`
    else:
        width, height = CFG['VIZ']['figsize']
    if do_gscale:
        width *= _gscale()
        height *= _gscale()
    fig.set_size_inches(width * (w or 1), height * (h or 1))

    # dpi
    if not got_fig_or_ax:
        fig.set_dpi(CFG['VIZ']['dpi'])

    # xlabels, ylabels
    if xlabel is not None:
        if isinstance(xlabel, tuple):
            xlabel, xkw = xlabel
        else:
            xkw = {}
        xkw = fill_default_args(xkw, CFG['VIZ']['xlabel'])
        if do_gscale:
            _handle_global_scale(xkw)
        ax.set_xlabel(xlabel, **xkw)
    if ylabel is not None:
        if isinstance(ylabel, tuple):
            ylabel, ykw = ylabel
        else:
            ykw = {}
        ykw = fill_default_args(ykw, CFG['VIZ']['ylabel'])
        if do_gscale:
            _handle_global_scale(ykw)
        ax.set_ylabel(ylabel, **ykw)

    # log scaling
    if logx:
        ax.set_xscale(mpl.scale.LogScale(ax, base=2))

    # show
    if show:
        plt.show()


def _colorize_complex(z):
    """Map complex `z` to 3D array suitable for complex image visualization.

    Borrowed from https://stackoverflow.com/a/20958684/10133797
    """
    from colorsys import hls_to_rgb
    z = z / np.abs(z).max()
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 / (1 + r)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, l, s)
    c = np.array(c)
    c = c.transpose(1, 2, 0)
    return c


def _get_compute_pairs(pairs, meta):
    # enforce pair order
    if pairs is None:
        pairs_all = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                     'psi_t * phi_f', 'psi_t * psi_f_up', 'psi_t * psi_f_dn')
    else:
        pairs_all = pairs if not isinstance(pairs, str) else [pairs]
    compute_pairs = []
    for pair in pairs_all:
        if pair in meta['n']:
            compute_pairs.append(pair)
    return compute_pairs


def _format_ticks(ticks, max_digits=3):
    # `max_digits` not strict
    not_iterable = bool(not isinstance(ticks, (tuple, list, np.ndarray)))
    if not_iterable:
        ticks = [ticks]
    _ticks = []
    for tk in ticks:
        negative = False
        if tk < 0:
            negative = True
            tk = abs(tk)

        n_nondecimal = np.log10(tk)
        if n_nondecimal < 0:
            n_nondecimal = int(np.ceil(abs(n_nondecimal)) + 1)
            n_total = n_nondecimal + 2
            tk = f"%.{n_total - 1}f" % tk
        else:
            n_nondecimal = int(np.ceil(abs(n_nondecimal)))
            n_decimal = max(0, max_digits - n_nondecimal)
            tk = round(tk, n_decimal)
            tk = f"%.{n_decimal}f" % tk

        if negative:
            tk = "-" + tk
        _ticks.append(tk)
    if not_iterable:
        _ticks = _ticks[0]
    return _ticks


def _check_savepath(savepath, overwrite):
    if os.path.isfile(savepath):  # no-cov
        if not overwrite:
            raise RuntimeError("File already exists at `savepath`; "
                               "set `overwrite=True` to overwrite.\n"
                               "%s" % str(savepath))
        else:
            # delete if exists
            os.unlink(savepath)


def _get_phi_for_psi_id(jtfs, psi_id):
    """Returns `phi_f_fr` at appropriate length, but always of scale `log2_F`."""
    scale_diff = [scale_diff for scale_diff, _psi_id in jtfs.scf.psi_ids.items()
                  if _psi_id == psi_id][0]
    scale_diff = list(jtfs.scf.psi_ids.values()).index(psi_id)
    pad_diff = jtfs.scf.J_pad_frs_max_init - jtfs.scf.J_pad_frs[scale_diff]
    return jtfs.phi_f_fr[0][pad_diff][0]


def _handle_fig_ax(fig, ax, newfig):
    got_fig_or_ax = bool(fig or ax)
    if newfig:
        if got_fig_or_ax:  # no-cov
            raise ValueError("Can't have `newfig=True` if `fig` or `ax` are "
                             "passed.")
        fig, ax = plt.subplots(1, 1)
    else:
        ax  = ax  or plt.gca()
        fig = fig or plt.gcf()
    return fig, ax, got_fig_or_ax


def _default_to_fig_wh(fig_wh, w=1, h=1):
    """`fig_wh` is the library's chosen `figsize` for the specific visual at one
    point. Here it's rescaled such that it equals itself if the global configs
    equal their original value - meaning if they're e.g. double the value,
    we get same aspect ratio but the figure is twice as large.
    """
    DCFG = configs.get_defaults(library=True)

    w_fig, h_fig = fig_wh
    w_cfg, h_cfg = CFG['VIZ']['figsize']
    w_dft, h_dft = DCFG['VIZ']['figsize']
    w_user, h_user = w, h

    figsize = (w_fig * w_user * (w_cfg / w_dft) * _gscale(),
               h_fig * h_user * (h_cfg / h_dft) * _gscale())
    return figsize


def _gscale():
    """Short-hand for verbose parameter."""
    return CFG['VIZ']['global_scale']


def _gscale_r():
    """Rescaled `_gscale` for fontsizes, works better than proportionality."""
    return _gscale()**(1 / 1.7)


def _handle_global_scale(dc):
    if not isinstance(dc, dict):  # no-cov
        return
    for k, v in dc.items():
        if isinstance(v, dict):
            _handle_global_scale(v)
        elif k == 'figsize':
            dc[k] = tuple(_gscale() * np.array(v))
        elif k in ('fontsize', 'labelsize'):
            dc[k] = _gscale_r() * v


def _handle_auto_xlims(auto_xlims, x, y, logx=None):
    # TODO use `ax.autoscale(tight=True)` instead
    if auto_xlims is None:
        # don't change limits when calling like `plot([], vlines=...)`
        cond0 = bool(
            ((x is not None and len(x) != 0) or
             (y is not None and len(y) != 0))
        )
        if logx is None:
            auto_xlims = cond0
        else:
            auto_xlims = bool(cond0 and not logx)
    return auto_xlims


def _handle_tick_params(dc):
    if 'tick_params' not in dc:
        dc['tick_params'] = deepcopy(CFG['VIZ']['tick_params'])
    dc['tick_params']['labelsize'] *= _gscale_r()


def _handle_xy(x, y, squeeze=None):
    def get_N(h):
        if isinstance(h, list):
            N = len(h)
        elif h.ndim == 2:
            # if 2D, take first dim as time, per how `plt.plot` expects it (to
            # show `M` separate plots of length `N`, where `h.shape==(N, M)`)
            N = len(h)
        else:
            # works directly with 1D; if 2D and one of dimensions is 1, assume
            # it's such that one line is plotted (i.e. `(N, 1)`).
            N = h.size
        return N

    if x is None and y is None:  # no-cov
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        x = np.arange(get_N(y))
    elif y is None:
        y = x
        x = np.arange(get_N(x))
    if squeeze:
        x = x.squeeze() if not isinstance(x, list) else x
        y = y.squeeze() if not isinstance(y, list) else y
    return x, y


def _no_ticks_borders(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
