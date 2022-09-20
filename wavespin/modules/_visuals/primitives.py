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
from . import plt


def imshow(x, title=None, show=True, cmap=None, norm=None, abs=0,
           w=None, h=None, ticks=True, borders=True, aspect='auto',
           ax=None, fig=None, yticks=None, xticks=None, xlabel=None, ylabel=None,
           **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    ticks: False to not plot x & y ticks
    borders: False to not display plot borders
    w, h: rescale width & height
    kw: passed to `plt.imshow()`
    """
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if norm is None:
        mx = np.max(np.abs(x))
        vmin, vmax = ((-mx, mx) if not abs else
                      (0, mx))
    else:
        vmin, vmax = norm
    if cmap == 'none':
        cmap = None
    elif cmap is None:
        cmap = 'turbo' if abs else 'bwr'
    _kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect, **kw)

    if abs:
        ax.imshow(np.abs(x), **_kw)
    else:
        ax.imshow(x.real, **_kw)

    _handle_ticks(ticks, xticks, yticks, ax)

    if title is not None:
        _title(title, ax=ax)
    if w or h:
        fig.set_size_inches(12 * (w or 1), 12 * (h or 1))

    _scale_plot(fig, ax, show=False, w=None, h=None, xlabel=xlabel,
                ylabel=ylabel, auto_xlims=False)

    if not borders:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)

    if show:
        plt.show()


def plot(x, y=None, title=None, show=0, complex=0, abs=0, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None,
         xlabel=None, ylabel=None, xticks=None, yticks=None, ticks=True,
         ax=None, fig=None, squeeze=True, auto_xlims=None, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    complex: plot `x.real` & `x.imag`
    ticks: False to not plot x & y ticks
    w, h: rescale width & height
    kw: passed to `plt.imshow()`
    """
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        y = y if isinstance(y, list) or not squeeze else y.squeeze()
        x = np.arange(len(y))
    elif y is None:
        x = x if isinstance(x, list) or not squeeze else x.squeeze()
        y = x
        x = np.arange(len(x))
    x = x if isinstance(x, list) or not squeeze else x.squeeze()
    y = y if isinstance(y, list) or not squeeze else y.squeeze()

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

    _handle_ticks(ticks, xticks, yticks, ax)

    if title is not None:
        _title(title, ax=ax)
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)


def scat(x, y=None, title=None, show=0, s=18, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None,
         complex=False, abs=False, xlabel=None, ylabel=None, xticks=None,
         yticks=None, ticks=1, ax=None, fig=None, auto_xlims=None, **kw):
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        x = np.arange(len(y))
    elif y is None:
        y = x
        x = np.arange(len(x))

    def do_scatter(x, y):
        if complex:
            ax.scatter(x, y.real, s=s, **kw)
            ax.scatter(x, y.imag, s=s, **kw)
        else:
            if abs:
                y = np.abs(y)
            ax.scatter(x, y, s=s, **kw)

    x_2d = bool(hasattr(x, 'ndim') and x.ndim == 2)
    y_2d = bool(y is not None and hasattr(y, 'ndim') and y.ndim == 2)
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

    _handle_ticks(ticks, xticks, yticks, ax)

    if title is not None:
        _title(title, ax=ax)
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)


def plotscat(*args, **kw):
    show = kw.pop('show', False)
    plot(*args, **kw)
    scat(*args, **kw)
    if show:
        plt.show()


def hist(x, bins=500, title=None, show=0, stats=0, ax=None, fig=None,
         w=1, h=1, xlims=None, ylims=None, xlabel=None, ylabel=None):
    """Histogram. `stats=True` to print mean, std, min, max of `x`."""
    def _fmt(*nums):
        return [(("%.3e" % n) if (abs(n) > 1e3 or abs(n) < 1e-3) else
                 ("%.3f" % n)) for n in nums]

    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    x = np.asarray(x)
    _ = ax.hist(x.ravel(), bins=bins)
    _title(title, ax)
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel)
    if show:
        plt.show()

    if stats:
        mu, std, mn, mx = (x.mean(), x.std(), x.min(), x.max())
        print("(mean, std, min, max) = ({}, {}, {}, {})".format(
            *_fmt(mu, std, mn, mx)))
        return mu, std, mn, mx


def _vhlines(lines, kind='v', ax=None):
    lfn = getattr(plt if ax is None else ax, f'ax{kind}line')

    if not isinstance(lines, (list, tuple)):
        lines, lkw = [lines], {}
    elif isinstance(lines, (list, np.ndarray)):
        lkw = {}
    elif isinstance(lines, tuple):
        lines, lkw = lines
        lines = lines if isinstance(lines, (list, np.ndarray)) else [lines]
    else:
        raise ValueError("`lines` must be list or (list, dict) "
                         "(got %s)" % lines)
    if 'linewidth' not in lkw:
        lkw['linewidth'] = 1
    if 'color' not in lkw:
        lkw['color'] = 'tab:red'

    for line in lines:
        lfn(line, **lkw)


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
def _ticks(xticks, yticks, ax):
    def fmt(ticks):
        if all(isinstance(h, str) for h in ticks):
            return "%s"
        return ("%.d" if all(float(h).is_integer() for h in ticks) else
                "%.3g")

    if yticks is not None:
        if not hasattr(yticks, '__len__') and not yticks:
            ax.set_yticks([])
        else:
            if isinstance(yticks, tuple):
                yticks, ykw = yticks
            else:
                ykw = {}

            idxs = np.linspace(0, len(yticks) - 1, 8).astype('int32')
            yt = [fmt(yticks) % h for h in np.asarray(yticks)[idxs]]
            ax.set_yticks(idxs)
            ax.set_yticklabels(yt, **ykw)
    if xticks is not None:
        if not hasattr(xticks, '__len__') and not xticks:
            ax.set_xticks([])
        else:
            if isinstance(xticks, tuple):
                xticks, xkw = xticks
            else:
                xkw = {}
            idxs = np.linspace(0, len(xticks) - 1, 8).astype('int32')
            xt = [fmt(xticks) % h for h in np.asarray(xticks)[idxs]]
            ax.set_xticks(idxs)
            ax.set_xticklabels(xt, **xkw)


def _handle_ticks(ticks, xticks, yticks, ax):
    ticks = ticks if isinstance(ticks, (list, tuple)) else (ticks, ticks)
    if not ticks[0]:
        ax.set_xticks([])
    if not ticks[1]:
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks, ax)


def _title(title, ax=None):
    if title is None:
        return
    title, kw = (title if isinstance(title, tuple) else
                 (title, {}))
    defaults = dict(loc='left', fontsize=17, weight='bold')
    for k, v in defaults.items():
        kw[k] = kw.get(k, v)

    if ax:
        ax.set_title(str(title), **kw)
    else:
        plt.title(str(title), **kw)


def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None,
                xlims=None, ylims=None, xlabel=None, ylabel=None,
                auto_xlims=True):
    if xlims:
        ax.set_xlim(*xlims)
    elif auto_xlims:
        xmin, xmax = ax.get_xlim()
        rng = xmax - xmin
        ax.set_xlim(xmin + .02 * rng, xmax - .02 * rng)

    if ylims:
        ax.set_ylim(*ylims)
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if xlabel is not None:
        if isinstance(xlabel, tuple):
            xlabel, xkw = xlabel
        else:
            xkw = dict(weight='bold', fontsize=15)
        ax.set_xlabel(xlabel, **xkw)
    if ylabel is not None:
        if isinstance(ylabel, tuple):
            ylabel, ykw = ylabel
        else:
            ykw = dict(weight='bold', fontsize=15)
        ax.set_ylabel(ylabel, **ykw)
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
    c = c.swapaxes(0, 2).transpose(1, 0, 2)
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
    if os.path.isfile(savepath):
        if not overwrite:
            raise RuntimeError("File already exists at `savepath`; "
                               "set `overwrite=True` to overwrite.\n"
                               "%s" % str(savepath))
        else:
            # delete if exists
            os.unlink(savepath)


def _get_phi_for_psi_id(jtfs, psi_id):
    """Returns `phi_f_fr` at appropriate length, but always of scale `log2_F`."""
    scale_diff = list(jtfs.psi_ids.values()).index(psi_id)
    pad_diff = jtfs.J_pad_frs_max_init - jtfs.J_pad_frs[scale_diff]
    return jtfs.phi_f_fr[0][pad_diff][0]
