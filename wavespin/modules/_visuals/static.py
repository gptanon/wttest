# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Static (line/image, non-animated) visuals."""
import numpy as np
import warnings
from scipy.fft import ifft, ifftshift
from copy import deepcopy

from ...toolkit import (coeff_energy, coeff_distance, energy, make_eps,
                        pack_coeffs_jtfs)
from ...utils.gen_utils import fill_default_args
from .primitives import (
    plot, scat, imshow,
    _get_phi_for_psi_id, _get_compute_pairs, _format_ticks, _colorize_complex,
    _gscale, _gscale_r, _handle_global_scale, _default_to_fig_wh,
    _handle_tick_params,
)
from . import plt
from ... import CFG


def filterbank_scattering(sc, zoom=0, filterbank=True, lp_sum=False, lp_phi=True,
                          first_order=True, second_order=False, plot_kw=None):
    """Visualize temporal filterbank in frequency domain, 1D.

    Parameters
    ----------
    sc : Scattering1D / TimeFrequencyScattering1D
        Scattering instance.

    zoom : int
        Will zoom plots by this many octaves.
        If -1, will show full frequency axis (including negatives),
        and both spins.

    filterbank : bool (default True)
        Whether to plot the filterbank.

    lp_sum : bool (default False)
        Whether to plot Littlewood-Paley sum of the filterbank.

    lp_phi : bool (default True)
        Whether to include the lowpass filter in LP-sum visual.
        Has no effect if `lp_sum == False`.

    first_order : bool (default True)
        Whether to plot the first-order filters.

    second_order : bool (default False)
        Whether to plot the second-order filters.

    plot_kw : None / dict
        Will pass to `wavespin.visuals.plot(**plot_kw)`.

    Example
    -------
    Also see `examples/visuals_tour.py`.
    ::

        sc = Scattering1D(shape=2048, J=8, Q=8)
        filterbank_scattering(sc)
    """
    def _plot_filters(ps, p0, lp, J, title):
        # determine plot parameters ##########################################
        Nmax = len(ps[0][0])
        # x-axis zoom
        if 'xlims' in user_plot_kw_names:
            xlims = plot_kw['xlims']
        else:
            if zoom == -1:
                xlims = (-.02 * Nmax, 1.02 * Nmax)
            else:
                xlims = (-.01 * Nmax/ 2**zoom, .55 * Nmax / 2**zoom)

        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = title

        # plot filterbank ####################################################
        figsize = _default_to_fig_wh((9.5, 7))
        init_fig_kw = dict(figsize=figsize, dpi=CFG['VIZ']['dpi'])
        N = len(ps[0][0])

        if filterbank:
            fig, ax = plt.subplots(1, 1, **init_fig_kw)
            figax = dict(fig=fig, ax=ax)
            # Morlets
            for p in ps:
                j = p['j']
                _plot(p[0], color=colors[j], linestyle=linestyles[j], **figax)
            # vertical lines (octave bounds)
            vlines = vlines=([Nmax//2**j for j in range(1, J + 2)],
                             dict(color='k', linewidth=1))
            _plot([], vlines=vlines, **figax)
            # lowpass
            if isinstance(p0[0], list):
                p0 = p0[0]
            vlines = (Nmax//2, dict(color='k', linewidth=1))
            _plot([], vlines=vlines, **figax)

            _filterbank_plots_handle_global_scale(plot_kw)
            _plot(p0[0], color='k', **plot_kw, **figax)

            _filterbank_style_axes(ax, N, xlims)
            plt.show()

        # plot LP sum ########################################################
        if lp_sum:
            if 'title' not in user_plot_kw_names:
                plot_kw['title'] = "Littlewood-Paley sum"
            fig, ax = plt.subplots(1, 1, **init_fig_kw)
            _filterbank_plots_handle_global_scale(plot_kw)

            hlines = (2, dict(color='tab:red', linestyle='--'))
            vlines = (Nmax//2, dict(color='k', linewidth=1))
            _plot(lp, **plot_kw, fig=fig, ax=ax, hlines=hlines, vlines=vlines,
                  show=0)
            _filterbank_style_axes(ax, N, xlims, ymax=lp.max()*1.03)
            plt.show()

    # handle `plot_kw`
    if plot_kw is not None:
        # don't alter external dict
        plot_kw = deepcopy(plot_kw)
    else:
        plot_kw = {}
    user_plot_kw_names = list(plot_kw)
    _handle_tick_params(plot_kw)

    # define colors & linestyles
    colors = [f"tab:{c}" for c in ("blue orange green red purple brown pink "
                                   "gray olive cyan".split())]
    linestyles = ('-', '--', '-.')
    nc = len(colors)
    nls = len(linestyles)

    # support J up to nc * nls
    colors = colors * nls
    linestyles = [ls_set for ls in "- -- -.".split() for ls_set in [ls]*nc]

    # shorthand references
    p0 = sc.phi_f
    p1 = sc.psi1_f
    if second_order:
        p2 = sc.psi2_f

    # compute LP sum
    lp1, lp2 = 0, 0
    if lp_sum:
        # it's list for JTFS
        p0_longest = p0[0] if not isinstance(p0[0], list) else p0[0][0]
        for p in p1:
            lp1 += np.abs(p[0])**2
        if lp_phi:
            lp1 += np.abs(p0_longest)**2

        if second_order:
            for p in p2:
                lp2 += np.abs(p[0])**2
            if lp_phi:
                lp2 += np.abs(p0_longest)**2

    # title & plot
    (Q0, Q1), (J0, J1) = sc.Q, sc.J
    if first_order:
        title = "First-order filterbank | J, Q1, T = {}, {}, {}".format(
            J0, Q0, sc.T)
        _plot_filters(p1, p0, lp1, J0, title=title)

    if second_order:
        title = "Second-order filterbank | J, Q2, T = {}, {}, {}".format(
            J1, Q1, sc.T)
        _plot_filters(p2, p0, lp2, J1, title=title)


def filterbank_jtfs_1d(jtfs, zoom=0, psi_id=0, filterbank=True, lp_sum=False,
                       lp_phi=True, center_dc=None, both_spins=True,
                       plot_kw=None):
    """Visualize JTFS frequential filterbank in frequency domain, 1D.

    Parameters
    ----------
    jtfs : TimeFrequencyScattering1D
        JTFS instance.

    zoom : int
        Will zoom plots by this many octaves.
        If -1, will show full frequency axis (including negatives),
        and both spins.

    psi_id : int
        Indexes `jtfs.psi1_f_fr_up` & `_dn` - the ID of the filterbank
        (lower = tailored to larger input along frequency).

    filterbank : bool (default True)
        Whether to plot the filterbank.

    lp_sum : bool (default False)
        Whether to plot Littlewood-Paley sum of the filterbank.

    lp_phi : bool (default True)
        Whether to include the lowpass filter in LP-sum visual.
        Has no effect if `lp_sum == False`.

    center_dc : bool / None
        If True, will `ifftshift` to center the dc bin.
        Defaults to `True` if `zoom == -1`.

    both_spins : bool (default True)
        Whether to plot both up and down, or only up.

    plot_kw : None / dict
        Will pass to `plot(**plot_kw)`.

    Example
    -------
    Also see `examples/visuals_tour.py`.
    ::

        jtfs = TimeFrequencyScattering1D(shape=2048, J=8, Q=8)
        filterbank_jtfs_1d(jtfs)
    """
    def _handle_global_scale(plot_kw):
        if 'title' in plot_kw and not isinstance(plot_kw['title'], tuple):
            fscaled = CFG['VIZ']['title']['fontsize'] * _gscale_r()
            plot_kw['title'] = (plot_kw['title'], {'fontsize': fscaled})

    def _plot_filters(ps, p0, lp, fig0, ax0, fig1, ax1, title_base, up):
        # determine plot parameters ##########################################
        # vertical lines (octave bounds)
        Nmax = len(ps[psi_id][0])
        j_dists = np.array([Nmax//2**j for j in range(1, jtfs.J_fr + 1)])
        if up and not (up and zoom == -1 and center_dc and not both_spins):
            vlines = (Nmax//2 - j_dists if center_dc else
                      j_dists)
        else:
            vlines = (Nmax//2 + j_dists if center_dc else
                      Nmax - j_dists)
        # x-axis zoom
        if 'xlims' in user_plot_kw_names:
            xlims = plot_kw['xlims']
        else:
            if zoom == -1:
                xlims = (-.02 * Nmax, 1.02 * Nmax)
            else:
                xlims = (-.01 * Nmax / 2**zoom, .55 * Nmax / 2**zoom)
                if not up:
                   xlims = (Nmax - xlims[1], Nmax - .2 * xlims[0])

        # title
        if zoom != -1:
            title = title_base % "up" if up else title_base % "down"
        else:
            title = title_base

        # handle `plot_kw`
        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = title

        # plot filterbank ####################################################
        N = len(ps[psi_id][0])

        if filterbank:
            # bandpasses
            for n1_fr, p in enumerate(ps[psi_id]):
                j = ps['j'][psi_id][n1_fr]
                pplot = p.squeeze()
                if center_dc:
                    pplot = ifftshift(pplot)
                _plot(pplot, color=colors[j], linestyle=linestyles[j], ax=ax0)
            # lowpass
            p0plot = _get_phi_for_psi_id(jtfs, psi_id)
            if center_dc:
                p0plot = ifftshift(p0plot)

            # plot & style
            _filterbank_plots_handle_global_scale(plot_kw)
            _plot(p0plot, color='k', **plot_kw, ax=ax0, fig=fig0,
                  vlines=(vlines, dict(color='k', linewidth=1)))

            _filterbank_style_axes(ax0, N, xlims, zoom=zoom, is_jtfs=True)
        else:
            plt.close(fig0)

        # plot LP sum ########################################################
        plot_kw_lp = {}
        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = ("Littlewood-Paley sum" +
                                " (no phi)" * int(not lp_phi))
        if 'ylims' not in user_plot_kw_names:
            plot_kw_lp['ylims'] = (0, None)
        _filterbank_plots_handle_global_scale(plot_kw)

        if lp_sum and not (zoom == -1 and not up):
            lpplot = ifftshift(lp) if center_dc else lp
            hlines = (1, dict(color='tab:red', linestyle='--'))
            vlines = (Nmax//2, dict(color='k', linewidth=1))

            _plot(lpplot, **plot_kw, **plot_kw_lp, ax=ax1, fig=fig1,
                  hlines=hlines, vlines=vlines)
            _filterbank_style_axes(ax1, N, xlims, ymax=lp.max()*1.03,
                                   zoom=zoom, is_jtfs=True)

    # handle `plot_kw`
    if plot_kw is not None:
        # don't alter external dict
        plot_kw = deepcopy(plot_kw)
    else:
        plot_kw = {}
    user_plot_kw_names = list(plot_kw)
    _handle_tick_params(plot_kw)

    # handle `center_dc`
    if center_dc is None:
        center_dc = bool(zoom == -1)

    # define colors & linestyles
    colors = [f"tab:{c}" for c in ("blue orange green red purple brown pink "
                                   "gray olive cyan".split())]
    linestyles = ('-', '--', '-.')
    nc = len(colors)
    nls = len(linestyles)

    # support J up to nc * nls
    colors = colors * nls
    linestyles = [ls_set for ls in "- -- -.".split() for ls_set in [ls]*nc]

    # shorthand references
    p0 = jtfs.scf.phi_f_fr
    pup = jtfs.psi1_f_fr_up
    pdn = jtfs.psi1_f_fr_dn

    # compute LP sum
    lp = 0
    if lp_sum:
        psi_fs = (pup, pdn) if both_spins else (pup,)
        for psi1_f in psi_fs:
            for p in psi1_f[psi_id]:
                lp += np.abs(p)**2
        if lp_phi:
            p0 = _get_phi_for_psi_id(jtfs, psi_id)
            lp += np.abs(p0)**2

    # title
    params = (jtfs.J_fr, jtfs.Q_fr, jtfs.F)
    if zoom != -1:
        title_base = ("Freq. filterbank | spin %s | J_fr, Q_fr, F = {}, {}, {}"
                      ).format(*params)
    else:
        title_base = ("Freq. filterbank | J_fr, Q_fr, F = {}, {}, {}"
                      ).format(*params)

    # plot ###################################################################
    def make_figs(init_fig_kw):
        fn = lambda: plt.subplots(1, 1, **init_fig_kw)
        return ([fn() for _ in range(2)] if lp_sum else
                (fn(), (None, None)))

    figsize = _default_to_fig_wh((9.5, 7))
    init_fig_kw = dict(figsize=figsize, dpi=CFG['VIZ']['dpi'])

    (fig0, ax0), (fig1, ax1) = make_figs(init_fig_kw)
    _plot_filters(pup, p0, lp, fig0, ax0, fig1, ax1, title_base=title_base,
                  up=True)
    if zoom != -1:
        plt.show()
        if both_spins:
            (fig0, ax0), (fig1, ax1) = make_figs(init_fig_kw)

    if both_spins:
        _plot_filters(pdn, p0, lp, fig0, ax0, fig1, ax1, title_base=title_base,
                      up=False)
    plt.show()


def filterbank_heatmap(sc, first_order=None, second_order=False,
                       frequential=None, parts='all', psi_id=0, **plot_kw):
    """Visualize sc filterbank as heatmap of all bandpasses.

    Parameters
    ----------
    sc : Scattering1D / TimeFrequencyScattering1D
        Scattering instance.

    first_order : bool / None
        Whether to show first-order filterbank. Defaults to `True` if
        `sc` is non-JTFS.

    second_order : bool (default False)
        Whether to show second-order filterbank.

    frequential : bool / tuple[bool]
        Whether to show frequential filterbank (requires JTFS `sc`).
        Tuple specifies `(up, down)` spins separately. Defaults to `(False, True)`
        if `sc` is JTFS and `first_order` is `False` or `None` and
        `second_order == False`. If bool, becomes `(False, frequential)`.

    parts : str / tuple / list
        One of: 'abs', 'real', 'imag', 'freq'. First three refer to time-domain,
        'freq' is abs of frequency domain.

    psi_id : int
        Indexes `jtfs.psi1_f_fr_up` & `_dn` - the ID of the filterbank
        (lower = tailored to larger input along frequency).

    plot_kw : None / dict
        Will pass to `wavespin.visuals.imshow(**plot_kw)`.

    Example
    -------
    Also see `examples/visuals_tour.py`.
    ::

        sc = Scattering1D(shape=2048, J=10, Q=16)
        filterbank_heatmap(sc, first_order=True, second_order=True)
    """
    def to_time_and_viz(psi_fs, name, get_psi):
        # move wavelets to time domain
        psi_fs = [get_psi(p) for p in psi_fs]
        psi_fs = [p for p in psi_fs if p is not None]
        psi1s = [ifftshift(ifft(p)) for p in psi_fs]
        psi1s = np.array([p / np.abs(p).max() for p in psi1s])

        # handle kwargs & 'global_scale'
        pkw = deepcopy(plot_kw)
        user_kw = list(plot_kw)
        if 'xlabel' not in user_kw:
            pkw['xlabel'] = 'time [samples]'
        if 'ylabel' not in user_kw:
            pkw['ylabel'] = 'wavelet index'
        if 'interpolation' not in user_kw and len(psi1s) < 30:
            pkw['interpolation'] = 'none'
        _handle_tick_params(pkw)

        # wrap `imshow` to handle text size args and handle `global_scale`
        fontsizes = {name: {'fontsize':
                            CFG['VIZ'][name]['fontsize'] * _gscale_r()}
                     for name in ('title', 'xlabel', 'ylabel')}
        fontfamily = {'fontfamily': CFG['VIZ']['long_title_fontfamily'][1]}

        def timshow(*args, **kwargs):
            for name in ('title', 'xlabel', 'ylabel'):
                if name in kwargs:
                    tkw = {**fontfamily, **fontsizes[name]}
                    kwargs[name] = (kwargs[name], tkw)

            figsize = _default_to_fig_wh((10.5, 7))
            init_fig_kw = dict(figsize=figsize, dpi=CFG['VIZ']['dpi'])
            fig, ax = plt.subplots(1, 1, **init_fig_kw)

            _imshow(*args, **kwargs, fig=fig, ax=ax)

        # do plotting
        if 'abs' in parts:
            apsi1s = np.abs(psi1s)
            timshow(apsi1s, abs=1, **pkw,
                    title=f"{name} filterbank | modulus | ampl.-equalized")
        if 'real' in parts:
            timshow(psi1s.real, **pkw,
                    title=f"{name} filterbank | real part | ampl.-equalized")
        if 'imag' in parts:
            timshow(psi1s.imag, **pkw,
                    title=f"{name} filterbank | imag part | ampl.-equalized")
        if 'freq' in parts:
            if 'xlabel' not in user_kw:
                pkw['xlabel'] = 'frequencies [samples] | dc, +, -'
            psi_fs = np.array(psi_fs)
            timshow(psi_fs, abs=1, **pkw,
                    title=f"{name} filterbank | freq-domain")

    # process `parts`
    supported = ('abs', 'real', 'imag', 'freq')
    if parts == 'all':
        parts = supported
    else:
        for p in parts:
            if p not in supported:
                raise ValueError(("unsupported `parts` '{}'; must be one of: {}"
                                  ).format(p, ', '.join(parts)))

    # process visuals selection
    is_jtfs = bool(hasattr(sc, 'scf'))
    if first_order is None:
        first_order = not is_jtfs
    if frequential is None:
        # default to frequential only if is jtfs and first_order wasn't requested
        frequential = (False, is_jtfs and not (first_order or second_order))
    elif isinstance(frequential, (bool, int)):
        frequential = (False, bool(frequential))
    if all(not f for f in frequential):
        frequential = False
    if frequential and not is_jtfs:
        raise ValueError("`frequential` requires JTFS `sc`.")
    if not any(arg for arg in (first_order, second_order, frequential)):
        raise ValueError("Nothing to visualize! (got False for all of "
                         "`first_order`, `second_order`, `frequential`)")

    # visualize
    if first_order or second_order:
        get_psi = lambda p: (p[0] if not hasattr(p[0], 'cpu') else
                             p[0].cpu().numpy())
        if first_order:
            to_time_and_viz(sc.psi1_f, '1st-order', get_psi)
        if second_order:
            to_time_and_viz(sc.psi2_f, '2nd-order', get_psi)
    if frequential:
        get_psi = lambda p: ((p if not hasattr(p, 'cpu') else
                              p.cpu().numpy()).squeeze())
        if frequential[0]:
            to_time_and_viz(sc.psi1_f_fr_up[psi_id], 'Freq. up',
                            get_psi)
        if frequential[1]:
            to_time_and_viz(sc.psi1_f_fr_dn[psi_id], 'Freq. down',
                            get_psi)


def viz_jtfs_2d(jtfs, Scx=None, viz_filterbank=True, viz_coeffs=None,
                viz_spins=(True, True), axis_labels=True, fs=None, psi_id=0,
                w=1., h=1., show=True, savename=None, plot_cfg=None):
    """Visualize JTFS filterbank and/or coefficients in their true 4D structure,
    via 2D time-frequency slices laid out in a 2D time-(log-quefrency) grid.

    Method accounts for `paths_exclude`.

    Parameters
    ----------
    jtfs : TimeFrequencyScattering1D
        JTFS instance.
        Requires `jtfs.out_type` that's 'dict:array' or 'dict:list'.

    Scx : None / dict / np.ndarray
        Coefficients to visualize. Requires:

            - `jtfs.out_type` to be`'dict:list'` or `'dict:array'`. Or,
            - `Scx` to be a 4D numpy array packed with `pack_coeffs_jtfs` and
              `structure=2` (which is what it will otherwise do internally).

    viz_filterbank : bool (default True)
        Whether to visualize the filterbank.

        Note, each 2D wavelet's plot is color-normed to the wavelet's maxima
        (otherwise most wavelets vanish).

    viz_coeffs : bool / None
        Whether to visualize the coefficients (requires `Scx`).

        The coefficients and filters are placed in same slots in the 2D grid,
        so if both are visualized, we see which wavelet produced which
        coefficient. An exception is `sampling_psi_fr='recalibrate'`, as the
        visual supports only one `psi_id`, while `'recalibrate'` varies it
        with `xi2`.

        Defaults to True if `Scx` is not None.

    viz_spins : tuple[bool]
        `viz_spin_up, viz_spin_dn = viz_spins` -- can use to visualize only
        one of the two spinned pairs.

    axis_labels : bool (default True)
        If True, will label plot with title, axis labels, and units.

    fs : None / int
        Sampling rate. If provided, will display physical units (Hz), else
        discrete (cycles/sample).

    savename : str / None
        If str, will save as `savename + '0.png'` and `savename + '1.png'`,
        for filterbank and coeffs respectively.

    psi_id : int
        Indexes `jtfs.psi1_f_fr_up` & `_dn` - the ID of the filterbank
        (lower = tailored to larger input along frequency).

    w, h : int, int
        Scale plot width and height, respectively.

    show : bool (default True)
        Whether to display generated plots. Else, will `plt.close(fig)`
        (after saving, if applicable).

    plot_cfg : None / dict
        Configures plotting. Will fill for missing values from defaults
        (see `plot_cfg_defaults` in source code). Will not warn if an argument
        is unused (e.g. per `viz_coeffs=False`). Supported key-values:

            'phi_t_blank' : bool
              If True, draws `phi_t * psi_f` pairs only once (since up == down).
              Can't be `True` with `phi_t_loc='both'`.

            'phi_t_loc' : str['top', 'bottom', 'both']
              'top' places `phi_t * psi_f` pairs alongside "up" spin,
              'bottom' alongside "down", and 'both' places them in both spots.
              Additionally, 'top' and 'bottom' will scale coefficients by
              `sqrt(2)` for energy norm (since they're shown in half of all
              places).

            'filter_part' : str['real', 'imag', 'complex', 'abs']
              Part of each filter to plot.

            'filter_label' : bool (default False)
              Whether to label each filter plot with its index/meta info.

            'filter_label_kw' : dict / None
              Passed to `ax.annotate` for filterbank visuals.

            'label_kw_xy' : dict
                Passed to all `ax.set_xlabel` and `ax.set_ylabel`.

            'title_kw' : dict
                Passed to all `fig.suptitle`.

            'suplabel_kw_x' : dict
                Passed to all `fig.supxlabel`.

            'suplabel_kw_y' : dict
                Passed to all `fig.supylabel`.

            'imshow_kw_filterbank' : dict
                Passed to all `ax.imshow` for filterbank visuals.

            'imshow_kw_coeffs' : dict
                Passed to all `ax.imshow` for coefficient visuals.

            'subplots_adjust_kw' : dict
                Passed to all `fig.subplots_adjust`.

            'savefig_kw': dict
                Passed to all `fig.savefig`.

            'filterbank_zoom': float / int
                Zoom factor for filterbank visual.
                  - >1: zoom in
                  - <1: zoom out.
                  - -1: zoom every wavelet to its own support. With 'resample',
                        all wavelets within the same pair should look the same,
                        per wavelet self-similarity.

            'coeff_color_max_mult' : float
                Scales plot color norm via
                    `ax.imshow(, vmin=0, vmax=coeff_color_max_mult * Scx.max())`
                `<1` will pronounce lower-valued coefficients and clip the rest.

            'coeff_color_max' : None / float
                Scales directly. Overrides 'coeff_color_max_mult'.

    Note: `xi1_fr` units
    --------------------
    Meta stores discrete, [cycles/sample].
    Want [cycles/octave].
    To get physical, we do `xi * fs`, where `fs [samples/second]`.
    Hence, find `fs` equivalent for `octaves`.

    If `Q1` denotes "number of first-order wavelets per octave", we
    realize that "samples" of `psi_fr` are actually "first-order wavelets":
        `xi [cycles/(first-order wavelets)]`
    Hence, set
        `fs [(first-order wavelets)/octave]`
    and so
        `xi1_fr = xi*fs = xi*Q1 [cycles/octave]`

     - This is consistent with raising `Q1` being equivalent of raising
       the physical sampling rate (i.e. sample `n1` more densely without
       changing the number of octaves).
     - Raising `J1` is then equivalent to increasing physical duration
       (seconds) without changing sampling rate, so `xi1_fr` is only a
       function of `Q1`.
    """
    # handle args ############################################################
    # `jtfs`, `Scx` sanity checks; set `viz_coeffs`
    if 'dict:' not in jtfs.out_type:
        raise ValueError("`jtfs.out_type` must be 'dict:array' or 'dict:list' "
                         "(got %s)" % str(jtfs.out_type))
    if Scx is not None:
        if not isinstance(Scx, dict):
            assert isinstance(Scx, np.ndarray), type(Scx)
            assert Scx.ndim == 4, Scx.shape
        else:
            assert isinstance(Scx, dict), type(Scx)
        if viz_coeffs is None:
            viz_coeffs = True
        elif not viz_coeffs:
            warnings.warn("Passed `Scx` and `viz_coeffs=False`; won't visualize!")
    elif viz_coeffs:
        raise ValueError("`viz_coeffs=True` requires passing `Scx`.")
    # `viz_coeffs`, `viz_filterbank` sanity check
    if not viz_coeffs and not viz_filterbank:
        raise ValueError("Nothing to visualize! (viz_coeffs and viz_filterbank "
                         "aren't True")
    # `psi_id` sanity check
    psi_ids_max = max(jtfs.psi_ids.values())
    if psi_id > psi_ids_max:
        raise ValueError("`psi_id` exceeds max existing value ({} > {})".format(
            psi_id, psi_ids_max))
    elif psi_id > 0 and jtfs.sampling_psi_fr == 'exclude':
        raise ValueError("`psi_id > 0` with `sampling_psi_fr = 'exclude'` "
                         "is not supported; to see which filters are excluded, "
                         "check which coefficients are zero.")

    # `plot_cfg`, defaults
    plot_cfg_defaults = {
        'phi_t_blank': None,
        'phi_t_loc': 'bottom',

        'filter_part': 'real',
        'filter_label': False,
        'filter_label_kw': dict(weight='bold', fontsize=26, xy=(.05, .82),
                                xycoords='axes fraction'),

        'label_kw_xy':   dict(fontsize=20),
        'title_kw':      dict(weight='bold', fontsize=26, y=1.025),
        'suplabel_kw_x': dict(weight='bold', fontsize=24, y=-.055),
        'suplabel_kw_y': dict(weight='bold', fontsize=24, x=-.075),
        'imshow_kw_filterbank': dict(aspect='auto', cmap='bwr'),
        'imshow_kw_coeffs':     dict(aspect='auto', cmap='turbo'),
        'subplots_adjust_kw': dict(left=0, right=1, bottom=0, top=1,
                                   wspace=.02, hspace=.02),
        'savefig_kw': dict(bbox_inches='tight'),

        'filterbank_zoom': .9,
        'coeff_color_max_mult': .8,
        'coeff_color_max': None,
    }
    C = fill_default_args(plot_cfg, plot_cfg_defaults, copy_original=True,
                          check_against_defaults=True)
    _handle_global_scale(C)

    # viz_spin; phi_t_loc; phi_t_blank
    viz_spin_up, viz_spin_dn = viz_spins

    assert C['phi_t_loc'] in ('top', 'bottom', 'both')
    if C['phi_t_loc'] == 'both':
        if C['phi_t_blank']:
            warnings.warn("`phi_t_blank` does nothing if `phi_t_loc='both'`")
            C['phi_t_blank'] = 0
    elif C['phi_t_blank'] is None:
        C['phi_t_blank'] = 1

    # fs
    if fs is not None:
        f_units = "[Hz]"
    else:
        f_units = "[cycles/sample]"

    # pack `Scx`, get meta ###################################################
    jmeta = jtfs.meta()
    if Scx is not None:
        if isinstance(Scx, dict):
            Scx = pack_coeffs_jtfs(Scx, jmeta, structure=2, out_3D=jtfs.out_3D,
                                   sampling_psi_fr=jtfs.sampling_psi_fr,
                                   reverse_n1=False)
            # reverse psi_t ordering
            Scx = Scx[::-1]

    # unpack filters and relevant meta #######################################
    n2s    = np.unique(jmeta['n']['psi_t * psi_f_up'][..., 0])
    n1_frs = np.unique(jmeta['n']['psi_t * psi_f_up'][..., 1])
    n_n2s, n_n1_frs = len(n2s), len(n1_frs)

    psi2s = [p for n2, p in enumerate(jtfs.psi2_f) if n2 in n2s]
    psis_up, psis_dn = [[p for n1_fr, p in enumerate(psi1_f_fr[psi_id])
                         if n1_fr in n1_frs]
                        for psi1_f_fr in (jtfs.psi1_f_fr_up, jtfs.psi1_f_fr_dn)]
    # must time-reverse to plot, so that
    #     low plot idx <=> high wavelet idx,    i.e.
    #                  <=> high spatial sample, i.e.
    #                  <=> high log-frequency
    # Up is time-reversed down, so just swap
    psis_dn, psis_up = psis_up, psis_dn
    pdn_meta = {field: [value for n1_fr, value in
                        enumerate(jtfs.psi1_f_fr_dn[field][psi_id])
                        if n1_fr in n1_frs]
                for field in jtfs.psi1_f_fr_dn if isinstance(field, str)}

    # Visualize ################################################################
    # helpers ###################################
    def show_filter(pt, pf, row_idx, col_idx, label_axis_fn=None,
                    n2_idx=None, n1_fr_idx=None, mx=None, up=None, skip=False):
        # style first so we can exit early if needed
        ax0 = axes0[row_idx, col_idx]
        no_border(ax0)
        if axis_labels and label_axis_fn is not None:
            label_axis_fn(ax0)

        if skip:
            return

        # trim to zoom on wavelet
        if zoom_each:
            if n2_idx == -1:
                stz = jtfs.phi_f['width'][0] * 8 // 2
            else:
                stz = psi2s[len(psi2s) - n2_idx - 1]['width'][0] * 8 // 2
            if n1_fr_idx == -1:
                scale_diff = list(jtfs.psi_ids.values()).index(psi_id)
                pad_diff = jtfs.J_pad_frs_max_init - jtfs.J_pad_frs[scale_diff]
                sfz = jtfs.phi_f_fr['width'][0][pad_diff][0] * 8 // 2
            else:
                widths = (jtfs.psi1_f_fr_up if up else
                          jtfs.psi1_f_fr_dn)['width'][psi_id]
                ix = (n1_frs if up else n1_frs[::-1])[n1_fr_idx]
                sfz = widths[ix] * 8 // 2
            stz = min(stz, ct)
            sfz = min(sfz, cf)
            pt = pt[ct - stz:ct + stz + 1]
            pf = pf[cf - sfz:cf + sfz + 1]
        else:
            pt = pt[ct - st:ct + st + 1]
            pf = pf[cf - sf:cf + sf + 1]

        Psi = pf[:, None] * pt[None]
        if mx is None:
            mx = np.abs(Psi).max()
        mn = -mx
        if C['filter_part'] == 'real':
            Psi = Psi.real
        elif C['filter_part'] == 'imag':
            Psi = Psi.imag
        elif C['filter_part'] == 'complex':
            Psi = _colorize_complex(Psi)
        elif C['filter_part'] == 'abs':
            Psi = np.abs(Psi)
            mn = 0

        ax0.imshow(Psi, vmin=mn, vmax=mx, **C['imshow_kw_filterbank'])
        if C['filter_label']:
            psi_txt = get_filter_label(n2_idx, n1_fr_idx, up)
            ax0.annotate(psi_txt, **C['filter_label_kw'])

    def get_filter_label(n2_idx, n1_fr_idx, up=None):
        if n2_idx != -1:
            n_t_psi = int(n2s[n_n2s - n2_idx - 1])
        if n1_fr_idx != -1:
            n_f_psi = int(n1_frs[n1_fr_idx] if up else
                          n1_frs[n_n1_frs - n1_fr_idx - 1])

        if n2_idx == -1 and n1_fr_idx == -1:
            info = ("\infty", "\infty", 0)
        elif n2_idx == -1:
            info = ("\infty", n_f_psi, 0)
        elif n1_fr_idx == -1:
            info = (n_t_psi, "\infty", 0)
        else:
            info = (n_t_psi, n_f_psi, '+1' if up else '-1')

        psi_txt = r"$\Psi_{%s, %s, %s}$" % info
        return psi_txt

    def no_border(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)

    def to_time(p_f):
        while isinstance(p_f, (dict, list)):
            p_f = p_f[0]
        return ifftshift(ifft(p_f.squeeze()))

    # generate canvas ###########################
    if viz_spin_up and viz_spin_dn:
        n_rows = 2*n_n1_frs + 1
    else:
        n_rows = n_n1_frs + 1
    n_cols = n_n2s + 1

    width  = 11 * w * _gscale()
    height = 11 * n_rows / n_cols * h * _gscale()

    skw = dict(figsize=(width, height), dpi=CFG['VIZ']['dpi'])
    if viz_filterbank:
        fig0, axes0 = plt.subplots(n_rows, n_cols, **skw)
    if viz_coeffs:
        fig1, axes1 = plt.subplots(n_rows, n_cols, **skw)

    # compute common params to zoom on wavelets based on largest wavelet
    # centers
    n1_fr_largest = n_n1_frs - 1
    n2_largest = n_n2s - 1
    pf_f = psis_dn[n1_fr_largest].squeeze()
    pt_f = psi2s[n2_largest][0].squeeze()
    ct = len(pt_f) // 2
    cf = len(pf_f) // 2

    zoom_each = bool(C['filterbank_zoom'] == -1)
    if not zoom_each:
        # supports; use widths instead to favor main lobe measure for visuals
        # `8` is a visually desired width->support conversion factor
        # `1/2` is base zoom since 'width'*8 is ~total support, so halving
        # allows using it like `psi[center-st:center+st]`.
        # `min` to not allow excess zoom out that indexes outside the array
        global_zoom = (1 / 2) / C['filterbank_zoom']
        st = min(ct, int(psi2s[n2_largest]['width'][0]    * 8 * global_zoom))
        sf = min(cf, int(pdn_meta['width'][n1_fr_largest] * 8 * global_zoom))

    # coeff max
    if Scx is not None:
        cmx = (C['coeff_color_max'] if C['coeff_color_max'] is not None else
               Scx.max() * C['coeff_color_max_mult'])

    # plot pairs ################################
    def plot_spinned(up):
        def label_axis(ax, n1_fr_idx, n2_idx):
            at_border = bool(n1_fr_idx == len(psi1_frs) - 1)
            if at_border:
                xi2 = psi2s[::-1][n2_idx]['xi']
                if fs is not None:
                    xi2 = xi2 * fs
                xi2 = _format_ticks(xi2)
                ax.set_xlabel(xi2, **C['label_kw_xy'])

        if up:
            psi1_frs = psis_up
        else:
            psi1_frs = psis_dn[::-1]

        for n2_idx, pt_f in enumerate(psi2s[::-1]):
            for n1_fr_idx, pf_f in enumerate(psi1_frs):
                # compute axis & coef indices ################################
                if up:
                    row_idx = n1_fr_idx
                    coef_n1_fr_idx = n1_fr_idx
                else:
                    if viz_spin_up:
                        row_idx = n1_fr_idx + 1 + n_n1_frs
                    else:
                        row_idx = n1_fr_idx + 1
                    coef_n1_fr_idx = n1_fr_idx + n_n1_frs + 1
                col_idx = n2_idx + 1
                coef_n2_idx = n2_idx + 1

                # visualize ##################################################
                # filterbank
                if viz_filterbank:
                    pt = to_time(pt_f)
                    pf = to_time(pf_f)
                    # if both spins, viz only on down
                    if (((viz_spin_up and viz_spin_dn) and not up) or
                        not (viz_spin_up and viz_spin_dn)):
                        label_axis_fn = lambda ax0: label_axis(ax0, n1_fr_idx,
                                                               n2_idx)
                    else:
                        label_axis_fn = None
                    show_filter(pt, pf, row_idx, col_idx, label_axis_fn,
                                n2_idx, n1_fr_idx, up=up)

                # coeffs
                if viz_coeffs:
                    c = Scx[coef_n2_idx, coef_n1_fr_idx]

                    ax1 = axes1[row_idx, col_idx]
                    ax1.imshow(c, vmin=0, vmax=cmx,
                               **C['imshow_kw_coeffs'])

                    # axis styling
                    no_border(ax1)
                    if axis_labels:
                        label_axis(ax1, n1_fr_idx, n2_idx)

    if viz_spin_up:
        plot_spinned(up=True)
    if viz_spin_dn:
        plot_spinned(up=False)

    # psi_t * phi_f ##########################################################
    if viz_filterbank:
        phif = to_time(_get_phi_for_psi_id(jtfs, psi_id))

    if viz_spin_up:
        row_idx = n_n1_frs
    else:
        row_idx = 0
    coef_n1_fr_idx = n_n1_frs

    for n2_idx, pt_f in enumerate(psi2s[::-1]):
        # compute axis & coef indices
        col_idx = n2_idx + 1
        coef_n2_idx = n2_idx + 1

        # filterbank
        if viz_filterbank:
            pt = to_time(pt_f)
            show_filter(pt, phif, row_idx, col_idx, None, n2_idx, n1_fr_idx=-1)

        # coeffs
        if viz_coeffs:
            ax1 = axes1[row_idx, col_idx]
            c = Scx[coef_n2_idx, coef_n1_fr_idx]
            ax1.imshow(c, vmin=0, vmax=cmx, **C['imshow_kw_coeffs'])
            no_border(ax1)

    # phi_t * psi_f ##########################################################
    def plot_phi_t(up):
        def label_axis(ax, n1_fr_idx):
            if up:
                filter_n1_fr_idx = n1_fr_idx
            else:
                filter_n1_fr_idx = n_n1_frs - n1_fr_idx - 1

            xi1_fr = pdn_meta['xi'][filter_n1_fr_idx] * jtfs.Q[0]
            if not up:
                xi1_fr = -xi1_fr
            xi1_fr = _format_ticks(xi1_fr)
            ax.set_ylabel(xi1_fr, **C['label_kw_xy'])

            at_border = bool(n1_fr_idx == len(psi1_frs) - 1)
            if at_border and axis_labels:
                ax.set_xlabel("0", **C['label_kw_xy'])

        if C['phi_t_loc'] == 'top' or (C['phi_t_loc'] == 'both' and up):
            if up:
                psi1_frs = psis_up
                assert not viz_spin_dn or (viz_spin_up and viz_spin_dn)
            else:
                if viz_spin_up and viz_spin_dn:
                    # don't show stuff if both spins given
                    psi1_frs = [p*0 for p in psis_up]
                else:
                    psi1_frs = psis_dn[::-1]
        elif C['phi_t_loc'] == 'bottom' or (C['phi_t_loc'] == 'both' and not up):
            if up:
                if viz_spin_up and viz_spin_dn:
                    # don't show stuff if both spins given
                    psi1_frs = [p*0 for p in psis_up]
                else:
                    psi1_frs = psis_up
            else:
                psi1_frs = psis_dn[::-1]
                assert not viz_spin_up or (viz_spin_up and viz_spin_dn)

        col_idx = 0
        coef_n2_idx = 0
        for n1_fr_idx, pf_f in enumerate(psi1_frs):
            if up:
                row_idx = n1_fr_idx
                coef_n1_fr_idx = n1_fr_idx
            else:
                if viz_spin_up and viz_spin_dn:
                    row_idx = n1_fr_idx + 1 + n_n1_frs
                else:
                    row_idx = n1_fr_idx + 1
                coef_n1_fr_idx = n1_fr_idx + 1 + n_n1_frs

            if viz_filterbank:
                pf = to_time(pf_f)

                # determine color `mx` and whether to skip
                skip = False
                if C['phi_t_loc'] != 'both':
                    # energy norm (no effect if color norm adjusted to Psi)
                    pf *= np.sqrt(2)

                if C['phi_t_loc'] == 'top':
                    if not up and (viz_spin_up and viz_spin_dn):
                        # actually zero but that defaults the plot to max negative
                        skip = True
                elif C['phi_t_loc'] == 'bottom':
                    if up and (viz_spin_up and viz_spin_dn):
                        # actually zero but that defaults the plot to max negative
                        skip = True
                elif C['phi_t_loc'] == 'both':
                    pass

                # show
                label_axis_fn = lambda ax0: label_axis(ax0, n1_fr_idx)
                show_filter(phit, pf, row_idx, col_idx, label_axis_fn,
                            n2_idx=-1, n1_fr_idx=n1_fr_idx, skip=skip, up=up)

            if viz_coeffs:
                ax1 = axes1[row_idx, col_idx]
                skip_coef = bool(
                    C['phi_t_blank'] and ((C['phi_t_loc'] == 'top' and not up) or
                                          (C['phi_t_loc'] == 'bottom' and up)))

                if not skip_coef:
                    c = Scx[coef_n2_idx, coef_n1_fr_idx]
                    if C['phi_t_loc'] != 'both':
                        # energy norm since we viz only once;
                        # did /= sqrt(2) in pack_coeffs_jtfs
                        c = c * np.sqrt(2)
                    if C['phi_t_loc'] == 'top':
                        if not up and (viz_spin_up and viz_spin_dn):
                            c = c * 0  # viz only once
                    elif C['phi_t_loc'] == 'bottom':
                        if up and (viz_spin_up and viz_spin_dn):
                            c = c * 0  # viz only once
                    ax1.imshow(c, vmin=0, vmax=cmx,
                               **C['imshow_kw_coeffs'])

                # axis styling
                no_border(ax1)
                if axis_labels:
                    label_axis(ax1, n1_fr_idx)

    if viz_filterbank:
        phit = to_time(jtfs.phi_f)

    if viz_spin_up:
        plot_phi_t(up=True)
    if viz_spin_dn:
        plot_phi_t(up=False)

    # phi_t * phi_f ##############################################################
    def label_axis(ax):
        ax.set_ylabel("0", **C['label_kw_xy'])

    if viz_spin_up:
        row_idx = n_n1_frs
    else:
        row_idx = 0
    col_idx = 0
    coef_n2_idx = 0
    coef_n1_fr_idx = n_n1_frs

    # filterbank
    if viz_filterbank:
        label_axis_fn = label_axis
        show_filter(phit, phif, row_idx, col_idx, label_axis_fn,
                    n2_idx=-1, n1_fr_idx=-1)

    # coeffs
    if viz_coeffs:
        c = Scx[coef_n2_idx, coef_n1_fr_idx]
        ax1 = axes1[row_idx, col_idx]
        ax1.imshow(c, vmin=0, vmax=cmx, **C['imshow_kw_coeffs'])

        # axis styling
        no_border(ax1)
        if axis_labels:
            label_axis(ax1)

    # finalize ###############################################################
    def fig_adjust(fig):
        if axis_labels:
            fig.supxlabel(f"Temporal modulation {f_units}",
                          **C['suplabel_kw_x'])
            fig.supylabel("Freqential modulation [cycles/octave]",
                          **C['suplabel_kw_y'])
        fig.subplots_adjust(**C['subplots_adjust_kw'])

    if viz_filterbank:
        fig_adjust(fig0)
        if axis_labels:
            if C['filter_part'] in ('real', 'imag'):
                info_txt = "%s part" % C['filter_part']
            elif C['filter_part'] == 'complex':
                info_txt = "complex"
            elif C['filter_part'] == 'abs':
                info_txt = "modulus"
            if zoom_each:
                info_txt += ", zoomed"
            fig0.suptitle("JTFS filterbank (%s)" % info_txt, **C['title_kw'])
    if viz_coeffs:
        fig_adjust(fig1)
        if axis_labels:
            fig1.suptitle("JTFS coefficients", **C['title_kw'])

    if savename is not None:
        if viz_filterbank:
            fig0.savefig(f'{savename}0.png', **C['savefig_kw'])
        if viz_coeffs:
            fig1.savefig(f'{savename}1.png', **C['savefig_kw'])

    if show:
        plt.show()
    else:
        if viz_filterbank:
            plt.close(fig0)
        if viz_coeffs:
            plt.close(fig1)


def scalogram(x, sc, fs=None, show_x=False, w=1., h=1., plot_cfg=None):
    """Compute and plot scalogram. Optionally plots `x`, separately.

    Parameters
    ----------
    x : np.ndarray
        Input, 1D.

    sc : Scattering1D
        Must be from NumPy backend, and have `average=False`. Will internally
        set `sc.oversampling=999` and `sc.max_order=1`.

    fs : None / int
        Sampling rate. If provided, will display physical units (Hz), else
        discrete (cycles/sample).

    show_x : bool (default False)
        Whether to plot `x` in time domain.

    w, h : float, float
        Scale width and height, separately.

    plot_cfg : None / dict
        Configures plotting. Will fill for missing values from defaults
        (see `plot_cfg_defaults` in source code). Supported key-values:

            'label_kw_xy' : dict
                Passed to all `ax.set_xlabel` and `ax.set_ylabel`.

            'title_kw' : dict
                Passed to all `ax.set_title.

            'imshow_kw' : dict
                Passed to `imshow`. Many kwargs are already set and can't
                be unset.

            'tick_params' : dict
                Passed to all `ax.tick_params`.

            'title_x' : str
                Title to show for plot of `x`, if applicable.

            'title_scalogram' : str
                Title to show for plot of scalogram.
    """
    # sanity checks
    assert isinstance(x, np.ndarray), type(x)
    assert x.ndim == 1, x.shape
    assert not sc.average
    assert 'numpy' in sc.__module__, sc.__module__

    # `plot_cfg`, defaults
    plot_cfg_defaults = {
        'label_kw_xy': dict(weight='bold', fontsize=14),
        'title_kw':    dict(weight='bold', fontsize=15),
        'imshow_kw':   dict(abs=1),
        'tick_params': dict(labelsize=12),
        'title_x': 'x',
        'title_scalogram': 'Scalogram',
    }
    _handle_global_scale(plot_cfg_defaults)
    # fill
    C = fill_default_args(plot_cfg, plot_cfg_defaults, copy_original=True,
                          check_against_defaults=True)

    # extract basic params, configure `sc`
    N = len(x)
    sc.oversampling = 999
    sc.max_order = 1

    # compute scalogram
    Scx = sc(x)
    meta = sc.meta()
    S1 = np.array([c['coef'].squeeze() for c in Scx])[meta['order'] == 1]

    # ticks & units
    if fs is not None:
        f_units = "[Hz]"
        t_units = "[sec]"
    else:
        f_units = "[cycles/sample]"
        t_units = "[samples]"

    yticks = np.array([p['xi'] for p in sc.psi1_f])
    if fs is not None:
        t = np.linspace(0, N/fs, N, endpoint=False)
        yticks *= fs
    else:
        t = np.arange(N)

    # axis labels
    xlabel  = (f"Time {t_units}",      C['label_kw_xy'])
    ylabel0 = ("Amplitude",            C['label_kw_xy'])
    ylabel1 = (f"Frequency {f_units}", C['label_kw_xy'])
    # titles
    title0 = (C['title_x'],         C['title_kw'])
    title1 = (C['title_scalogram'], C['title_kw'])
    # format yticks (limit # of shown decimal digits, and round the rest)
    yticks = _format_ticks(yticks)

    # plot ###################################################################
    width, height = tuple(_gscale() * np.array(CFG['VIZ']['figsize']))
    if show_x:
        figsize = _default_to_fig_wh((14, 5))
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax0, ax1 = axes

        _plot(t, x, xlabel=xlabel, ylabel=ylabel0, fig=fig, ax=ax0, title=title0,
              show=0)
        ax0.tick_params(**C['tick_params'])
        fig.subplots_adjust(wspace=.25)
    else:
        figsize = _default_to_fig_wh((6.5, 5))
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    _imshow(S1, xlabel=xlabel, ylabel=ylabel1, title=title1, yticks=yticks,
            xticks=t, fig=fig, ax=ax1, **C['imshow_kw'], show=0)
    ax1.tick_params(**C['tick_params'])
    plt.show()


# energy visuals #############################################################
def energy_profile_jtfs(Scx, meta, x=None, pairs=None, kind='l2', flatten=False,
                        plots=True, **plot_kw):
    """Plot & print relevant energy information across coefficient pairs.
    Works for all `'dict' in out_type` and `out_exclude`.
    Also see `help(wavespin.toolkit.coeff_energy)`.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    x : tensor
        Original input to print `E_out / E_in`.

    pairs: None / list/tuple[str]
        Computes energies for these pairs in provided order. None will compute
        for all in default order:
            ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
             'psi_t * psi_f_up', 'psi_t * psi_f_dn')

    kind : str['l1', 'l2']
        - L1: `sum(abs(x))`
        - L2: `sum(abs(x)**2)` -- actually L2^2

    flatten : bool (default False)
        If True, will return quantities on per-`n1` (per frequency row) basis,
        rather than per-`(n2, n1_fr)` (per joint slice).

    plots : bool (default True)
        Whether to visualize the energies and print statistics
        (will print E_out / E_in if `x` is passed regardless).

    plot_kw : kwargs
        Will pass to `wavespin.visuals.plot()`.

    Returns
    -------
    energies: list[float]
        List of coefficient energies.

    pair_energies: dict[str: float]
        Keys are pairs, values are sums of all pair's coefficient energies.
    """
    if not isinstance(Scx, dict):
        raise NotImplementedError("input must be dict. Set out_type='dict:array' "
                                  "or 'dict:list'.")
    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta)
    # make `titles`
    titles = _make_titles_jtfs(compute_pairs,
                               target="L1 norm" if kind == 'l1' else "Energy")
    # make `fn`
    fn = lambda Scx, meta, pair: coeff_energy(
        Scx, meta, pair, aggregate=False, kind=kind)

    # compute, plot, print
    energies, pair_energies = _iterate_coeff_pairs(
        Scx, meta, fn, pairs, plots=plots, flatten=flatten,
        titles=titles, **plot_kw)

    # E_out / E_in
    if x is not None:
        e_total = np.sum(energies)
        print("E_out / E_in = %.3f" % (e_total / energy(x)))
    return energies, pair_energies


def coeff_distance_jtfs(Scx0, Scx1, meta0, meta1=None, pairs=None, kind='l2',
                        flatten=False, plots=True, **plot_kw):
    """Computes relative distance between JTFS coefficients.

    Parameters
    ----------
    Scx0, Scx1: dict[list] / dict[np.ndarray]
        `jtfs(x0)`, `jtfs(x1)` (or `jtfs0` vs `jtfs1`, but see `meta1`).

    meta0: dict[dict[np.ndarray]]
        `jtfs.meta()` for `Scx0`.

    meta1: dict[dict[np.ndarray]] / None
        `jtfs.meta()` for `Scx1`. Configuration cannot differ in any way
        that alters coefficient shapes.

    pairs: None / list/tuple[str]
        Computes distances for these pairs in provided order. None will compute
        for all in default order:
            ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
             'psi_t * psi_f_up', 'psi_t * psi_f_dn')

    kind : str['l1', 'l2']
        - L1: `sum(abs(x))`
        - L2: `sum(abs(x)**2)` -- actually L2^2, i.e. energy

    flatten : bool (default False)
        If True, will return quantities on per-`n1` (per frequency row) basis,
        rather than per-`(n2, n1_fr)` (per joint slice).

    plots : bool (default True)
        Whether to visualize the distances.

    plot_kw : kwargs
        Will pass to `wavespin.visuals.plot()`.

    Returns
    -------
    distances : list[float]
        List of coefficient distances.

    pair_distances : dict[str: float]
        Keys are pairs, values are sums of all pair's coefficient distances.
    """
    if not all(isinstance(Scx, dict) for Scx in (Scx0, Scx1)):
        raise NotImplementedError("inputs must be dict. Set "
                                  "out_type='dict:array' or 'dict:list'.")
    if meta1 is None:
        meta1 = meta0

    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta0)
    # make `titles`
    titles = _make_titles_jtfs(compute_pairs,
                               target=("Absolute reldist" if kind == 'l1'
                                       else "Euclidean reldist"))
    # make `fn`
    fn = lambda Scx, meta, pair: coeff_distance(*Scx, *meta, pair, kind=kind)

    # compute, plot, print
    distances, pair_distances = _iterate_coeff_pairs(
        (Scx0, Scx1), (meta0, meta1), fn, pairs, plots=plots,
        titles=titles, **plot_kw)

    return distances, pair_distances


def _iterate_coeff_pairs(Scx, meta, fn, pairs=None, flatten=False, plots=True,
                         titles=None, **plot_kw):
    # in case multiple meta passed
    meta0 = meta[0] if isinstance(meta, tuple) else meta
    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta0)

    # extract energy info
    energies = []
    pair_energies = {}
    idxs = [0]
    for pair in compute_pairs:
        if pair not in meta0['n']:
            continue
        E_flat, E_slices = fn(Scx, meta, pair)
        data = E_flat if flatten else E_slices
        # flip to order freqs low-to-high
        pair_energies[pair] = data[::-1]
        energies.extend(data[::-1])
        # don't repeat 0
        idxs.append(len(energies) - 1 if len(energies) != 1 else 1)

    # format & plot ##########################################################
    energies = np.array(energies)
    ticks = np.arange(len(energies))
    vlines = (idxs, {'color': 'tab:red', 'linewidth': 1})

    if plots:
        # handle args
        if titles is None:
            titles = ('', '')
        plot_kw = deepcopy(plot_kw)  # preserve original
        plot_kw['ylims'] = plot_kw.get('ylims', (0, None))
        _handle_tick_params(plot_kw)
        figsize = tuple(_gscale() * np.array(CFG['VIZ']['figsize']))

        # plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _scat(ticks[idxs], energies[idxs], s=20, fig=fig, ax=ax)
        _plot(energies, vlines=vlines, title=titles[0], show=1, fig=fig, ax=ax,
              **plot_kw)

    # cumulative sum
    energies_cs = np.cumsum(energies)

    if plots:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _scat(ticks[idxs], energies_cs[idxs], s=20, fig=fig, ax=ax)
        _plot(energies_cs, vlines=vlines, title=titles[1], show=1, fig=fig, ax=ax,
              **plot_kw)

    # print report ###########################################################
    def sig_figs(x, n_sig=3):
        s = str(x)
        nondecimals = len(s.split('.')[0]) - int(s[0] == '0')
        decimals = max(n_sig - nondecimals, 0)
        return s.lstrip('0')[:decimals + nondecimals + 1].rstrip('.')

    e_total = np.sum(energies)
    pair_energies_sum = {pair: np.sum(pair_energies[pair])
                         for pair in pair_energies}
    nums = [sig_figs(e, n_sig=3) for e in pair_energies_sum.values()]
    longest_num = max(map(len, nums))

    if plots:
        i = 0
        for pair in compute_pairs:
            E_pair = pair_energies_sum[pair]
            eps = make_eps(e_total)
            e_perc = sig_figs(E_pair / (e_total + eps) * 100, n_sig=3)
            print("{} ({}%) -- {}".format(
                nums[i].ljust(longest_num), str(e_perc).rjust(4), pair))
            i += 1
    return energies, pair_energies


def compare_distances_jtfs(pair_distances, pair_distances_ref, plots=True,
                           verbose=True, title=None):
    """Compares distances as per-coefficient ratios, as a generally more viable
    alternative to the global L2 measure.

    Parameters
    ----------
    pair_distances : dict[tensor]
        (second) Output of `wavespin.visuals.coeff_distance_jtfs`, or alike.
        The numerator of the ratio.

    pair_distances_ref : dict[tensor]
        (second) Output of `wavespin.visuals.coeff_distance_jtfs`, or alike.
        The denominator of the ratio.

    plots : bool (default True)
        Whether to plot the ratios.

    verbose : bool (default True)
        Whether to print a summary of ratio statistics.

    title : str / None
        Will append to pre-made title.

    Returns
    -------
    ratios : dict[tensor]
        Distance ratios, keyed by pairs.

    stats : dict[tensor]
        Mean, minimum, and maximum of ratios along pairs, respectively,
        keyed by pairs.
    """
    # don't modify external
    pd0, pd1 = deepcopy(pair_distances), deepcopy(pair_distances_ref)

    ratios, stats = {}, {}
    for pair in pd0:
        p0, p1 = np.asarray(pd0[pair]), np.asarray(pd1[pair])
        # threshold out small points
        idxs = np.where((p0 < .001*p0.max()).astype(int) +
                        (p1 < .001*p1.max()).astype(int))[0]
        p0[idxs], p1[idxs] = 1, 1
        R = p0 / p1
        ratios[pair] = R
        stats[pair] = dict(mean=R.mean(), min=R.min(), max=R.max())

    if plots:
        # fetch quantities
        vidxs = np.cumsum([len(r) for r in ratios.values()])
        ratios_flat = np.array([r for rs in ratios.values() for r in rs])

        # styling
        if title is None:
            title = ''
        _title = _make_titles_jtfs(list(ratios), f"Distance ratios | {title}")[0]
        hlines = (1,     dict(color='tab:red', linestyle='--'))
        vlines = (vidxs, dict(color='k', linewidth=1))
        plot_kw = {}
        _handle_tick_params(plot_kw)

        # plot
        _plot(ratios_flat, title=_title, hlines=hlines, vlines=vlines,
              ylims=(0, None), **plot_kw)
        _scat(idxs, ratios_flat[idxs], color='tab:red', show=1)

    if verbose:
        print("Ratios (Sx/Sx_ref):")
        print("mean  min   max   | pair")
        for pair in ratios:
            print("{:<5.2f} {:<5.2f} {:<5.2f} | {}".format(
                *list(stats[pair].values()), pair))
    return ratios, stats


# utils ######################################################################
# `_plot` and others are for when global scaling is already handled
def _plot(*args, **kwargs):
    """`plot` with `do_gscale=False`."""
    plot(*args, **kwargs, do_gscale=False)


def _scat(*args, **kwargs):
    """`scat` with `do_gscale=False`."""
    scat(*args, **kwargs, do_gscale=False)


def _imshow(*args, **kwargs):
    """`imshow` with `do_gscale=False`."""
    imshow(*args, **kwargs, do_gscale=False)


def _make_titles_jtfs(compute_pairs, target):
    """For energies and distances."""
    # make `titles`
    titles = []
    pair_aliases = {'psi_t * phi_f': '* phi_f', 'phi_t * psi_f': 'phi_t *',
                    'psi_t * psi_f_up': 'up', 'psi_t * psi_f_dn': 'down'}
    title = "%s | " % target
    for pair in compute_pairs:
        if pair in pair_aliases:
            title += "{}, ".format(pair_aliases[pair])
        else:
            title += "{}, ".format(pair)
    title = title.rstrip(', ')
    titles.append(title)

    title = "cumsum(%s)" % target
    titles.append(title)

    # add font size & family
    tkw = {'fontsize': CFG['VIZ']['title']['fontsize'] * _gscale_r(),
           'fontfamily': CFG['VIZ']['long_title_fontfamily'][1]}
    for i in range(len(titles)):
        titles[i] = (titles[i], tkw)
    return titles


def _filterbank_style_axes(ax, N, xlims, ymax=None, zoom=None, is_jtfs=False):
    if zoom != -1:
        xticks = np.linspace(0, N, 9, endpoint=1).astype(int)
        # x limits and labels
        w = np.linspace(0, 1, len(xticks), 1)
        w[w > .5] -= 1
        ax.set_xticks(xticks[:-1])
        ax.set_xticklabels(w[:-1])
        ax.set_xlim(*xlims)
    else:
        xticks = np.linspace(0, N, 9, endpoint=1).astype(int)
        w = [-.5, -.375, -.25, -.125, 0, .125, .25, .375, .5]
        ax.set_xticks(xticks)
        ax.set_xticklabels(w)

    # y limits
    ax.set_ylim(-.05, ymax)


def _filterbank_plots_handle_global_scale(plot_kw):
    if 'title' in plot_kw and not isinstance(plot_kw['title'], tuple):
        fscaled = CFG['VIZ']['title']['fontsize'] * _gscale_r()
        tkw = {'fontsize': fscaled,
               'fontfamily': CFG['VIZ']['long_title_fontfamily'][1]}
        plot_kw['title'] = (plot_kw['title'], tkw)
