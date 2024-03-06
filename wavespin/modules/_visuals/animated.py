# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Animated visuals.

See `examples/visuals_tour.py`, or
https://wavespon.readthedocs.io/en/latest/examples/visuals_tour.html
"""
import os
import warnings
from pathlib import Path
import numpy as np
from scipy.fft import ifft, ifftshift

from ...toolkit import pack_coeffs_jtfs, energy, drop_batch_dim_jtfs
from ...utils.gen_utils import fill_default_args
from ...scattering1d.filter_bank import morlet_1d, gauss_1d
from ... import Scattering1D, CFG
from .primitives import (
    imshow, plot_box, _check_savepath, _ticks,
    _gscale, _gscale_r, _default_to_fig_wh, _no_ticks_borders,
)
from .static import _equalize_pairs_jtfs
from . import plt, animation


def gif_jtfs_2d(Scx, meta, savedir='', base_name='jtfs2d', images_ext='.png',
                overwrite=False, save_images=None, show=None, cmap='turbo',
                norms=None, skip_spins=False, skip_unspinned=False, sample_idx=0,
                verbose=True, gif_kw=None):
    """Slice heatmaps of JTFS outputs.

    Parameters
    ----------
    Scx : dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta : dict[dict[np.ndarray]]
        `jtfs.meta()`.

    savedir : str / None
        Path of directory to save GIF/images to. Defaults to current
        working directory. `None` to not save.

    base_name : str
        Will save gif with this name, and images with same name enumerated.
        Should not include any extension.

    images_ext : str
        Generates images with this format. '.png' (default) is lossless but takes
        more space, '.jpg' is compressed.

    overwrite : bool (default False)
        If True and file at `savepath` exists, will overwrite it.

    save_images : bool
        Whether to save images. Images are always saved if `savedir` is not None,
        but deleted after if `save_images=False`.
        If `True` and `savedir` is None, will save images to current working
        directory (but won't gif).

        See `show` concerning default behavior.

    show : None / bool
        Whether to display images to console.

        Defaults based on other args, see
        `help(wavespin.visuals.animated._handle_gif_args)`.

    cmap : str
        Passed to `imshow`.

    norms: None / tuple / float
        Plot color norms for 1) `psi_t * psi_f`, 2) `psi_t * phi_f`, and
        3) `phi_t * psi_f` pairs, respectively.

            - tuple: of length tree (upper limits only, lower assumed 0).
            - float: will reuse this value
            - None: will norm to `.5 * max(coeffs)`, where coeffs = all joint
              coeffs except `phi_t * phi_f`.

    skip_spins: bool (default False)
        Whether to skip `psi_t * psi_f` pairs.

    skip_unspinned: bool (default False)
        Whether to skip `phi_t * phi_f`, `phi_t * psi_f`, `psi_t * phi_f`
        pairs.

    sample_idx : int (default 0)
        Index of sample in batched input to visualize.

    verbose : bool (default True)
        Whether to print to console the location of save file upon success.

    gif_kw : dict / None
        Passed as kwargs to `wavespin.visuals.make_gif`.

    Example
    -------
    ::

        N, J, Q = 2049, 7, 16
        x = toolkit.echirp(N)

        jtfs = TimeFrequencyScattering1D(N, J, Q, J_fr=4, Q_fr=2,
                                         out_type='dict:list')
        Scx = jtfs(x)
        meta = jtfs.meta()

        gif_jtfs_2d(Scx, meta)
    """
    def _title(meta_idx, pair, spin):
        txt = r"$|\Psi_{%s, %s, %s} \star \mathrm{U1}|$"
        values = ns[pair][meta_idx[0]]
        assert values.ndim == 1, values
        mu, l, _ = [int(n) if (float(n).is_integer() and n >= 0) else r'\infty'
                    for n in values]
        return (txt % (mu, l, spin), {'fontsize': 14 * _gscale_r()})

    def _n_n1s(pair):
        n2, n1_fr, _ = ns[pair][meta_idx[0]]
        return np.sum(np.all(ns[pair][:, :2] == np.array([n2, n1_fr]), axis=1))

    def _get_coef(i, pair, meta_idx):
        n_n1s = _n_n1s(pair)
        start, end = meta_idx[0], meta_idx[0] + n_n1s
        if out_list:
            coef = Scx[pair][i]['coef']
        elif out_3D:
            coef = Scx[pair][i]
        else:
            coef = Scx[pair][start:end]
        assert len(coef) == n_n1s
        return coef

    def _save_image(fig):
        path = os.path.join(savedir, f'{base_name}{img_idx[0]}{images_ext}')
        if os.path.isfile(path) and overwrite:
            os.unlink(path)
        if not os.path.isfile(path):
            fig.savefig(path, bbox_inches='tight')
        img_paths.append(path)
        img_idx[0] += 1

    def _two_col_subplot():
        figsize = _default_to_fig_wh((10, 5))  # handles global scaling
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        return fig, axes

    def _viz_spins(Scx, i, ckw):
        kup = 'psi_t * psi_f_up'
        kdn = 'psi_t * psi_f_dn'
        sup = _get_coef(i, kup, meta_idx)
        sdn = _get_coef(i, kdn, meta_idx)

        fig, axes = _two_col_subplot()
        kw = dict(**ckw, fig=fig)

        _imshow(sup, ax=axes[0], **kw, title=_title(meta_idx, kup, '+1'))
        _imshow(sdn, ax=axes[1], **kw, title=_title(meta_idx, kdn, '-1'))
        plt.subplots_adjust(wspace=0.01)
        if save_images or do_gif:
            _save_image(fig)
        if show:  # no-cov
            plt.show()
        plt.close(fig)

        meta_idx[0] += len(sup)

    def _viz_simple(Scx, pair, i, ckw):
        coef = _get_coef(i, pair, meta_idx)

        kw = dict(**ckw, title=_title(meta_idx, pair, '0'))
        if do_gif:
            # make spacing (alignment in gif) consistent with up & down
            fig, axes = _two_col_subplot()
            _imshow(coef, ax=axes[0], fig=fig, **kw)
            fig.subplots_adjust(wspace=0.01)
            _no_ticks_borders(axes[1])
        else:
            # optimize spacing for single image
            figsize = tuple(np.array(CFG['VIZ']['figsize']) * _gscale())
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            _imshow(coef, **kw, fig=fig, ax=ax)
        if save_images or do_gif:
            _save_image(fig)
        if show:  # no-cov
            plt.show()
        plt.close(fig)

        meta_idx[0] += len(coef)

    # handle args & check if already exists (if so, delete if `overwrite`)
    (savedir, savepath, images_ext, base_name, save_images, show, do_gif
     ) = _handle_gif_args(
        savedir, base_name, images_ext, save_images, overwrite, show)

    # set params
    out_3D = bool(meta['n']['psi_t * phi_f'].ndim == 3)
    out_list = isinstance(Scx['S0'], list)
    ns = {pair: meta['n'][pair].reshape(-1, 3) for pair in meta['n']}

    Scx = drop_batch_dim_jtfs(Scx, sample_idx)

    n_norms = 5
    if isinstance(norms, (list, tuple)):
        assert len(norms) == n_norms, (len(norms), n_norms)
        norms = [(0, n) for n in norms]
    elif isinstance(norms, float):
        norms = [(0, norms) for _ in range(n_norms)]
    else:
        # set to .5 times the max of any joint coefficient (except phi_t * phi_f)
        mx = np.max([(c['coef'] if out_list else c).max()
                     for pair in Scx for c in Scx[pair]
                     if pair not in ('S0', 'S1', 'phi_t * phi_f')])
        norms = [(0, .5 * mx)] * n_norms

    ckw = dict(abs=1, ticks=0, cmap=cmap)

    # spinned pairs ##########################################################
    img_paths = []
    img_idx = [0]
    meta_idx = [0]
    if not skip_spins:
        ckw['norm'] = norms[0]
        i = 0
        while True:
            _viz_spins(Scx, i, ckw)
            i += 1
            if meta_idx[0] > len(ns['psi_t * psi_f_up']) - 1:
                break

    # unspinned pairs ########################################################
    if not skip_unspinned:
        pairs = ('psi_t * phi_f', 'phi_t * psi_f', 'phi_t * phi_f')
        for j, pair in enumerate(pairs):
            ckw['norm'] = norms[1 + j]
            meta_idx = [0]
            i = 0
            while True:
                _viz_simple(Scx, pair, i, ckw)
                i += 1
                if meta_idx[0] > len(ns[pair]) - 1:
                    break

    # make gif & cleanup #####################################################
    new_paths = _rename_to_sort_alphabetically(img_paths, base_name, images_ext)
    try:
        if do_gif:
            if gif_kw is None:
                gif_kw = {}
            make_gif(loaddir=savedir, savepath=savepath, ext=images_ext,
                     overwrite=overwrite, delimiter=base_name, verbose=verbose,
                     delete_images=False, **gif_kw)
    finally:
        if not save_images:
            # guarantee cleanup
            for path in new_paths:
                if os.path.isfile(path):
                    os.unlink(path)


def gif_jtfs_3d(Scx, jtfs=None, preset='spinned', savedir='',
                base_name='jtfs3d', images_ext='.png', cmap='turbo', cmap_norm=.5,
                axes_labels=('xi2', 'xi1_fr', 'xi1'), overwrite=False,
                save_images=False, width=800, height=800, surface_count=30,
                opacity=.2, zoom=1, angles=None, equalize_pairs=False,
                verbose=True, gif_kw=None):
    """Generate and save GIF of 3D JTFS slices.

    NOTE: as of Kaleido `0.2.1`, it's bugged on Windows, downgrade to `0.1.0`.
    https://github.com/plotly/Kaleido/issues/134

    Parameters
    ----------
    Scx : dict / tensor, 4D
        Output of `jtfs(x)` with `out_type='dict:array'` or `'dict:list'`,
        or output of `wavespin.toolkit.pack_coeffs_jtfs()`.

        Note, axes are always labeled `0` to `0.5`, so `paths_exclude`
        isn't accounted for.

    jtfs : TimeFrequencyScattering1D
        Required if `Scx` is dict or `preset` is not `None`.

    preset : str['spinned', 'all'] / None
        If `Scx = jtfs(x)`, then

            - `'spinned'`: show only `psi_t * psi_f_up` and `psi_t * psi_f_dn`
              pairs
            - `'all'`: show all pairs

        `None` is for when `Scx` is already packed via `pack_coeffs_jtfs`.

    savedir, base_name, images_ext, overwrite :
        See `help(wavespin.visuals.gif_jtfs_2d)`.

    cmap : str
        Colormap to use.

    cmap_norm : float
        Colormap norm to use, as fraction of maximum value of `packed`
        (i.e. `norm=(0, cmap_norm * packed.max())`).

    axes_labels : tuple[str]
        Names of last three dimensions of `packed`. E.g. `structure==2`
        (in `pack_coeffs_jtfs`) will output `(n2, n1_fr, n1, t)`, so
        `('xi2', 'xi1_fr', 'xi1')` (default).

    width : int
        2D width of each image (GIF frame), in pixels.

    height : int
        2D height of each image (GIF frame), in pixels.

    surface_count : int
        Greater improves 3D detail of each frame, but takes longer to render.

    opacity : float
        Lesser makes 3D surfaces more transparent, exposing more detail.

    zoom : float (default=1) / None
        Zoom factor on each 3D frame. If None, won't modify `angles`.
        If not None, will first divide by L2 norm of `angles`, then by `zoom`.

    angles : None / np.ndarray / list/tuple[np.ndarray] / str['rotate']
        Controls display angle of the GIF.

          - None: default angle that faces the line extending from min to max
            of `xi1`, `xi2`, and `xi1_fr` (assuming default `axes_labels`).
          - Single 1D array: will reuse for each frame.
          - 'rotate': will use a preset that rotates the display about the
            default angle.

        Resulting array is passed to `go.Figure.update_layout()` as
        `'layout_kw': {'scene_camera': 'center': dict(x=e[0], y=e[1], z=e[2])}`,
        where `e = angles[0]` up to `e = angles[len(packed) - 1]`.

    equalize_pairs : bool (default False)
        See `equalize_pairs` in `help(wavespin.visuals.viz_jtfs_2d)`.

    verbose : bool (default True)
        Whether to print GIF generation progress.

    gif_kw : dict / None
        Passed as kwargs to `wavespin.visuals.make_gif`.

    Example
    -------
    ::

        N, J, Q = 2049, 7, 16
        x = toolkit.echirp(N)

        jtfs = TimeFrequencyScattering1D(N, J, Q, J_fr=4, Q_fr=2,
                                         out_type='dict:list')
        Scx = jtfs(x)
        gif_jtfs_3d(Scx, jtfs, savedir='', preset='spinned')
    """
    try:
        import plotly.graph_objs as go
    except ImportError as e:  # no-cov
        print("\n`plotly.graph_objs` is needed for `gif_jtfs_3d`.")
        raise e

    # handle args & check if gif already exists (if so, delete if `overwrite`)
    (savedir, savepath_gif, images_ext, base_name, save_images, *_
     ) = _handle_gif_args(
         savedir, base_name, images_ext, save_images, overwrite, show=False)
    if preset not in ('spinned', 'all', None):  # no-cov
        raise ValueError("`preset` must be 'spinned', 'all', or None (got %s)" % (
            preset))

    # handle `axes_labels`
    supported = ('t', 'xi2', 'xi1_fr', 'xi1')
    for label in axes_labels:
        if label not in supported:  # no-cov
            raise ValueError(("unsupported `axes_labels` element: {} -- must "
                              "be one of: {}").format(
                                  label, ', '.join(supported)))
    frame_label = [label for label in supported if label not in axes_labels][0]

    # handle `Scx`
    if not isinstance(Scx, (dict, np.ndarray)):  # no-cov
        raise ValueError("`Scx` must be dict or numpy array (need `out_type` "
                         "'dict:array' or 'dict:list'). Got %s" % type(Scx))
    elif isinstance(Scx, dict):
        assert jtfs is not None
        if equalize_pairs:
            Scx = _equalize_pairs_jtfs(Scx)
        ckw = dict(Scx=Scx, meta=jtfs.meta(), reverse_n1=False,
                   out_3D=jtfs.out_3D,
                   sampling_psi_fr=jtfs.scf.sampling_psi_fr)
        if preset == 'spinned':
            _packed = pack_coeffs_jtfs(structure=2, separate_lowpass=True, **ckw)
            _packed = _packed[0]  # spinned only
        elif preset == 'all':
            _packed = pack_coeffs_jtfs(structure=2, separate_lowpass=False, **ckw)
        _packed = _packed[0]  # drop batch dim
        packed = _packed.transpose(-1, 0, 1, 2)  # time first
    elif isinstance(Scx, np.ndarray):
        packed = Scx[0]  # drop batch dim

    # 3D meshgrid
    def slc(i, g):
        label = axes_labels[i]
        start = {'xi1': .5, 'xi2': .5, 't': 0, 'xi1_fr': .5}[label]
        end   = {'xi1': 0., 'xi2': 0., 't': 1, 'xi1_fr': -.5}[label]
        return slice(start, end, g*1j)

    a, b, c = packed.shape[1:]
    X, Y, Z = np.mgrid[slc(0, a), slc(1, b), slc(2, c)]

    # handle `angles`; camera focus
    if angles is None:
        eye = np.array([2.5, .3, 2])
        eye /= np.linalg.norm(eye)
        eyes = [eye] * len(packed)
    elif (isinstance(angles, (list, tuple)) or
          (isinstance(angles, np.ndarray) and angles.ndim == 2)):
        eyes = angles
    elif isinstance(angles, str):
        assert angles == 'rotate', angles
        n_pts = len(packed)

        def gauss(n_pts, mn, mx, width=20):
            t = np.linspace(0, 1, n_pts)
            g = np.exp(-(t - .5)**2 * width)
            g *= (mx - mn)
            g += mn
            return g

        # alt scheme
        # x = np.logspace(np.log10(2.5), np.log10(8.5), n_pts, endpoint=1)
        # y = np.logspace(np.log10(0.3), np.log10(6.3), n_pts, endpoint=1)
        # z = np.logspace(np.log10(2.0), np.log10(2.0), n_pts, endpoint=1)

        x, y, z = [gauss(n_pts, mn, mx) for (mn, mx)
                   in [(2.5, 8.5), (0.3, 6.3), (2, 2)]]

        eyes = np.vstack([x, y, z]).T
    else:
        eyes = [angles] * len(packed)
    assert len(eyes) == len(packed), (len(eyes), len(packed))

    # camera zoom
    if zoom is not None:
        for i in range(len(eyes)):
            eyes[i] /= (np.linalg.norm(eyes[i]) * .5 * zoom)
    # colormap norm
    mx = cmap_norm * packed.max()

    # gif configs
    volume_kw = dict(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        opacity=opacity,
        surface_count=surface_count,
        colorscale=cmap,
        showscale=False,
        cmin=0,
        cmax=mx,
    )
    layout_kw = dict(
        margin_pad=0,
        margin_l=0,
        margin_r=0,
        margin_t=0,
        title_pad_t=0,
        title_pad_b=0,
        margin_autoexpand=False,
        scene_aspectmode='cube',
        width=width,
        height=height,
        scene=dict(
            xaxis_title=axes_labels[0],
            yaxis_title=axes_labels[1],
            zaxis_title=axes_labels[2],
        ),
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
        ),
    )

    # generate gif frames ####################################################
    img_paths = []
    for k, vol4 in enumerate(packed):
        # make frame ---------------------------------------------------------
        fig = go.Figure(go.Volume(value=vol4.flatten(), **volume_kw))

        eye = dict(x=eyes[k][0], y=eyes[k][1], z=eyes[k][2])
        layout_kw['scene_camera']['eye'] = eye
        fig.update_layout(
            **layout_kw,
            title={'text': f"{frame_label}={k}",
                   'x': .5, 'y': .09,
                   'xanchor': 'center', 'yanchor': 'top'}
        )

        # save frame ---------------------------------------------------------
        savepath = os.path.join(savedir, f'{base_name}{k}{images_ext}')
        if os.path.isfile(savepath) and overwrite:
            os.unlink(savepath)
        fig.write_image(savepath)
        img_paths.append(savepath)
        if verbose:  # no-cov
            print("{}/{} frames done".format(k + 1, len(packed)), flush=True)

    # make gif ###############################################################
    new_paths = _rename_to_sort_alphabetically(img_paths, base_name, images_ext)
    try:
        if gif_kw is None:
            gif_kw = {}
        make_gif(loaddir=savedir, savepath=savepath_gif, ext=images_ext,
                 delimiter=base_name, overwrite=overwrite, verbose=verbose,
                 **gif_kw)
    finally:
        if not save_images:
            # guarantee cleanup
            for path in new_paths:
                if os.path.isfile(path):
                    os.unlink(path)


def viz_top_fdts(jtfs, x, top_k=4, savepath=None, measure='energy', fs=None,
                 render='gif', wav_zoom=1.05, patch_size=(.2, .2), idxs=None,
                 render_kw=None, fps=0.5, close_figs=True):
    """Shows the top spinned coefficients along their activation localizations
    on the scalogram, and generating wavelets.

    Parameters
    ----------
    jtfs : TimeFrequencyScattering1D
        JTFS instance.

    x : tensor
        1D input.

    top_k : int[>0]
        Number of coefficients to show, sorted from greatest to lowest.

    savepath : str / None
        Path to save output to. Defaults to 'viz.gif' or 'viz.mp4'
        (suffix set from `render`).

    measure : str
        Measure by which to sort coefficients (what decides "top"). One of:

            - `'energy'`: energy of a joint slice.
            - `'max'`: maximum of a joint slice.
            - `'energy-max'`: energy of a sub-region of a joint slice, centered
              about its maximum.

        A "joint slice" is a `(n2, n1_fr)` slice of the 4D JTFS tensor.

    fs : None / int
        Sampling rate. Affects displayed x-axis labels and slope measures.
        Defaults to `jtfs.N`.

    render : str
        How to display:

            - `'show'`: `plt.show()`, won't save anything
            - `'gif'`: generates a .gif using `make_gif`
            - `'mp4'`: generates an .mp4 using `matplotlib.animation`.

        To make a GIF, choosing `'mp4'` then converting (via e.g. ezgif.com) may
        yield higher quality results.

    wav_zoom : float
        Zoom factor on the displayed wavelets. By default will set zoom such that
        the widest wavelet is well-shown.

    patch_size : tuple[float]
        Rectangular region around maximum to zero (see "Overview"), specified
        as tuple `psy, psx = patch_size`, with `0 < psy <= 1` specifying the
        fraction of length along vertical (frequency), and `psx` along horizontal.
        Defaults to a value that should work well generally.

    idxs : None / list[int]
        Custom indices to display. This method returns the indices in order of
        which they were displayed, which can be used to hand-pick the desired
        frames.

    render_kw : None / dict
        Passed to `make_gif` for `render='gif'` or `FDTSAnimator` for
        `render='mp4'`.

    fps : float
        Frames per second for `render='mp4'`. Defaults to `0.5`.

    close_figs : bool (default True)
        Whether to `plt.close(fig)` after `plt.show()` or `plt.save()`.
        This arg is mostly for development: accumulating open figures is
        discouraged and may leak memory, but is needed for documentation builds
        (image scraping).

    Returns
    -------
    idxs_done : list[int]
        Indices that slice the unrolled concatenation of 4D-packed up and down
        coefficients, in order of which they were displayed. Useful via `idxs`.

    data : dict
        Relevant data for each frame. Used internally for `render='mp4'`.

    Overview
    --------
    The method fetches top_k spinned coefficients by sorting them according to
    `measure`, while striving to return "unique" coefficients such that we don't
    show weaker activations of the same strongest FDTS region.

    Localization in the source scalogram is obtained from the index of maximum
    of each coefficient. The bounding boxes' heights and widths are set from
    the temporal and frequential 'width' metas of the generating wavelets.

    High `T` or `F` perform poorly, as they're supposed to, but improvements can
    be made via `oversampling > 0` or `oversampling_fr > 0` to counter the
    additional resolution limitation due to a finite coordinate grid.

    Uniqueness is achieved as follows. Once one coefficient is obtained, the
    spatial patch around it is carved out (zeroed) for all subsequent searches.
    This however isn't foulproof, and more advanced methods are required to match
    what a human would pick by hand. Instead, `idxs` is provided.
    """
    def wheremax(x):
        return tuple(map(lambda x: x[0], np.where(x == x.max())))

    def next_idx(odnf, idxs_done):
        maxima = []
        for slc_idx, slc in enumerate(odnf):
            if slc_idx in idxs_done:
                maxima.append(0)
                continue

            slc = slc.copy()
            for ixs_y, ixs_x in zip(ixs_y_all, ixs_x_all):
                slc[ixs_y, ixs_x] = 0

            mx_idx_y, mx_idx_x = wheremax(slc)
            for ixs_y, ixs_x in zip(ixs_y_all, ixs_x_all):
                if (ixs_y.start <= mx_idx_y < ixs_y.stop or
                    ixs_x.start <= mx_idx_x < ixs_x.stop):
                    slc_metric = 0
            else:
                if measure == 'max':
                    slc_metric = slc.max()
                elif measure == 'energy':
                    slc_metric = energy(slc)
                elif measure == 'energy-max':
                    scy, scx = wheremax(slc)
                    n2, n1_fr = ns[slc_idx % len(ns)]
                    psi_id = jtfs.scf.psi_ids[jtfs.scf.scale_diffs[n2]]
                    wdx = jtfs.psi2_f[n2]['width'][0]
                    wdy = jtfs.psi1_f_fr_up['width'][psi_id][n1_fr]
                    eixs_y = slice(max(scy - wdy*2, 0), scy + wdy*4 + 1)
                    eixs_x = slice(max(scx - wdx*2, 0), scx + wdx*4 + 1)
                    slc_metric = energy(slc[eixs_y, eixs_x])

            maxima.append(slc_metric)

        idxs = np.argsort(maxima)[::-1]
        return idxs[0]

    # handle args ############################################################
    if render == 'show':  # no-cov
        if savepath is not None:
            warnings.warn("`savepath` does nothing with `render='show'`.")
    elif render in ('gif', 'mp4'):  # no-cov
        if savepath is None:
            savepath = 'viz_top_fdts.' + render
    else:  # no-cov
        raise ValueError(("Unsupported `render` %s; must be one of: 'show', "
                          "'gif', 'mp4'") % render)
    assert top_k > 0, top_k

    # compute JTFS and scalogram #############################################
    Scx = jtfs(x)

    # instantiate time scattering object based off of JTFS, get only U1
    sc = Scattering1D(average=False, out_type='list', max_order=1,
                      oversampling=99, frontend='numpy',
                      **{k: getattr(jtfs, k) for k in
                         ('shape', 'Q', 'J', 'r_psi', 'normalize',
                          'max_pad_factor', 'pad_mode')})
    scgram = sc(x)
    scgram = np.array([c['coef'] for c in scgram[1:]])

    # pack jtfs into 4D structs ##############################################
    # get meta
    jmeta = jtfs.meta()
    # unique (n2, n1_fr) meta
    if jtfs.out_3D:
        ns = jmeta['n']['psi_t * psi_f_up'][:, 0, :2]
        slopes = jmeta['slope']['psi_t * psi_f_up'][:, 0]
    else:
        _ns = jmeta['n']['psi_t * psi_f_up']
        unique_ixs = np.where(_ns[:, -1] == 0)[0]
        ns = _ns[unique_ixs][:, :2]
        slopes = jmeta['slope']['psi_t * psi_f_up'][unique_ixs]
    # pack
    outs = pack_coeffs_jtfs(
        Scx, jmeta, structure=5, out_3D=jtfs.out_3D,
        sampling_psi_fr=jtfs.scf.sampling_psi_fr)
    oup, odn, *_ = outs

    # cat up & down, unroll n2 & n1_fr
    odnf = odn.reshape(-1, *odn.shape[-2:])
    oupf = oup.reshape(-1, *oup.shape[-2:])
    odnf = np.concatenate([odnf, oupf], axis=0)

    # compute relevant params ################################################
    if idxs is None:
        slc_shape = odnf.shape[-2:]
        patch_y, patch_x = [int(np.ceil( slc_shape[j] * patch_size[j] ))
                            for j in (0, 1)]
        wy, wx = patch_y//2, patch_x//2

    # main loop ##############################################################
    if idxs is None:
        ixs_y_all, ixs_x_all = [], []
        idxs_done = []
        while len(idxs_done) < top_k:
            idx = next_idx(odnf, idxs_done)

            slc = odnf[idx]
            cy, cx = wheremax(slc)
            ixs_y = slice(max(cy - wy, 0), cy + wy + 1)
            ixs_x = slice(max(cx - wx, 0), cx + wx + 1)
            ixs_y_all.append(ixs_y)
            ixs_x_all.append(ixs_x)

            idxs_done.append(idx)
    else:
        idxs_done = idxs

    # visualize ##############################################################
    # compute relevant params
    N = jtfs.N
    if fs is None:
        fs = N
    t = np.linspace(0, N/fs, N, endpoint=False)
    freqs = np.array([p['xi'] for p in jtfs.psi1_f]) * fs
    n_freqs = len(freqs)
    # maximum width, for displaying the wavelets (see comment in viz_jtfs_2d)
    wt_max = int(max(p['width'][0] for p in jtfs.psi2_f) * 8 / wav_zoom)
    wf_max = int(max(jtfs.psi1_f_fr_up['width'][0])      * 8 / wav_zoom)
    # plot color norm
    mx = max(oup.max(), odn.max()) * .95
    # total spinned energy
    e_total = energy(oup) + energy(odn)
    # images savedir, based on gif savepath
    if render == 'gif':
        savedir = Path(savepath).parent
        # image params
        img_ext = '.png'
        img_delimiter = 'im'
    # track gif data
    data = {'boxes00': [], 'imgs10': [], 'imgs11': [], 'titles': [], 'n': [],
            'imgs00': scgram, 't': t, 'freqs': freqs, 'mx': mx}

    # main loop
    for i, idx in enumerate(idxs_done):
        # get plot info ######################################################
        # fetch slice
        slc = odnf[idx]

        # compute max and translate it to scalogram's coordinates
        cy, cx = wheremax(slc)
        cy *= n_freqs / len(slc)
        cx *= N / slc.shape[-1]
        ctr = (cy, cx)

        # fetch relevant meta
        n2, n1_fr = ns[idx % len(ns)]
        slope = slopes[idx % len(ns)] / jtfs.Q[0] * fs
        psi_id = jtfs.scf.scale_diffs[n2]

        # joint wavelet's widths in scalogram's coordinates
        wd_t = jtfs.psi2_f[n2]['width'][0]
        wd_f = jtfs.psi1_f_fr_up['width'][psi_id][n1_fr]
        wd_f *= len(jtfs.psi1_f_fr_up[0][0]) / len(jtfs.psi1_f_fr_up[psi_id][0])
        w = (wd_f, wd_t)

        # compute time-domain wavelet
        pt_f = jtfs.psi2_f[n2][0]
        up = bool(idx > odnf.shape[0]//2)
        if up:
            pf_f = jtfs.psi1_f_fr_up[psi_id][n1_fr]
        else:
            pf_f = jtfs.psi1_f_fr_dn[psi_id][n1_fr]
        pt, pf = [ifftshift(ifft(p.squeeze())) for p in (pt_f, pf_f)]
        Psi = pf[:, None] * pt[None]
        # trim and time-reverse along freq (since scalogram low idx <=> high freq)
        Pcy, Pcx = Psi.shape
        Pcy, Pcx = Pcy//2, Pcx//2
        Psi = Psi[max(Pcy - wf_max, 0):Pcy + wf_max,
                  max(Pcx - wt_max, 0):Pcx + wt_max][:, ::-1]

        # do plotting ########################################################
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=CFG['VIZ']['dpi'])
        gs = axes[0, 0].get_gridspec()
        for ax in axes.flat:
            ax.remove()
        del ax
        ax0 = fig.add_subplot(gs[0, :2])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        efrac = "%.3g" % (energy(slc) / e_total * 100)
        title = "{}{:.3g} octaves/sec | {}% energy (spinned)".format(
            '-' if up else '+', abs(slope), efrac)

        # scalogram w/ focus box
        _imshow(scgram, fig=fig, ax=ax0, abs=1, interpolation='none',
                title=title, xticks=t, yticks=freqs,
                xlabel="time [sec]", ylabel="frequency [Hz]")
        _plot_box(ctr, w, fig=fig, ax=ax0, xmax=N - 1, ymax=len(scgram) - 1)

        # JTFS coeff
        _imshow(slc, fig=fig, ax=ax1, abs=1, interpolation='none',
                ticks=0, norm=(0, mx))
        # the wavelet
        _imshow(Psi.real, fig=fig, ax=ax2, ticks=0)

        # postprocess
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=.05, hspace=.12)
        ax0.set_xlim(0, N - 1)
        ax0.set_ylim(len(scgram) - 1, 0)

        # finish according to `render`
        if render == 'gif':
            # save image in same directory as the gif
            img_savepath = Path(savedir, img_delimiter + str(i) + img_ext)
            fig.savefig(img_savepath, bbox_inches='tight')
        elif render == 'show':
            plt.show()
        if close_figs:
            plt.close(fig)

        # append data
        data['boxes00'].append((ctr, w))
        data['imgs10'].append(slc)
        data['imgs11'].append(Psi.real)
        data['titles'].append(title)
        data['n'].append((n2, n1_fr))

    # save gif ###############################################################
    if render == 'gif':
        # handle kwargs
        defaults = dict(overwrite=True, delete_images=True, verbose=True,
                        duration=1000, start_end_pause=0, HD=True)
        render_kw = fill_default_args(render_kw, defaults, copy_original=True)

        # animate
        make_gif(savedir, savepath, delimiter=img_delimiter, ext=img_ext,
                 **render_kw)
    elif render == 'mp4':
        # handle kwargs
        defaults = {
            'imshow_kw00': dict(aspect='auto', animated=True, cmap='turbo'),
            'imshow_kw10': dict(aspect='auto', animated=True, cmap='turbo',
                                interpolation='none'),
            'imshow_kw11': dict(aspect='auto', animated=True, cmap='bwr',
                                interpolation='none'),
        }
        render_kw = fill_default_args(render_kw, defaults, copy_original=True)

        # process data
        imgs00, imgs10, imgs11, boxes00, mx, t, freqs, titles = [
            data[k] for k in
            'imgs00 imgs10 imgs11 boxes00 mx t freqs titles'.split()]
        imgs00 = [imgs00] * len(imgs10)

        inflator = 10
        inflate = int(max(1, inflator / fps))
        fps_inflated = inflate * fps

        # animate
        ani = FDTSAnimator(imgs00, imgs10, imgs11, boxes00, mx, t, freqs, titles,
                           inflate=inflate, **render_kw)
        ani.save(savepath, fps=fps_inflated, savefig_kwargs=dict(pad_inches=0))

    # return #################################################################
    return idxs_done, data


class FDTSAnimator(animation.TimedAnimation):
    def __init__(self, imgs00, imgs10, imgs11, boxes00, mx, t, freqs, titles,
                 inflate, imshow_kw00, imshow_kw10, imshow_kw11):
        self.imgs00 = imgs00
        self.imgs10 = imgs10
        self.imgs11 = imgs11
        self.boxes00 = boxes00
        self.titles = titles
        self.inflate = inflate

        self.n_frames = len(self.imgs00) * self.inflate

        # configure args
        self.txt_kw = dict(x=0, y=1.015, s="", ha="left", fontweight='bold')
        self.label_kw = dict(weight='bold', fontsize=15, labelpad=3)
        self.title_kw = dict(weight='bold', fontsize=18, loc='left')

        # create figure & axes
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        gs = axes[0, 0].get_gridspec()
        for ax in axes.flat:
            ax.remove()
        del ax

        ax00 = fig.add_subplot(gs[0, :2])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        self.fig, self.ax00, self.ax10, self.ax11 = fig, ax00, ax10, ax11

        # create images ######################################################
        self.imshow_kw = dict(aspect='auto', animated=True)

        # img00
        im = ax00.imshow(self.imgs00[0], **imshow_kw00)
        self.ims00 = [im]
        _ticks(xticks=t, yticks=freqs, ax=ax00)
        ax00.set_xlabel("time [sec]", **self.label_kw)
        ax00.set_ylabel("frequency [Hz]", **self.label_kw)
        ax00.set_xlim(0, len(t) - 1)
        ax00.set_ylim(len(freqs) - 1, 0)

        self.txt00 = ax00.text(transform=ax00.transAxes, **self.txt_kw,
                               fontsize=17)

        # boxes00
        box_data = self.unpack_box(self.boxes00[0])

        ckw = dict(color='tab:red', linewidth=2)
        lines = []
        for x_y in box_data:
            line = ax00.plot(*x_y, **ckw)[0]
            lines.append(line)
        self.lines00 = lines

        # img10
        im = ax10.imshow(self.imgs10[0], **imshow_kw10, vmin=0, vmax=mx)
        self.ims10 = [im]
        ax10.set_xticks([]); ax10.set_yticks([])

        self.txt10 = ax10.text(transform=ax10.transAxes, fontsize=17,
                               x=.63, y=-.08, fontweight='bold',
                               ha="left", s="JTFS coefficient & wavelet")

        # img11
        im = ax11.imshow(self.imgs11[0], **imshow_kw11)
        self.ims11 = [im]
        ax11.set_xticks([]); ax11.set_yticks([])

        # finalize #######################################################
        wx = .07
        wy = .03
        fig.subplots_adjust(bottom=wy*1.5, top=1-wy, left=wx, right=1-wx/4,
                            wspace=.1, hspace=.14)
        animation.TimedAnimation.__init__(self, fig, blit=True)

    @staticmethod
    def unpack_box(bbox):
        cy, cx = bbox[0]
        wy, wx = bbox[1]
        x0, x1, y0, y1 = [loc for loc in
                          (cx - wx, cx + wx, cy - wy, cy + wy)]
        M = 100
        x0_y01 = np.ones(M) * x0, np.linspace(y0, y1, M)
        x1_y01 = np.ones(M) * x1, np.linspace(y0, y1, M)
        x01_y0 = np.linspace(x0, x1, M), np.ones(M) * y0
        x01_y1 = np.linspace(x0, x1, M), np.ones(M) * y1

        return x0_y01, x1_y01, x01_y0, x01_y1

    def _draw_frame(self, frame_idx):
        if frame_idx % self.inflate != 0:
            return

        idx = frame_idx // self.inflate
        self.ims00[0].set_array(self.imgs00[idx])
        self.txt00.set_text(self.titles[idx])
        self.ims11[0].set_array(self.imgs11[idx])

        self.ims10[0].set_array(self.imgs10[idx])

        mx = np.abs(self.imgs11[idx]).max()
        self.im11 = self.ax11.imshow(self.imgs11[idx], **self.imshow_kw,
                                     cmap='bwr', vmin=-mx, vmax=mx)

        box_data = self.unpack_box(self.boxes00[idx])
        for i, x_y in enumerate(box_data):
            self.lines00[i].set_data(*x_y)

        self._drawn_artists = [*self.ims00, *self.ims10, self.txt00,
                               *self.lines00, self.im11, self.txt10]

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass


# demonstrative ##############################################################
# visuals likelier for one-time use rather than filterbank/coeff introspection

def viz_spin_2d(pair_waves=None, pairs=None, preset=None, axis_labels=None,
                pair_labels=True, fps=30, savepath='spin2d.gif', is_time=None,
                anim_kw=None, verbose=True):
    """Visualize the complete 4D behavior of 2D (1D-separable) complex Morlet
    wavelets, with the time dimension unrolled.

    Also supports all JTFS pairs, and a general 2D complex input.

    Parameters
    ----------
    pair_waves : dict / None
        Wavelets/lowpasses to use to generate pairs, centered in time about
        `n=0` (index 0, DFT-centered). If not provided, will use defaults.
        Supported keys:

            - `'up'`: psi_f_up (frequential bandpass, spin up / analytic)
            - `'dn'`: psi_f_dn (frequential bandpass, spin down / anti-analytic)
            - `'psi_t'`: psi_t (temporal bandpass)
            - `'phi_t'`: phi_t (temporal lowpass)
            - `'phi_f'`: phi_f (frequential lowpass)

        Must provide all keys that are provided in `pairs`, except `phi_t_dn`
        which instead requires 'dn'.

        Supports a second input mode that assumes a completely specified input
        (2D, in time domain, centered): as long as the arrays are 2D, will
        assume this mode. This supports an additional key, `'phi_t_dn'`,
        that's otherwise built from 1D inputs. Example of such an input is in
        `wavespin.visuals.animated.make_jtfs_pair()`.

    pairs : None / tuple[str['up', 'dn', 'phi_t', 'phi_f', 'phi', 'phi_t_dn']]
        Pairs to visualize. Number of specified pairs must be 1, 2, or 6.
        If not `None`, then `pair_waves` must be `None`.
        Defaults to either what's in `preset` or what's in `pair_waves`.

    preset : None / int[0, 1, 2]
        Animation preset to use:

            - `0`: pairs=('up',)
            - `1`: pairs=('up', 'dn')
            - `2`: pairs=('up', 'dn', 'phi_t', 'phi_f', 'phi', 'phi_t_dn')

        If wavelets/lowpasses aren't passed in, will generate them
        (if `pairs_preset` is not None).

    axis_labels : None / bool
        If False, will omit axis tick labels, axis labels, and axis planes.
        Defaults to True if `len(pair_waves) == 1`.

    pair_labels : bool (default True)
        If True, will title plot with name of pair being plotted, with LaTeX.

    fps : int
        Frames per second of the animation. Note, for .gif, `>30` may *lower* FPS
        (don't know why).

    savepath : str
        Path to save the animation to, as .gif or .mp4.
        .mp4 requires FFMPEG.

    is_time : None / bool
        Whether the provided `pair_waves` are in time-domain. Defaults to `False`.

        Method internally centers about `N//2` (visual center); for `False`,
        applies

                `p = ifft(ifft(p, axis=0), axis=1)`
                `p = ifftshift(ifftshift(p, axes=0), axes=1)`

        `True` is required if `pair_waves` is specified by
        `wavespin.visuals.animated.make_jtfs_pair()`.

    anim_kw : dict / None
        Passed to animator `wavespin.visuals.animated.SpinAnimator2D`.

            - `'linewidth'`: passed to `plt.plot()`.

    verbose : bool (default True)
        Whether to print where the animation is saved.
    """
    # handle arguments #######################################################
    pair_presets = {0: ('up',),
                    1: ('up', 'dn'),
                    2: ('up', 'phi_f', 'dn', 'phi_t', 'phi', 'phi_t_dn')}

    # handle `preset`
    if preset is None:
        preset = 0
    elif preset not in pair_presets:  # no-cov
        raise ValueError("`preset` %s is unsupported, must be one of %s" % (
            preset, list(pair_presets)))

    # handle `is_time`
    if is_time is not None and pair_waves is None:  # no-cov
        warnings.warn("`is_time` does nothing if `pair_waves` is `None`.")

    # handle `pairs`
    if pairs is None:
        if pair_waves is not None:
            pairs = list(pair_waves)
        else:
            pairs = pair_presets[preset]
    else:  # no-cov
        if pair_waves is not None:
            raise ValueError("Can't provide both `pair_waves` and `pairs`.")
        elif isinstance(pairs, str):
            pairs = (pairs,)
        elif not (isinstance(pairs, tuple) and isinstance(pairs[0], str)):
            raise TypeError("`pairs` must be None, str, or tuple of str, got "
                            "%s" % type(pairs))

    # handle `pair_waves`
    if pair_waves is None:
        N, xi0, sigma0 = 128, 4., 1.35
        N_time = int(N * (fps / 30))
        pair_waves = {pair: make_jtfs_pair(N, pair, xi0, sigma0, N_time)
                      for pair in pairs}
    else:
        pair_waves = pair_waves.copy()  # don't affect external keys
        passed_pairs = list(pair_waves)
        for pair in passed_pairs:
            if pair not in pairs:  # no-cov
                del pair_waves[pair]

        if not is_time:
            # convert to time, center
            for pair in pair_waves:
                pair_waves[pair] = ifftshift(ifftshift(
                    ifft(ifft(pair_waves[pair], axis=0), axis=-1),
                    axes=0), axes=-1)
                if pair == 'phi':
                    pair_waves['phi'] = pair_waves['phi'].real

    # handle `axis_labels`
    if len(pair_waves) > 1 and axis_labels:  # no-cov
        raise ValueError("`axis_labels=True` is only supported for "
                         "`len(pair_waves) == 1`")
    elif axis_labels is None and len(pair_waves) == 1:  # no-cov
        axis_labels = True

    # visualize ##############################################################
    savepath, writer = _handle_animation_savepath(savepath)

    # animate & save
    ani = SpinAnimator2D(pair_waves, axis_labels, pair_labels, anim_kw=anim_kw)
    ani.save(savepath, fps=fps, savefig_kwargs=dict(pad_inches=0), writer=writer)
    plt.close()

    if verbose:
        print("Saved animation to", savepath)


def viz_spin_1d(psi_f=None, fps=30, savepath='spin1d.gif', end_pause=None,
                w=None, h=None, is_time=None, anim_kw=None, verbose=True):
    """Visualize the complete 3D behavior of 1D complex Morlet wavelets.

    Also supports a general 1D complex input.

    Parameters
    ----------
    psi_f : tensor / None
        1D complex Morlet wavelet. If None, will make a default.

    fps : int
        Frames per second of the animation.

    savepath : str
        Path to save the animation to, as .gif or .mp4.
        .mp4 requires FFMPEG.

    end_pause : int / None
        Number of frames to insert at the end of animation that duplicate the
        last frame, effectively "pausing" the animation at finish.
        Defaults to `fps`, i.e. one second.

    w, h : float / None
        Animation width and height scaling factors.
        Act via `subplots(, figsize=(width*w, height*h))`.

        Defaults motivated same as `subplots_adjust_kw`.

    is_time : None / bool
        Whether the provided `psi_f` is in time-domain. Defaults to `False`.

        Method internally centers about `N//2` (visual center); for `False`,
        applies `p = ifftshift(ifft(p))`.

    anim_kw : dict / None
        Passed to animator `wavespin.visuals.animated.SpinAnimator1D`.

          - `'subplots_adjust'`: passed to `fig.subplots_adjust()`.
            Defaults strive for a `plt.tight()` layout, with presets for
            `len(psi_f)=1` and `=2`.

    verbose : bool (default True)
        Whether to print where the animation is saved.
    """
    # handle arguments #######################################################
    if is_time is not None and psi_f is None:  # no-cov
        warnings.warn("`is_time` does nothing if `psi_f` is `None`.")
        is_time = False

    if end_pause is None:
        end_pause = fps
    if psi_f is None:  # no-cov
        N, xi0, sigma0 = 128, 4., 1.35
        psi_f = morlet_1d(N, xi=xi0/N, sigma=sigma0/N).squeeze()
    if not isinstance(psi_f, (list, tuple)):
        psi_f = [psi_f]
    if is_time:
        psi_t = psi_f
    else:
        psi_t = [ifftshift(ifft(p)) for p in psi_f]

    # visualize ##############################################################
    savepath, writer = _handle_animation_savepath(savepath)

    ani = SpinAnimator1D(psi_t, end_pause=end_pause, anim_kw=anim_kw)
    ani.save(savepath, fps=fps, savefig_kwargs=dict(pad_inches=0), writer=writer)
    plt.close()

    if verbose:  # no-cov
        print("Saved animation to", savepath)


class SpinAnimator2D(animation.TimedAnimation):
    def __init__(self, pair_waves, axis_labels=False, pair_labels=True,
                 anim_kw=None):
        assert isinstance(pair_waves, dict), type(pair_waves)
        assert len(pair_waves) in (1, 2, 6), len(pair_waves)
        assert not (len(pair_waves) > 1 and axis_labels)

        self.pair_waves = pair_waves
        self.axis_labels = axis_labels
        self.pair_labels = pair_labels

        defaults = dict(linewidth=2, time_spin=False)
        self.anim_kw = fill_default_args(anim_kw, defaults,
                                         check_against_defaults=True)

        self.ref = list(pair_waves.values())[0]
        self.plot_frames = list(pair_waves.values())
        self.n_pairs = len(pair_waves)

        # make titles
        titles = {'up':       r"$\psi(t) \psi(+\lambda) \uparrow$",
                  'phi_f':    r"$\psi(t) \phi(\lambda)$",
                  'dn':       r"$\psi(t) \psi(-\lambda) \downarrow$",
                  'phi_t':    r"$\phi(t) \psi(+\lambda)$",
                  'phi':      r"$\phi(t) \phi(\lambda)$",
                  'phi_t_dn': r"$\phi(t) \psi(-\lambda)$"}
        if self.anim_kw['time_spin']:  # no-cov
            titles['up'] = r"$\psi(+t) \psi(\lambda) \uparrow$"
            titles['dn'] = r"$\psi(-t) \psi(\lambda) \downarrow$"
        self.titles = [titles[pair] for pair in self.pair_waves]

        # get quantities from reference
        self.n_f, self.N = self.ref.shape
        self.n_frames = self.N
        self.z = np.arange(self.n_f) / self.n_f
        self.T_all = np.arange(self.N) / self.N

        # get axis limits
        mx = max(np.abs(p).max() for p in list(pair_waves.values()))
        z_max  = self.z.max()

        # configure label args
        fontsizes = {1: (26, 24), 2: (29, 27), 6: (26, 24)}[self.n_pairs]
        self.title_kw = dict(y=.83, weight='bold', fontsize=fontsizes[0])
        self.txt_kw = dict(x=3*mx, y=25*mx, z=-2*z_max, s="", ha="left")
        self.label_kw = dict(weight='bold', fontsize=fontsizes[1])

        # create figure & axes
        fig = plt.figure(figsize=(16, 8))
        axes = []
        subplot_args = {1: [(1, 1, 1)],
                        2: [(1, 2, i) for i in range(1, 2+1)],
                        6: [(2, 3, i) for i in range(1, 6+1)]}[self.n_pairs]
        for arg in subplot_args:
            axes.append(fig.add_subplot(*arg, projection='3d'))

        # initialize plots ###################################################
        def init_plot(i):
            ax = axes[i]
            # plot ####
            xc = self.plot_frames[i][:, 0]
            line = ax.plot(xc.real, xc.imag, label='parametric curve',
                           linewidth=self.anim_kw['linewidth'])[0]
            line.set_data(xc.real, xc.imag)
            line.set_3d_properties(self.z)
            setattr(self, f'lines{i}', [line])

            # axes styling ####
            xlims = (-mx, mx)
            ylims = (-mx, mx)
            zlims = (0, z_max)

            ax.set_xlim3d(xlims)
            ax.set_ylim3d(ylims)
            ax.set_zlim3d(zlims)
            if self.pair_labels:
                ax.set_title(self.titles[i], **self.title_kw)

            if not axis_labels:
                # no border, panes, spines; 0 margin ####
                for anm in ('x', 'y', 'z'):
                    getattr(ax, f'set_{anm}ticks')([])
                    getattr(ax, f'{anm}axis').set_pane_color((1, 1, 1, 0))
                    getattr(ax, f'{anm}axis').line.set_color((1, 1, 1, 0))
                    getattr(ax, f'set_{anm}margin')(0)
                    ax.patch.set_alpha(0.)
            else:
                ax.set_xlabel("real", **self.label_kw)
                ax.set_ylabel("imag", **self.label_kw)
                ax.set_zlabel(r"$\lambda$", **self.label_kw)
                setattr(self, f'txt{i}',
                        ax.text(transform=ax.transAxes, **self.txt_kw,
                                fontsize=18))

        for i in range(len(axes)):
            init_plot(i)

        # finalize #######################################################
        configs = {
            1: dict(top=1,   bottom=0,   right=1.1, left=-.1),
            2: dict(top=1.3, bottom=-.4, right=1.1, left=-.1),
            6: dict(top=1.1, bottom=-.2, right=1.3, left=-.3, hspace=-.4),
        }[self.n_pairs]
        wspace = -.75 if self.n_pairs == 6 else -.3

        fig.subplots_adjust(**configs, wspace=wspace)
        animation.TimedAnimation.__init__(self, fig, blit=True)

    def _draw_frame(self, frame_idx):
        # plot ###############################################################
        lines, txts = [], []
        for i in range(self.n_pairs):
            xc = self.plot_frames[i][:, frame_idx]
            line = getattr(self, f'lines{i}')
            line[0].set_data(xc.real, xc.imag)
            line[0].set_3d_properties(self.z)
            lines.append(*line)

            if self.axis_labels:
                T_sec = self.T_all[frame_idx]
                txt = getattr(self, f'txt{i}')
                txt.set_text("t=%.3f" % T_sec)
                txts.append(txt)

        # finalize ###########################################################
        self._drawn_artists = [*lines, *txts]

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass


class SpinAnimator1D(animation.TimedAnimation):
    def __init__(self, plot_frames, end_pause=0, w=None, h=None, anim_kw=None):
        self.plot_frames = plot_frames
        self.end_pause = end_pause
        n_plots = len(plot_frames)
        self.n_plots = n_plots
        ref = plot_frames[0]

        # handle `anim_kw`
        sakw_defaults = {
            1: dict(top=1, bottom=0, right=.975, left=.075, hspace=.1,
                    wspace=.1),
            2: dict(top=1, bottom=0, right=.975, left=.075, hspace=-.7,
                    wspace=.1),
        }[n_plots]
        defaults = dict(subplots_adjust=sakw_defaults)
        self.anim_kw = fill_default_args(anim_kw, defaults,
                                         check_against_defaults=True)

        # handle `w, h`
        if w is None:
            w = {1: 1, 2: 1}[n_plots]
        if h is None:
            h = {1: 1, 2: 1.1}[n_plots]

        # get quantities from reference
        self.n_frames = len(ref)
        self.n_frames_total = self.n_frames + self.end_pause
        self.z = np.arange(len(ref)) / len(ref)
        self.T_all = np.arange(self.n_frames) / self.n_frames

        # get axis limits
        zmax = self.z.max()
        mx = max(np.abs(p).max() for p in plot_frames)

        # configure labels
        self.txt_kw = dict(x=-.25*mx, y=1.03*mx, s="", ha="left")
        self.label_kw = dict(weight='bold', fontsize=18)

        # create figure & axes
        width, height = 13/1.02, 16/1.1
        width *= w
        height *= h
        n_rows_base = 6
        fig, axes = plt.subplots(n_rows_base*n_plots, 7, figsize=(width, height))

        # gridspec object allows treating multiple axes as one
        gs = axes[0, 0].get_gridspec()
        # remove existing axes
        for ax in axes.flat:
            ax.remove()

        def init_plot(i):
            # create two axes with greater height and width ratio for the 2D
            # plot, since 3D is mainly padding
            inc = i * n_rows_base  # index increment
            ax0 = fig.add_subplot(gs[(inc + 2):(inc + 4), :3])
            ax1 = fig.add_subplot(gs[(inc + 0):(inc + n_rows_base), 3:],
                                  projection='3d')

            # initialize plots ###############################################
            plot_frames = self.plot_frames[i]
            xc = plot_frames[0]
            color = np.array([[102, 0, 204]])/256
            dot0 = ax0.scatter(xc.real, xc.imag, c=color)
            setattr(self, f'dots{i}0', [dot0])

            xcl = plot_frames[:1]
            line1 = ax1.plot(xcl.real, xcl.imag, label='parametric curve')[0]
            line1.set_data(xcl.real, xcl.imag)
            line1.set_3d_properties(0.)

            dot1 = ax1.scatter(xc.real, xc.imag, 0., c=color)
            dot1.set_3d_properties(0., 'z')
            setattr(self, f'lines{i}1', [line1, dot1])

            # styling ####
            # limits
            ax0.set_xlim(-mx, mx)
            ax0.set_ylim(-mx, mx)

            ax1.set_xlim(-mx, mx)
            ax1.set_ylim(-mx, mx)
            ax1.set_zlim(0, zmax)

            # labels
            ax0.set_xlabel("real", **self.label_kw)
            ax0.set_ylabel("imag", **self.label_kw)
            setattr(self, f'txt{i}', ax0.text(**self.txt_kw, fontsize=18))

            ax1.set_xlabel("real", **self.label_kw)
            ax1.set_ylabel("imag", **self.label_kw)
            ax1.set_zlabel(r"$t$", **self.label_kw)

        for i in range(self.n_plots):
            init_plot(i)

        # finalize #######################################################
        fig.subplots_adjust(**self.anim_kw['subplots_adjust'])
        animation.TimedAnimation.__init__(self, fig, blit=True)

    def _draw_frame(self, frame_idx):
        if frame_idx < self.n_frames:
            self._drawn_artists = []
            for i in range(self.n_plots):
                plot_frames = self.plot_frames[i]
                # plot #######################################################
                # dot
                name = f'dots{i}0'
                dotsi0 = getattr(self, name)

                xc = plot_frames[frame_idx]
                xc = np.array([[xc.real, xc.imag]])
                dotsi0[0].set_offsets(xc)

                setattr(self, name, dotsi0)
                self._drawn_artists.append(dotsi0)

                # spiral
                name = f'lines{i}1'
                linesi1 = getattr(self, name)

                xcl = plot_frames[:frame_idx+1]
                linesi1[0].set_data(xcl.real, xcl.imag)
                linesi1[0].set_3d_properties(self.z[:frame_idx+1])

                linesi1[1].set_offsets(xc)
                linesi1[1].set_3d_properties(frame_idx/self.n_frames, 'z')

                setattr(self, name, linesi1)
                self._drawn_artists.append(linesi1)

                # text
                name = f'txt{i}'
                txti = getattr(self, name)

                T_sec = self.T_all[frame_idx]
                txti.set_text("t=%.3f" % T_sec)

                setattr(self, name, txti)
                self._drawn_artists.append(txti)
        else:
            # repeat the last frame
            pass

    def new_frame_seq(self):
        return iter(range(self.n_frames_total))

    def _init_draw(self):
        pass


# utils ######################################################################
def make_gif(loaddir, savepath, duration=250, start_end_pause=0, ext='.png',
             delimiter='', overwrite=False, delete_images=False, HD=None,
             verbose=False):
    """Makes gif out of images in `loaddir` directory with `ext` extension,
    and saves to `savepath`.

    Parameters
    ----------
    loaddir : str
        Path to directory from which to fetch images to use as GIF frames.

    savepath : path
        Save path, must end with '.gif'.

    duration : int
        Interval between each GIF frame, in milliseconds.

    start_end_pause : int / tuple[int]
        Number of times to repeat the start and end frames, which multiplies
        their `duration`; if tuple, first element is for start, second for end.

    ext : str
        Images filename extension.

    delimiter : str
        Substring common to all image filenames, e.g. `'img'` for `'img0.png'`,
        `'img1.png'`, ... .

    overwrite : bool (default False)
        If True and file at `savepath` exists, will overwrite it.

    HD : bool / int[0, 1, 2] / None
            - `1`: use `imageio`.
            - `2`: use `ImageMagick`.
            - `0`: use `PIL.Image`.

        `2` may offer highest quality, followed by `1` then `0`. Will default
        to highest option that's installed, if compatible with `start_end_pause`.
        `True` forces picking between `2` and `1`.

        `2` renumbers images so their alphabetic sorting matches their
        alphanumeric sorting, e.g. `im5.png` -> `im005.png`.

    delete_images : bool (default False)
        Whether to delete the images used to make the GIF.

    verbose : bool (default False)
        Whether to print to console the location of save file upon success.
    """
    # handle GIF writer ######################################################
    def try_ImageMagick(do_error=False):
        import subprocess
        response = subprocess.getoutput("magick -version")
        if 'ImageMagick' not in response:  # no-cov
            if do_error:
                raise ImportError("`HD=2` requires ImageMagick installed.\n"
                                  "https://imagemagick.org/script/download.php")
            return False
        return True

    def try_imageio(do_error=False):  # no-cov
        try:
            import imageio
            return True
        except ImportError as e:
            if do_error:
                print("`HD=1` requires `imageio` installed")
                raise e
            return False

    def try_PIL(do_error=False):  # no-cov
        try:
            from PIL import Image
            return True
        except ImportError as e:
            if do_error:
                print("`HD=False` requires `PIL` installed.")
                raise e
            return False

    if HD is None:
        # Default to highest or raise error if none are available
        got = {'ImageMagick': try_ImageMagick(),
               'imageio': try_imageio(),
               'PIL': try_PIL()}
        if not any(got.values()):  # no-cov
            raise ImportError("`make_gif` requires `ImageMagick`, `imageio`, "
                              "or `PIL` installed.")
        elif got['ImageMagick'] and not start_end_pause:  # no-cov
            HD = 2
        elif got['imageio']:  # no-cov
            HD = 1
        elif got['PIL']:  # no-cov
            HD = 0
        else:  # no-cov
            raise ValueError("Couldn't pick a default `HD`. See docs.")

    elif HD is True:
        # Default to highest or raise error if none are available
        if try_ImageMagick():  # no-cov
            HD = 2
        elif try_imageio():  # no-cov
            HD = 1
        else:  # no-cov
            raise ImportError("`HD=True` requires `ImageMagick` or `imageio` "
                              "installed.")
    if HD == 2:  # no-cov
        try_ImageMagick(do_error=True)
    elif HD == 1:  # no-cov
        try_imageio(do_error=True)
        import imageio
    elif not HD:  # no-cov
        try_PIL(do_error=True)
        from PIL import Image

    # handle `start_end_pause`
    if start_end_pause:
        if isinstance(start_end_pause, (list, tuple)):
            reps0, reps1 = start_end_pause
        else:
            reps0 = reps1 = start_end_pause

    # fetch frames ###########################################################
    loaddir = os.path.abspath(loaddir)
    names = [n for n in os.listdir(loaddir)
             if (n.startswith(delimiter) and n.endswith(ext))]
    names = sorted(names, key=lambda p: int(
        ''.join(s for s in p.split(os.sep)[-1] if s.isdigit())))
    paths = [os.path.join(loaddir, n) for n in names]

    if HD == 2:
        new_paths = _rename_to_sort_alphabetically(paths, delimiter, ext)
    else:
        # load frames
        frames = [(imageio.imread(p) if HD else Image.open(p))
                  for p in paths]

        # handle frame duplication to increase their duration
        if start_end_pause:
            for repeat_start in range(reps0):
                frames.insert(0, frames[0])
            for repeat_end in range(reps1):
                frames.append(frames[-1])

    # write GIF ##############################################################
    # handle `savepath`
    savepath = os.path.abspath(savepath)
    if os.path.isfile(savepath) and overwrite:
        # delete if exists
        os.unlink(savepath)

    # save
    if HD == 2:
        delay = duration // 10
        delim_regex = f"{loaddir}{os.sep}{delimiter}*{ext}"

        if start_end_pause:
            rep_nums = ( "0," * (reps0 + 1) + "1--1," +
                         ("-1," * reps1) ).rstrip(',')
            rep_cmd = f"-write mpr:imgs -delete 0--1 mpr:imgs[{rep_nums}]"
        else:
            rep_cmd = ""
        command = (
            f'magick -delay {delay} "{delim_regex}" {rep_cmd} "{savepath}"'
        )
        out = os.system(command)
        if out:
            raise RuntimeError(f"System command exited with status {out} "
                               f"for command `{command}`")
    elif HD:
        imageio.mimsave(savepath, frames, fps=1000/duration)
    else:
        frame_one = frames[0]
        frame_one.save(savepath, format="GIF", append_images=frames,
                       save_all=True, duration=duration, loop=0)

    # finishing ##############################################################
    if verbose:
        print("Saved gif to", savepath)

    if delete_images:
        delete_paths = (new_paths if HD == 2 else
                        paths)
        for p in delete_paths:
            os.unlink(p)
        if verbose:
            print("Deleted images used in making the GIF (%s total)" % len(paths))


def make_jtfs_pair(N, pair='up', xi0=4, sigma0=1.35, N_time=None):
    """Creates a 2D JTFS wavelet. Used in `wavespin.visuals`.

    `N_time` will return `(N, N_time)`-shaped output, in case we want different
    from the default `(N, N)`.
    """
    m_fn = lambda M: morlet_1d(M, xi=xi0/M, sigma=sigma0/M).squeeze()
    g_fn = lambda M: gauss_1d(M, sigma=sigma0/M).squeeze()

    morlf = m_fn(N)
    gausf = g_fn(N)
    if N_time is None:  # no-cov
        morlt = morlf
        gaust = gausf
    else:
        morlt = m_fn(N_time)
        gaust = g_fn(N_time)

    if pair in ('up', 'dn'):
        i0, i1 = 0, 0
    elif pair == 'phi_f':
        i0, i1 = 1, 0
    elif pair in ('phi_t', 'phi_t_dn'):
        i0, i1 = 0, 1
    elif pair == 'phi':
        i0, i1 = 1, 1
    else:  # no-cov
        supported = {'up', 'dn', 'phi_f', 'phi_t', 'phi', 'phi_t_dn'}
        raise ValueError("unknown pair %s; supported are %s" % (
            pair, '\n'.join(supported)))

    pf_f = (morlf, gausf)[i0]
    pt_f = (morlt, gaust)[i1]
    pf_f, pt_f = pf_f.copy(), pt_f.copy()
    if pair in ('dn', 'phi_t_dn'):
        # time reversal
        pf_f[1:] = pf_f[1:][::-1]
    pf, pt = [ifftshift(ifft(p)) for p in (pf_f, pt_f)]

    Psi = pf[:, None] * pt[None]
    return Psi


# helpers ####################################################################
# for when global scaling is already handled
def _imshow(*args, **kwargs):
    """`imshow` with `do_gscale=False` and `show=False`."""
    imshow(*args, **kwargs, do_gscale=False, show=False)


def _plot_box(*args, **kwargs):
    """`plot_box` with `do_gscale=False`."""
    plot_box(*args, **kwargs, do_gscale=False)


def _handle_gif_args(savedir, base_name, images_ext, save_images, overwrite,
                     show):
    """
    If all, `savedir`, `save_images`, and `show` are `None`, defaults to
    `show=True` and `save_images=False`. Else, defaults such that minimal
    actions are taken to fulfill whatever was specified (Ex 1: if `show=True`,
    then don't save images, since the user "already saw" the images. Ex 2: if
    `savedir` is specified, also won't save, since there's already a GIF).
    """
    do_gif = bool(savedir is not None)
    if save_images is None:
        if savedir is None:  # no-cov
            if show is None:
                save_images = False
                show = True
            else:
                save_images = bool(not show)
        else:
            save_images = False
    if show is None:  # no-cov
        show = bool(not save_images and not do_gif)

    if savedir is None and save_images:  # no-cov
        savedir = ''
    if savedir is not None:  # no-cov
        savedir = os.path.abspath(savedir)

    if not images_ext.startswith('.'):  # no-cov
        images_ext = '.' + images_ext

    if base_name.endswith('.gif'):  # no-cov
        base_name = base_name[:-4]

    if savedir is not None:
        savepath = os.path.join(savedir, base_name + '.gif')
        _check_savepath(savepath, overwrite)
    else:
        savepath = None
    return savedir, savepath, images_ext, base_name, save_images, show, do_gif


def _handle_animation_savepath(savepath):
    # handle `savepath`
    supported = ('.gif', '.mp4')
    if not any(savepath.endswith(ext) for ext in supported):  # no-cov
        savepath += supported[0]
    savepath = os.path.abspath(savepath)

    # set `writer`
    if (savepath.endswith('.gif') and
            animation.FFMpegWriter.isAvailable()):  # no-cov
        writer = 'ffmpeg'
    else:
        writer = None
    return savepath, writer


def _rename_to_sort_alphabetically(paths, delimiter, ext):
    """Rename image files so alphabetic and alphanumeric sorting match.
    Use true delimiter for this.
    """
    if len(paths) == 0:
        return []
    names = [Path(p).name for p in paths]

    delimiter_full = os.path.commonprefix(names)
    assert delimiter in delimiter_full, (delimiter, delimiter_full)
    # fetch max length
    delim_len = len(delimiter_full)
    strip_fn = lambda s: s[delim_len:-len(ext)]
    longest_num = max(len(strip_fn(nm)) for nm in names)

    # rename
    new_paths = []
    for p in paths:
        nm = Path(p).name
        num = ''.join(c for c in strip_fn(nm) if c.isdigit())

        # split into delimiter and non-delimiter so we don't replace in delimiter
        nondelim = nm[delim_len:]
        renamed = (delimiter_full +
                   nondelim.replace(num, f'%.{longest_num}d' % int(num)))
        new_p = p.replace(nm, renamed)

        new_paths.append(new_p)
        if new_p == p:
            continue
        elif os.path.isfile(new_p):
            os.remove(new_p)
        os.rename(p, new_p)
    return new_paths
