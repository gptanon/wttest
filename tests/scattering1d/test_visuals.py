# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Test that wavespin/visuals.py methods run without error."""
import pytest, os, warnings
import numpy as np
from copy import deepcopy

from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.toolkit import echirp, pack_coeffs_jtfs, top_spinned
from wavespin import visuals as v
from wavespin.scattering1d.filter_bank import morlet_1d
from utils import tempdir, SKIPS, FORCED_PYTEST

# backend to use for most tests
default_backend = 'numpy'
# set True to execute all test functions without pytest
run_without_pytest = 1
# precision to use for all but precision-sensitive tests
default_precision = 'single'
# set True to disable matplotlib plots
# (done automatically for CI via `conftest.py`, but `False` here takes precedence)
no_plots = 1
# set True to skip this file entirely
SKIP_ALL = SKIPS['visuals']
# same as `SKIP_ALL` but passed from command line
CMD_SKIP_ALL = bool(os.environ.get('CMD_SKIP_VISUALS', '0') == '1')

# disable plots for pytest unit testing
# restart kernel when changing `no_plots`
if no_plots:
    import matplotlib
    matplotlib.use('template')
    matplotlib.pyplot.ioff()

# set up reusable references in global scope
sc_tms, jtfss, sc_all = [], [], []
metas = []
xs = []
out_tms, out_jtfss, out_all = [], [], []


def make_reusables():
    # run after __main__ so test doesn't fail during collection
    # reusable scattering objects
    skip_all = bool(CMD_SKIP_ALL or SKIP_ALL)
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    N = 256
    kw0 = dict(shape=N, T=2**7, J=8, Q=8, frontend=default_backend,
               precision=default_precision)
    sc_tms.extend([Scattering1D(**kw0, out_type='array')])

    sfs = [('resample', 'resample'), ('exclude', 'resample'),
           ('recalibrate', 'recalibrate')]
    kw1 = dict(out_type='dict:array', out_3D=False, **kw0)
    jtfss.extend([
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[0]),
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[1]),
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[2]),
    ])
    sc_all.extend([*sc_tms, *jtfss])

    # reusable input
    xs.append(echirp(N))
    # reusable outputs
    out_tms.extend([sc(xs[0]) for sc in sc_tms])
    out_jtfss.extend([jtfs(xs[0]) for jtfs in jtfss])
    out_all.extend([*out_tms, *out_jtfss])
    # metas
    metas.extend([sc.meta() for sc in sc_all])

    return sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all


#### Tests ###################################################################

def test_filterbank_heatmap(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    for i, sc in enumerate(sc_all):
        frequential = bool(i > 0)
        v.filterbank_heatmap(sc, first_order=True, second_order=True,
                             frequential=frequential)


def test_filterbank_scattering(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    sc_all = G['sc_all']
    sc_tms = G['sc_tms']
    for sc in sc_all:
        v.filterbank_scattering(sc, second_order=1, lp_sum=1, zoom=0)
    for zoom in (4, -1):
        v.filterbank_scattering(sc_tms[0], second_order=1, lp_sum=1, zoom=zoom)


def test_filterbank_jtfs_1d(G):
    skip_all = bool(CMD_SKIP_ALL or SKIP_ALL)
    if skip_all:
        return None if run_without_pytest else pytest.skip()

    jtfss = G['jtfss']
    for jtfs in jtfss:
        v.filterbank_jtfs_1d(jtfs, lp_sum=1, zoom=0)
    for zoom in (4, -1):
        v.filterbank_jtfs_1d(jtfs, lp_sum=0, lp_phi=0, zoom=zoom)


def test_viz_jtfs_2d(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    jtfss = G['jtfss']
    out_jtfss = G['out_jtfss']

    # without save
    _ = v.viz_jtfs_2d(jtfss[1], Scx=out_jtfss[1], show=0,
                      plot_cfg={'filter_label': True, 'phi_t_loc': 'both',
                                'subplots_kw': dict(facecolor='k')})
    # for coverage, also a warning
    with warnings.catch_warnings(record=True) as ws:
        plot_cfg = {'phi_t_loc': 'both', 'phi_t_blank': True,
                    'filter_part': 'complex', 'filterbank_subsample': (4, 4)}
        _ = v.viz_jtfs_2d(jtfss[1], Scx=out_jtfss[1], show=0, equalize_pairs=1,
                          viz_spins=(0, 1), plot_cfg=plot_cfg)
        assert any('does nothing' in str(w.message) for w in ws), ws

    # with save
    base = 'viz_jtfs2d'
    fn = lambda savedir: v.viz_jtfs_2d(
        jtfss[1], Scx=out_jtfss[1], show=1, savename=os.path.join(savedir, base),
        plot_cfg={'filter_part': 'imag', 'filterbank_zoom': -1})
    # name changes internally
    _run_with_cleanup(fn, [base + '0.png', base + '1.png'])


def test_gif_jtfs_2d(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    out_jtfss, metas = G['out_jtfss'], G['metas']
    base_name = 'jtfs2d'
    savename = base_name + '.gif'
    fn = lambda savedir: v.gif_jtfs_2d(out_jtfss[1], metas[2],
                                       savedir=savedir, base_name=base_name)
    _run_with_cleanup(fn, savename)

    # `do_gif = False`
    # manually implement `_run_with_cleanup` per needing to pass `savedir=None`
    with tempdir() as savedir:
        cwd = os.getcwd()
        os.chdir(savedir)

        fn = lambda: v.gif_jtfs_2d(out_jtfss[1], metas[2], savedir=None,
                                   norms=(1, 2, 3, 4, 5))
        try:
            _ = fn()
        finally:
            # revert `cwd`
            os.chdir(cwd)


def test_gif_jtfs_3d(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    try:
        import plotly
    except ImportError:
        warnings.warn("Skipped `test_gif_jtfs_3d` since `plotly` not installed.")
        return

    jtfss, out_jtfss, metas = G['jtfss'], G['out_jtfss'], G['metas']
    packed = pack_coeffs_jtfs(out_jtfss[1], metas[2], structure=2,
                              sampling_psi_fr='exclude')

    base_name = 'jtfs3d'
    savename = base_name + '.gif'
    kw = dict(base_name=base_name, images_ext='.png', verbose=0)

    fn = lambda savedir: v.gif_jtfs_3d(packed, savedir=savedir, **kw)
    _run_with_cleanup_handle_exception(fn, savename)
    fn = lambda savedir: v.gif_jtfs_3d(out_jtfss[1], jtfss[1], angles='rotate',
                                       savedir=savedir, **kw)
    _run_with_cleanup_handle_exception(fn, savename)


def test_energy_profile_jtfs(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    out_jtfss = G['out_jtfss']
    for i, Scx in enumerate(out_jtfss):
      for flatten in (False, True):
        for pairs in (None, ('phi_t * psi_f',)):
          test_params = dict(flatten=flatten, pairs=pairs)
          try:
              _ = v.energy_profile_jtfs(Scx, metas[1 + i], **test_params)
          except Exception as e:
              test_params['i'] = i
              print('\n'.join(f'{k}={v}' for k, v in test_params.items()))
              raise e


def test_coeff_distance_jtfs(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    out_jtfss = G['out_jtfss']
    for i, Scx in enumerate(out_jtfss):
      for flatten in (False, True):
        for pairs in (None, ('phi_t * psi_f',)):
          test_params = dict(flatten=flatten, pairs=pairs)
          try:
              _ = v.coeff_distance_jtfs(Scx, Scx, metas[1 + i], **test_params)
          except Exception as e:
              test_params['i'] = i
              print('\n'.join(f'{k}={v}' for k, v in test_params.items()))
              raise e


def test_compare_distances_jtfs(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    out_jtfss = G['out_jtfss']
    Scx0, Scx1 = out_jtfss[0], deepcopy(out_jtfss[0])
    for pair in Scx1:
        Scx1[pair] += 1

    dists0 = v.coeff_distance_jtfs(Scx0, Scx1, metas[1])[1]
    dists1 = v.coeff_distance_jtfs(Scx1, Scx0, metas[1], plots=1)[1]
    _ = v.compare_distances_jtfs(dists0, dists1, plots=1)


def test_scalogram(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    sc_tm = G['sc_tms'][0]
    sc_tm.average = False
    sc_tm.out_type = 'list'
    _ = v.scalogram(np.random.randn(sc_tm.shape), sc_tm, show_x=1, fs=1)


def test_viz_spin_1d(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    # recreate method's `psi_f` but with smaller `N`
    N, xi0, sigma0 = 16, 4., 1.35
    psi_f = morlet_1d(N, xi=xi0/N, sigma=sigma0/N).squeeze()

    savename = 'spin_1d.mp4'
    fn = lambda savedir: v.viz_spin_1d(psi_f, verbose=0,
                                       savepath=os.path.join(savedir, savename))
    _run_with_cleanup_handle_exception(fn, savename)


def test_viz_spin_2d(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    def preset_to_pair_waves(preset):
        """Since default `N` takes too long."""
        pair_presets = {0: ('up',),
                        1: ('up', 'dn'),
                        2: ('up', 'phi_f', 'dn', 'phi_t', 'phi', 'phi_t_dn')}
        pairs = pair_presets[preset]
        N, xi0, sigma0 = 32, 4., 1.35
        pair_waves = {pair: v.make_jtfs_pair(N, pair, xi0, sigma0)
                      for pair in pairs}
        return pair_waves

    savename = 'spin_2d.mp4'
    # manually implement presets; axis_labels=0
    pair_waves = preset_to_pair_waves(preset=2)
    fn = lambda savedir: v.viz_spin_2d(pair_waves, axis_labels=0, verbose=0,
                                       savepath=os.path.join(savedir, savename))
    _run_with_cleanup_handle_exception(fn, savename)
    # axis_labels=1
    pair_waves = preset_to_pair_waves(preset=0)
    fn = lambda savedir: v.viz_spin_2d(pair_waves, axis_labels=1, verbose=0,
                                       savepath=os.path.join(savedir, savename))
    _run_with_cleanup_handle_exception(fn, savename)

    # no preset
    data = np.arange(6 * 6).reshape(6, 6)
    pair_waves = {'dn': data, 'phi_t': data[::-1]}
    fn = lambda savedir: v.viz_spin_2d(
        pair_waves, savepath=os.path.join(savedir, savename), verbose=1)
    _run_with_cleanup_handle_exception(fn, savename)


def test_viz_top_fdts(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    _jtfs = G['jtfss'][0]
    x = G['xs'][0]

    # need out_3D to test agreement
    jtfs = TimeFrequencyScattering1D(
        **{k: getattr(_jtfs, k) for k in
           "shape J Q out_type sampling_filters_fr T F aligned".split()},
        frontend=default_backend, out_3D=True, average_fr=True)
    Scx = jtfs(x)

    # `viz_top_fdts`; maximize coverage with 'energy-max'
    savename = 'top_fdts.mp4'
    fn = lambda savedir: v.viz_top_fdts(jtfs, x, top_k=3, render='mp4',
                                        measure='energy-max',
                                        savepath=os.path.join(savedir, savename))
    _, data = _run_with_cleanup_handle_exception(fn, savename)

    # `top_spinned`
    # pretend it's out_3D=True
    jmeta = jtfs.meta()
    Scx = deepcopy(Scx)
    oup, odn, *_ = pack_coeffs_jtfs(Scx, jmeta, structure=5,
                                    sampling_psi_fr=jtfs.scf.sampling_psi_fr,
                                    out_3D=jtfs.out_3D)
    info = top_spinned(Scx, jmeta, top_k=3, verbose=1)

    # assert agreement
    n_viz_top_fdts = data['n'][0]
    n_top_spinned = info[0][0]
    assert n_viz_top_fdts == n_top_spinned, (n_viz_top_fdts, n_top_spinned)

    # now 'gif' and `out_3D=False` for coverage
    jtfs = G['jtfss'][0]
    savename = 'top_fdts.gif'
    fn = lambda savedir: v.viz_top_fdts(jtfs, x, top_k=3, render='gif',
                                        savepath=os.path.join(savedir, savename))
    _, data = _run_with_cleanup_handle_exception(fn, savename)


def test_misc(G):
    if CMD_SKIP_ALL or SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    _ = v.plot([1, 2], xticks=[0, 1], yticks=[0, 1], show=0)
    _ = v.primitives._colorize_complex(np.array([[1 + 1j]]))
    _ = v.hist(np.random.randn(24), stats=1, show=1)
    _ = v.plotscat([1, 2])

    xc = np.array([1 + 2j, 1 + 3j])
    _ = v.scat(xc, complex=1, ticks=0)
    _ = v.plot(x=None, y=xc, complex=1)

# helpers ####################################################################
def _run_with_cleanup(fn, savename):
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    if not isinstance(savename, list):
        savename = [savename]

    out = None
    with tempdir() as savedir:
        try:
            out = fn(savedir)
            # assert file was created
            for nm in savename:
                path = os.path.join(savedir, nm)
                assert os.path.isfile(path), path
                os.unlink(path)
        finally:
            # clean up images, if any were made
            for nm in savename:
                paths = [os.path.join(savedir, n) for n in os.listdir(savedir)
                         if (n.startswith(nm) and n.endswith('.png'))]
                for p in paths:
                    os.unlink(p)
    return out

def _run_with_cleanup_handle_exception(fn, savename):
    out = None
    try:
        out = _run_with_cleanup(fn, savename)
    except Exception as e:
        if 'ffmpeg' not in str(e):
            # automated testing has some issues with this
            raise e
        warnings.warn("Ignored error:\n%s" % str(e))
    return out

# create testing objects #####################################################
if not SKIP_ALL:
    if run_without_pytest and not FORCED_PYTEST:
        sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all = (
            make_reusables())
        G = dict(sc_tms=sc_tms, jtfss=jtfss, sc_all=sc_all, metas=metas,
                 xs=xs, out_tms=out_tms, out_jtfss=out_jtfss, out_all=out_all)
    else:
        mr = [False]
        @pytest.fixture(scope='module')
        def G():
            if not mr[0]:
                sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all = (
                    make_reusables())
                mr[0] = True
            return dict(sc_tms=sc_tms, jtfss=jtfss, sc_all=sc_all, metas=metas,
                        xs=xs, out_tms=out_tms, out_jtfss=out_jtfss,
                        out_all=out_all)
else:
    @pytest.fixture(scope='module')
    def G():
        pass


# run tests ##################################################################
if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_filterbank_heatmap(G)
        test_filterbank_scattering(G)
        test_filterbank_jtfs_1d(G)
        test_viz_jtfs_2d(G)
        test_gif_jtfs_2d(G)
        test_gif_jtfs_3d(G)
        test_energy_profile_jtfs(G)
        test_coeff_distance_jtfs(G)
        test_compare_distances_jtfs(G)
        test_scalogram(G)
        test_viz_spin_1d(G)
        test_viz_spin_2d(G)
        test_viz_top_fdts(G)
        test_misc(G)
    else:
        pytest.main([__file__, "-s"])
