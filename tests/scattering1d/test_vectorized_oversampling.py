# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""`vectorized`- and `oversampling`-related tests."""
import os
import pytest
import numpy as np

from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.utils.gen_utils import npy, backend_has_gpu
from utils import SKIPS, FORCED_PYTEST, run_meta_tests_jtfs

# backends to test
backends = ('numpy', 'torch', 'tensorflow', 'jax')[1]
# precision to use for all but precision-sensitive tests
default_precision = 'single'
# set True to execute all test functions without pytest
run_without_pytest = 1
# skip flags
SKIP_LONG = SKIPS['long_in_jtfs']
CMD_SKIP_LONG = bool(os.environ.get('CMD_SKIP_LONG_JTFS', '0') == '1')

if isinstance(backends, str):
    backends = (backends,)


def test_scattering():
    """Test that for any given `oversampling`, outputs of `vectorized=True`
    and `vectorized=False` agree, for `Scattering1D`.
    """
    N = 497
    x = np.random.randn(N)
    base_kw = dict(shape=N, Q=8)

    for backend in backends:
        ckw = dict(**base_kw, frontend=backend)
        sc0 = Scattering1D(**ckw, vectorized=False)
        sc1 = Scattering1D(**ckw, vectorized=True)

        for try_gpu in (False, True):
            if try_gpu:
                if backend_has_gpu(backend):
                    for sc in (sc0, sc1):
                        sc.gpu()
                else:
                    continue

            for oversampling in (0, 1, 5):
                for sc in (sc0, sc1):
                    sc.update(oversampling=oversampling)

                o0 = npy(sc0(x))
                o1 = npy(sc1(x))
                assert np.allclose(o0, o1)

                # assert equal metas (should be independent of vectorization)
                # meta is independent of device
                if not try_gpu:
                    meta0, meta1 = [sc.meta() for sc in (sc0, sc1)]
                    _assert_equal_metas(meta0, meta1)


def test_jtfs():
    """`test_scattering()` but for JTFS, including `vectorized_fr` and
    `oversampling_fr`.
    """
    skip_long = bool(CMD_SKIP_LONG or SKIP_LONG)

    N = 497
    np.random.seed(0)
    x = np.random.randn(N)

    base_kw = dict(shape=N, Q=8, max_pad_factor_fr=1, average=True,
                   # `-2` to test skipping over a path
                   paths_exclude={'n2': -2, 'n1_fr': -2},
                   out_type='dict:list', precision='single')
    vec_kw = [dict(vectorized=False, vectorized_fr=False),
              dict(vectorized=True,  vectorized_fr=True),
              dict(vectorized=True,  vectorized_fr=2)]

    oversamplings    = (0, 1, 5) if not skip_long else (1,)
    oversampling_frs = (0, 1, 3) if not skip_long else (1,)

    for backend in backends:
      reproduced_non_grouped_case = False
      for sampling_filters_fr in ('exclude', 'recalibrate'):
        for aligned in (False, True):
          if aligned and sampling_filters_fr == 'recalibrate':
              # `recal., recal.` unsupported
              sampling_filters_fr = ('recalibrate', 'resample')
          for out_3D in (False, True):
            for average_fr in (False, True):
                if out_3D and not average_fr:  # invalid config
                    continue
                # pack configs
                cfg_kw = dict(
                    frontend=backend, sampling_filters_fr=sampling_filters_fr,
                    aligned=aligned, out_3D=out_3D, average_fr=average_fr)
                ckw = dict(**base_kw, **cfg_kw)

                try:
                    # build objects
                    jtfs0 = TimeFrequencyScattering1D(**ckw, **vec_kw[0])
                    jtfs1 = TimeFrequencyScattering1D(**ckw, **vec_kw[1])
                    jtfs2 = TimeFrequencyScattering1D(**ckw, **vec_kw[2])

                    # handle device
                    for try_gpu in (False, True):
                      if try_gpu:
                          if backend_has_gpu(backend):
                              for jtfs in (jtfs0, jtfs1, jtfs2):
                                  jtfs.gpu()
                          else:
                              continue

                      # test oversamplings
                      for oversampling, oversampling_fr in zip(
                              oversamplings, oversampling_frs):
                          okw = dict(oversampling=oversampling,
                                     oversampling_fr=oversampling_fr)
                          for jtfs in (jtfs0, jtfs1, jtfs2):
                              jtfs.update(**okw)
                          ckw.update(okw)

                          # check if non-grouped case is reproduced anywhere
                          if not reproduced_non_grouped_case:
                              for jtfs in (jtfs0, jtfs1, jtfs2):
                                  DF = jtfs.compute_graph_fr['DF']
                                  if any(not DF[n2][
                                          'part2_grouped_by_n1_fr_subsample']
                                         for n2 in DF if isinstance(n2, int)):
                                      reproduced_non_grouped_case = True
                                      break

                          o0 = jtfs0(x)
                          o1 = jtfs1(x)
                          o2 = jtfs2(x)

                          _jtfs_run_asserts(jtfs0, jtfs1, jtfs2, o0, o1, o2,
                                            ckw, try_gpu)

                except Exception as e:
                    try:
                        cfg_kw.update(try_gpu=try_gpu)
                    except:
                        # means we failed at __init__
                        pass
                    kw_text = '\n'.join(f"  {k}={v}" for k, v in cfg_kw.items())
                    print("Failed with\n" + kw_text)
                    raise e

      # reproduce for each backend (should be backend-independent but play safe)
      assert reproduced_non_grouped_case


def _jtfs_run_asserts(jtfs0, jtfs1, jtfs2, o0, o1, o2, ckw, try_gpu):
    # test outputs ###########################################################
    o_npys = []
    for o in (o0, o1, o2):
        o_npys.append({})
        for pair in o0:
            o_npys[-1][pair] = np.concatenate([npy(c['coef']) for c in o[pair]],
                                              axis=-2)

    for pair in o0:
        assert np.allclose(o_npys[0][pair], o_npys[1][pair])
        assert np.allclose(o_npys[0][pair], o_npys[2][pair])

    # test meta ##############################################################
    # meta is independent of device
    if not try_gpu:
        # assert equal metas (should be independent of vectorization)
        jmeta0, jmeta1, jmeta2 = [jtfs.meta() for jtfs in (jtfs0, jtfs1, jtfs2)]
        _assert_equal_metas(jmeta0, jmeta1, jmeta2)

        run_meta_tests_jtfs(jtfs0, o0, jmeta0, ckw, extras=False)


def _assert_equal_metas(*metas):
    def _assert_equal(vs):
        if hasattr(vs[0], 'ndim'):
            if np.isnan(vs[0].max()):
                vs = [v.copy() for v in vs]
                for v in vs:
                    v[np.isnan(v)] = -2
            for i, v in enumerate(vs[1:]):
                assert np.all(vs[0] == v), i + 1
        else:
            for i, v in enumerate(vs[1:]):
                assert vs[0] == v, i + 1

    is_jtfs = bool('spin' in metas[0])
    if is_jtfs:
        for field in metas[0]:
            for pair in metas[0][field]:
                vs = [meta[field][pair] for meta in metas]
                _assert_equal(vs)
    else:
        for field in metas[0]:
            vs = [meta[field] for meta in metas]
            _assert_equal(vs)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_scattering()
        test_jtfs()
    else:
        pytest.main([__file__, "-s"])
