# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Tests that current implementation of `scattering1d` produces outputs identical
to the previous, unvectorized and slower version.
"""
import pytest
import numpy as np
import torch
from wavespin import Scattering1D
from wavespin.utils.gen_utils import npy
from wavespin.scattering1d.frontend import base_frontend
from wavespin.scattering1d.core.scattering1d import scattering1d

from utils import cant_import, scattering1d_legacy, FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1


def test_current_vs_legacy():
    """Coefficient values, metas, and gradients must agree."""
    # ensure it doesn't mess with other tests
    try:
        _test_current_vs_legacy()
    finally:
        base_frontend.scattering1d = scattering1d


def _test_current_vs_legacy():
    N = 475  # non-dyadic to better test unpad
    x = np.random.randn(N)
    params_common = dict(shape=N, Q=4, max_pad_factor=0, smart_paths=.01,
                         J=int(np.log2(N)) - 2)
    Ts = (N//4, 'global')

    for frontend in ('numpy', 'torch', 'tensorflow'):
      if cant_import(frontend):
          continue
      if frontend == 'torch' and torch.cuda.is_available():
          device = 'cuda'
      else:
          device = 'cpu'
      for T in Ts:
        for average in (True, False):
          if not average and T == Ts[-1]:
              continue  # repeats Ts[0]
          for vectorized in (0, 1, 2):
            for out_type in ('list', 'array'):
              if not average and out_type == 'array':
                  continue  # invalid
              out_list = bool(out_type == 'list')
              # set up scattering object and add legacy method
              def make_sc(vectorized):
                  sc = Scattering1D(
                      **params_common, T=T, average=average, frontend=frontend,
                      vectorized=vectorized, out_type=out_type)
                  if device == 'cuda':
                      sc.cuda()
                  return sc

              sc = make_sc(vectorized)

              if frontend == 'torch':
                  xt0, xt1 = [torch.from_numpy(x).cuda() for _ in range(2)]
                  xt0.requires_grad, xt1.requires_grad = True, True
              else:
                  xt0 = xt1 = x

              # reset method to base, scatter
              base_frontend.scattering1d = scattering1d
              o0 = sc.scattering(xt0)
              # set to legacy, scatter
              sc = make_sc(vectorized=0)
              base_frontend.scattering1d = scattering1d_legacy
              o1 = sc.scattering(xt1)

              # compare coefficient and meta values ########################
              assert len(o0) == len(o1), (len(o0), len(o1))

              # first, unpack once
              coef1s, meta1s_rev = [], []
              loss1 = 0
              for i1, c1 in enumerate(o1):
                  coef1 = c1['coef'] if out_list else c1
                  meta1_rev = (None if not out_list else
                               {k: v[::-1] for k, v in c1.items()
                                if k != 'coef'})
                  if frontend == 'torch':
                      loss1 += coef1.mean()
                  if device == 'cuda':
                      coef1 = npy(coef1)
                  coef1s.append(coef1)
                  meta1s_rev.append(meta1_rev)

              # run loops
              loss0 = 0
              for i0, c0 in enumerate(o0):
                  coef0 = c0['coef'] if out_list else c0
                  meta0 = (None if not out_list else
                           {k: v for k, v in c0.items()
                            if k != 'coef'})
                  if frontend == 'torch':
                      loss0 += coef0.mean()
                  if device == 'cuda':
                      coef0 = npy(coef0)

                  # will `break` if a match is found.
                  # if for-loop completes without `break`, executes `else`.
                  conds = []
                  for i1, (coef1, meta1_rev
                           ) in enumerate(zip(coef1s, meta1s_rev)):
                      try:
                          A = bool(coef0.shape == coef1.shape)
                          B = bool(meta0 == meta1_rev)
                          C = bool(np.allclose(coef0, coef1))
                          conds.append((A, B, C))
                          assert A and B and C
                          break
                      except:
                          pass
                  else:
                      print("Failed on i0={}\nconds per i1:\n{}".format(
                          i0, '\n'.join(str(c) for c in conds)))
                      assert False
              # grads check
              if frontend == 'torch':
                  loss0.backward()
                  loss1.backward()
                  loss0, loss1 = npy(loss0), npy(loss1)
                  grad0, grad1 = npy(xt0.grad), npy(xt1.grad)
                  assert np.allclose(loss0, loss1)
                  assert np.allclose(grad0, grad1)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_current_vs_legacy()
    else:
        pytest.main([__file__, "-s"])
