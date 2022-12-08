# -*- coding: utf-8 -*-
"""Tests that `bool(FORCED_PYTEST) == True` in `tests/scattering1d/utils.py`.
It's `test_z*` so it runs last, in case done locally.
"""
import pytest


def test_forced_pytest():
    from scattering1d.utils import FORCED_PYTEST
    assert FORCED_PYTEST


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
