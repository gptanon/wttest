# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""PyTest configurations file."""
import os


def pytest_addoption(parser):
    parser.addoption("--skip_long_jtfs", action="store_true")
    parser.addoption("--skip_visuals", action="store_true")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    opts = metafunc.config.option

    os.environ['CMD_SKIP_LONG_JTFS'] = '1' if opts.skip_long_jtfs else '0'
    os.environ['CMD_SKIP_VISUALS'] = '1' if opts.skip_visuals else '0'
