# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""PyTest configurations file."""
# TODO rm?


def pytest_addoption(parser):
    parser.addoption("--skip_long_jtfs", action="store_true")
    parser.addoption("--skip_visuals", action="store_true")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.skip_long_jtfs
    if 'skip_long_cmd' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("skip_long_cmd", [option_value])

    option_value = metafunc.config.option.skip_visuals
    if 'skip_visuals_cmd' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("skip_visuals_cmd", [option_value])
