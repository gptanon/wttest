# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------

"""
WaveSpin
========

Joint Time-Frequency Scattering, Wavelet Scattering: features for audio,
biomedical, and other applications, in Python

WaveSpin features scattering transform implementations that maximize accuracy,
flexibility, and speed. Included are visualizations and convenience utilities
for transform and coefficient introspection and debugging.
"""

import os
import re
from numpy import get_include as numpy_get_include
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return reader.read()


def get_requirements(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_matched = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                version_file, re.M)
    if version_matched:
        return version_matched.group(1)
    raise RuntimeError('Unable to find version')

# Cython extensions
ext_modules = cythonize(Extension("wavespin.utils._compiled._algos",
                                  ["wavespin/utils/_compiled/_algos.pyx"]),
                        language_level=3)

setup(
    name="WaveSpin",
    version=find_version('wavespin', '__init__.py'),
    packages=find_packages(exclude=['tests', 'examples']),
    url="https://github.com/OverLordGoldDragon/wavespin",
    license="MIT",
    author="John Muradeli",
    author_email="john.muradeli@gmail.com",
    description=("Joint Time-Frequency Scattering, Wavelet Scattering: features "
                 "for audio, biomedical, and other applications, in Python"),
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    keywords=(
        "scattering-transform wavelets signal-processing visualization "
        "pytorch tensorflow jax python"
    ),
    install_requires=get_requirements('requirements.txt'),
    python_requires=">=3.7",
    tests_require=["pytest>=4.0", "pytest-cov"],
    ext_modules=ext_modules,
    include_dirs=[numpy_get_include()],
    include_package_data=True,
    zip_safe=True,
    #Specify any non-python files to be distributed with the package
    package_data = {'' : ['utils/_fonts/*.ttf']},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
