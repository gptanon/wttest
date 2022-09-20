# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Meta lookup for JTFS
====================

Show how a coefficient's meta can be retrieved, and vice versa.
"""

###############################################################################
# Import the necessary packages
# -----------------------------
import numpy as np
from wavespin.numpy import TimeFrequencyScattering1D
from wavespin.visuals import plot
from wavespin.toolkit import coeff2meta_jtfs, meta2coeff_jtfs, energy
from pprint import pprint

###############################################################################
# Generate A.M. cosine and create scattering object
# -------------------------------------------------
N = 2048
f1 = N // 8
f2 = N // 80
t = np.linspace(0, 1, N, 1)
c = np.cos(2*np.pi * f1 * t)
a = np.cos(2*np.pi * f2 * t)
x = a * c

J = 9
Q = 16
T = 2**7
J_fr = 4
Q_fr = 2
F = 8
average_fr = True
out_type = 'array'

jtfs = TimeFrequencyScattering1D(N, J, Q, J_fr=J_fr, Q_fr=Q_fr, T=T, F=F,
                                 average_fr=average_fr, out_type=out_type)
jmeta = jtfs.meta()
Scx = jtfs(x)

print("JTFS(x).shape == %s" % str(Scx.shape))

###############################################################################
# Helper function
# ---------------
def get_and_viz(meta_goal, title):
    coeffs_and_meta = meta2coeff_jtfs(Scx, jmeta, meta_goal)
    # This returns a list of tuples, [(coeff0, meta0), (coeff1, meta1), ...],
    # so unpack.
    coeffs = np.array([cm[0] for cm in coeffs_and_meta])
    # Compute and plot energies
    coeffs_e = energy(coeffs, axis=-1)
    # energy ratio
    e_x = energy(x)
    r = coeffs_e.sum() / e_x
    title = title % (100*r, '%')
    plot(coeffs_e, title=title, show=1)

    return coeffs_and_meta  # for user inspection


###############################################################################
# Fetch all coeffs with matching carrier frequency, `f1`
# -------------------------------------------------
meta_goal = {'xi': [None, None, f1/N]}
title = r"Energies, all xi1 = f1 | %.3g%s of energy(x)"
coeffs_and_meta = get_and_viz(meta_goal, title)

###############################################################################
# Fetch all coeffs with matching modulator frequency, `f2`
# -------------------------------------------------
meta_goal = {'xi': [None, f2/N, None]}
title = r"Energies, all xi2 = f2 | %.3g%s of energy(x)"
coeffs_and_meta = get_and_viz(meta_goal, title)

###############################################################################
# Fetch all coeffs with matching `f1, f2`
# ---------------------------------------
meta_goal = {'xi': [None, f2/N, f1/N]}
title = r"Energies, all xi2 = f2 & xi1 = f1 | %.3g%s of energy(x)"
coeffs_and_meta = get_and_viz(meta_goal, title)

# Display one coeff-meta pair
pprint(coeffs_and_meta[0])

# Considering there are 1600+ coefficients, above shows that peak activation
# coefficients are indeed fetched by querying modulator and carrier frequencies.

###############################################################################
# Fetch meta for a given coefficient
# ----------------------------------
coeff_idx = Scx.shape[1]//2
meta = coeff2meta_jtfs(Scx, jmeta, out_idx=coeff_idx)

pprint(meta)

# These methods will work with any `out_type`, but input signature will differ,
# refer to their docs.
