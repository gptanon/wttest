# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
JTFS 2D Conv-Net
================
Build a JTFS network to output batched 3D tensors - two spatial dims plus one
channel dim - and feed them to a 2D convolutional newtork.

Application to audio, with reproducing code, is found in

    - Paper: https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_25.pdf
    - Code: https://github.com/OverLordGoldDragon/wavespin/tree/dafx2022-jtfs
"""

###############################################################################
# Import the necessary packages
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
import torch
import torch.nn as nn
from wavespin import TimeFrequencyScattering1D
from wavespin.toolkit import normalize

#%%############################################################################
# Generate trumpet and create scattering object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Refer to the "Intro to JTFS" example for parameter descriptions.
x = np.load('librosa_trumpet.npy')
N = x.shape[-1]

J = 10
Q = 16
T = 2**11
J_fr = 4
Q_fr = 1
F = 8
average_fr = True
out_type = 'array'
# the key config
out_3D = True

configs = dict(J=J, shape=N, Q=Q, T=T, J_fr=J_fr, Q_fr=Q_fr, F=F,
               average_fr=average_fr, out_type=out_type, out_3D=out_3D)
jtfs = TimeFrequencyScattering1D(**configs, frontend='torch')

#%%############################################################################
# Scatter
# ^^^^^^^
out = jtfs(x)

#%%
# `out_3D=True` with `out_type='array'` must split the first-order and joint
# coefficients into a tuple, as the shapes can't directly concatenate.
# In this example, for simplicity, we only keep the joint coefficients and
# discard first-order ones: *this is bad*. Instead,
#   - The first order should either pad and concatenate with joint, or,
#     much better, be fed as a separate input to the CNN.
#   - The different joint pairs should also not be all contatenated, instead
#     `wavespin.toolkit.pack_coeffs_jtfs()` should be used; refer to its docs.
#
Scx10, Scx_joint = out
_Scx = Scx_joint

# Show shape
print("\nJTFS all-in-one shape:\n{}".format(_Scx.shape))

###############################################################################
# The shape - `(batch_size, n1_fr * n2, n1, t)` - is described as:
#
#     - `n1`: frequency [Hz], first-order temporal variation
#     - `n2`: frequency [Hz], second-order temporal variation
#       (frequency of amplitude modulation)
#     - `n1_fr`: quefrency [cycles/octave], first-order frequential variation
#       (frequency of frequency modulation bands, roughly.
#     - `t`: time [sec]
#     - `n1_fr * n2` is the unrolling (flattening) of `n1_fr` and `n2` axes,
#       which we treat as the channel axis.

#%%############################################################################
# Normalize
# ^^^^^^^^^
# Since we convolve in 2D - specifically over `(n1, t)` - we normalize
# the last two dimensions: set their standard deviation to 1 and mean to 0.
# The log is taken before this step. Refer to `wavespin.toolkit.normalize()`.

# Must unroll spatial dimensions, as `normalize` expects input as
# `(samples, features, spatial)` - then undo the reshape.
s = _Scx.shape
Scx = normalize(_Scx.reshape(s[0], s[1], -1),
                mean_axis=(-1, -2), std_axis=(-1, -2)
                ).reshape(*s)

# print stats
print("\nNormed stats:\n{:.1f}, {:.1f} -- mean, std".format(
    Scx.mean(), Scx.std()))

###############################################################################
# It's fine if these differ from 1 and 0 - in fact, they should. They'll be
# 1 and 0 on *per-channel* basis, so they're also likely to be near 1 and 0
# globally.

#%%############################################################################
# Feed to simple 2D conv-net
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Minimal deep network
class NetSimple(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(n_channels, out_channels=32, kernel_size=3)
        self.conv1 = nn.Conv2d(32, out_channels=64, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)  # global avg
        self.fc   = nn.Linear(64, 2)  # e.g. binary classification

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.pool(x).squeeze(-1).squeeze(-1)  # drop spatial dims
        return self.fc(x)

xt = torch.from_numpy(x)

# initialize network
n_paths = Scx.shape[1]
net_s = NetSimple(n_paths)

# get outputs, backprop
out = net_s(Scx)
loss = out.mean()
loss.backward()

# confirm gradients
assert net_s.conv0.weight.grad is not None

# print network
print("\n{}".format(net_s))

#%%############################################################################
# Advanced example: multi-input network
# -------------------------------------
# Here we handle the different output pairs correctly:
#   - `S0`: (time) zeroth order -- `(t,)`
#   - `S1`: (time) first order -- `(n1, t)`
#   - `phi_t * phi_f`: (joint) joint lowpass -- `(1, 1, n1, t)`
#   - `phi_t * psi_f`: (joint) time lowpass (w/ freq bandpass) -- `(1, n1_fr, n1, t)`
#   - `psi_t * phi_f`: (joint) freq lowpass (w/ time bandpass) -- `(n2, 1, n1, t)`
#   - `psi_t * psi_f_up`: (joint) spin up -- `(n2, n1_fr, n1, t)`
#   - `psi_t * psi_f_dn`: (joint) spin down -- `(n2, n1_fr, n1, t)`
# These pairs both capture distinct time-frequency geometries, and have greatly
# different norms: lumping them together messes with learned features and
# the normalization preprocessing.
#
# `out_type` is a dynamic parameter, so we can change it without rebuilding.
# For fun, also show all such parameters:
jtfs.out_type = 'dict:array'
print("\nJTFS dynamic parameters:\n%s" % str(jtfs.DYNAMIC_PARAMETERS_JTFS))

#%%
# Scatter and print shapes
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Note, again, dim 1 is `n1_fr * n2`.
out = jtfs(x)

print("\n(batch_size, n2*n1_fr, n1, t):")
for pair, c in out.items():
    print(tuple(c.shape), '--', pair)

#%%
# Normalize each pair separately
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Use same norm parameters, except for `S0`. We can overwrite `out` but
# make a new variable in case we want to debug.

Scx = {}
for pair in out:
    if pair == 'S0':
        continue
    s = out[pair].shape
    Scx[pair] = normalize(out[pair].reshape(s[0], s[1], -1),
                          mean_axis=(-1, -2), std_axis=(-1, -2),
                          ).reshape(*s)

###############################################################################
# `S0` generally shouldn't be thrown out, but in audio it'll often add little,
# and here it complicates the example, so discard. `S1` doesn't involve any
# reshaping, it's already in its true structure.

#%%###########################################################################
# Concatenate
# ^^^^^^^^^^^
# Now that we've normed separately, we can re-join the joint pairs along channels,
# while keeping `S1` separate.

S1 = Scx['S1']
# extend channels dim, equivalently 1 channel
S1 = S1[:, None]

S_joint = torch.cat([c for pair, c in Scx.items() if pair not in ('S0', 'S1')],
                    dim=1)
print("\nPair-handled shapes:\n{} -- S1.shape\n{} -- S_joint.shape".format(
      tuple(S1.shape), tuple(S_joint.shape)))

#%%############################################################################
# Feed to multi-input 2D conv-net
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Each pair gets its own tail.
class Net(nn.Module):
    def __init__(self, n_channels0, n_channels1):
        super().__init__()
        # S1 network
        self.conv0_0 = nn.Conv2d(n_channels0, out_channels=4, kernel_size=3)
        self.conv0_1 = nn.Conv2d(4,           out_channels=8, kernel_size=3)

        # S_joint network: should be deeper than S1 as it's more intricate
        self.conv1_0 = nn.Conv2d(n_channels1, out_channels=32, kernel_size=3,
                                 stride=2)
        self.conv1_1 = nn.Conv2d(32, out_channels=64, kernel_size=3,
                                 stride=2)
        self.conv1_2 = nn.Conv2d(64, out_channels=128, kernel_size=3)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)  # global avg
        self.fc   = nn.Linear(128 + 8, 2)  # e.g. binary classification
        self.flatten = nn.Flatten()

    def forward(self, S1, s_joint):
        x0, x1 = S1, s_joint
        # S1
        x0 = self.relu(self.conv0_0(x0))
        x0 = self.relu(self.conv0_1(x0))

        # S_joint
        x1 = self.relu(self.conv1_0(x1))
        x1 = self.relu(self.conv1_1(x1))
        x1 = self.relu(self.conv1_2(x1))

        # Feature merger
        x0 = self.flatten(self.pool(x0))
        x1 = self.flatten(self.pool(x1))
        x = torch.cat([x0, x1], dim=1)

        # out
        return self.fc(x)

xt = torch.from_numpy(x)

# initialize network
n_paths_S1 = S1.shape[1]
n_paths_joint = S_joint.shape[1]
net = Net(n_paths_S1, n_paths_joint)

# get outputs, backprop
out = net(S1, S_joint)
loss = out.mean()
loss.backward()

# confirm gradients
assert net.conv0_0.weight.grad is not None
assert net.conv1_0.weight.grad is not None

# print network
print("\n{}".format(net))

###############################################################################
# Note that this network doesn't address the "messes with learned features"
# problem due to concatenating the pairs; while not ideal, the configuration
# should still work well. One can extend this example by adding more network
# inputs and not concatenating (before learned layers).
# Future examples will explore more ideal handling, including 3D and 4D convs.
#
# Also note, above hyperparemeters are awful, despite being used in some
# publications. A better selection is provided in the referenced paper.

#%%############################################################################
# Visualize network
# ^^^^^^^^^^^^^^^^^^
# Used Netron - https://github.com/lutzroeder/netron - first converting to ONNX
# with
#
# ::
#
#     torch.onnx.export(net.cuda(), (S1.cuda(), S_joint.cuda()), 'jtfs_net.pt',
#                       opset_version=9)
from wavespin.utils._examples_utils import display_image

display_image('../docs/source/_images/examples/jtfs_2d_cnn_net.png',
              copy_to_pwd=True)
