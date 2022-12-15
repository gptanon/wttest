# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
"""
Reconstruct a Signal
====================
Generate an exponential chirp spanning the entire spectrum, transform it with
1D time scattering, and reconstruct it back using gradient descent - in
PyTorch, TensorFlow, and Jax.

Also see

  1. Animated reconstruction:
     https://github.com/OverLordGoldDragon/StackExchangeAnswers/blob/main/SignalProcessing/Q78512%20-%20Wavelet%20Scattering%20explanation/reconstruction.py

  2. JTFS reconstruction: `test_reconstruction_torch()`, in
     https://github.com/OverLordGoldDragon/wavespin/blob/main/tests/scattering1d/test_jtfs.py
"""

###############################################################################
# Import the necessary packages, configure
# ----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import warnings
from wavespin import Scattering1D
from wavespin.toolkit import echirp
from wavespin.utils.gen_utils import backend_has_gpu

# whether to try to use a GPU, if available and properly installed
TRY_GPU = True

###############################################################################
# Define visualizer, error function
# ---------------------------------
def viz(x_recovered, x_original_npy, losses, recon_errs):
    # `plt.title()` arguments
    title_kw = dict(weight='bold', fontsize=15, loc='left')
    # `plt.xlabel()`, `plt.ylabel()` arguments
    label_kw = dict(weight='bold', fontsize=15)

    # visualize reconstruction ###############################################
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ax = axes[0, 0]
    ax.plot(x_original_npy)
    ax.set_title("Original signal", **title_kw)
    ax.set_xlabel("Samples", **label_kw)

    ax = axes[0, 1]
    ax.specgram(x_original_npy, cmap='turbo')
    ax.set_title("Spectrogram: original signal", **title_kw)
    ax.set_xlabel("Time [samples]", **label_kw)
    ax.set_ylabel("Frequency (normalized)", **label_kw)

    ax = axes[1, 0]
    ax.plot(x_recovered)
    ax.set_title("Recovered signal", **title_kw)
    ax.set_xlabel("Samples", **label_kw)

    ax = axes[1, 1]
    ax.specgram(x_recovered, cmap='turbo')
    ax.set_title("Spectrogram: recovered signal", **title_kw)
    ax.set_xlabel("Time [samples]", **label_kw)
    ax.set_ylabel("Frequency (normalized)", **label_kw)

    plt.show()

    # visualize loss #########################################################
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    ax.plot(np.log10(losses))
    ax.set_title("log10(MSE): sc(original) vs sc(recon)", **title_kw)
    ax.set_xlabel("Iteration", **label_kw)

    ax = axes[1]
    ax.plot(recon_errs)
    ax.set_title("Relative L2: original vs recon", **title_kw)
    ax.set_xlabel("Iteration", **label_kw)

    plt.show()


def reconstruction_error(a, b):
    """Relative Euclidean distance to measure difference between original and
    reconstructed. "Relative" accounts for norms.
    """
    l2 = lambda x: np.sqrt(np.sum(np.abs(x)**2))
    return l2(a - b) / l2(a)

#%%############################################################################
# Generate signal, configure scattering
# -------------------------------------
N = 1024
x_original_npy = echirp(N, fmin=1, fmax=N/2)

J = 6
Q = 8
T = 2**J

#%%############################################################################
# PyTorch: create scattering object
# ---------------------------------
try:
    import torch

    # create scattering object
    sc = Scattering1D(N, J, Q, T=T, frontend='torch')

    # handle device
    if TRY_GPU and backend_has_gpu('torch'):
        device = 'cuda'
        sc.gpu()
    else:
        device = 'cpu'

    # handle signal array format
    x_original = torch.from_numpy(x_original_npy).to(
        device=device, dtype=torch.float32)

except ImportError:
    torch = None
    warnings.warn("Couldn't import torch, skipping the sub-example")

#%%############################################################################
# Optimize
# --------
if torch is None:
    pass
else:
    # we'll take distance with respect to these coefficients to use as loss
    Sx_original = sc(x_original)
    # primitive normalize
    div = Sx_original.max()
    Sx_original /= div

    # start our signal to reconstruct from random noise
    torch.manual_seed(0)
    x_recovered = torch.randn(N, device=device)
    x_recovered.requires_grad = True

    # create optimizer & loss function
    n_iters = 100
    optimizer = torch.optim.SGD([x_recovered], lr=40000, momentum=.9,
                                nesterov=True)
    loss_fn = torch.nn.MSELoss()

    # optimize, track losses
    print("PyTorch: optimizing over %s iterations" % n_iters, flush=True)
    losses, recon_errs = [], []

    for _ in range(n_iters):
        optimizer.zero_grad()
        Sx_recovered = sc(x_recovered)
        Sx_recovered /= div
        loss = loss_fn(Sx_recovered, Sx_original)
        loss.backward()
        optimizer.step()

        xon = x_original.detach().cpu().numpy()
        xrn = x_recovered.detach().cpu().numpy()
        losses.append(float(loss.detach().cpu().numpy()))
        recon_errs.append(reconstruction_error(xon, xrn))
        print(end='.', flush=True)

#%%############################################################################
# Plot results
# ------------
if torch is None:
    pass
else:
    x_recovered = x_recovered.cpu().detach().numpy()
    viz(x_recovered, x_original_npy, losses, recon_errs)

#%%############################################################################
# TensorFlow: create scattering object
# ------------------------------------
try:
    import tensorflow as tf

    # create scattering object
    sc = Scattering1D(N, J, Q, T=T, frontend='tensorflow')

    # handle device
    if TRY_GPU and backend_has_gpu('tensorflow'):
        device = 'cuda'
        sc.gpu()
    else:
        device = 'cpu'

    # handle signal array format
    x_original = tf.convert_to_tensor(x_original_npy, dtype=tf.float32)

except ImportError:
    tf = None
    # in ReadTheDocs this is done to reduce build time
    warnings.warn("Couldn't import tensorflow, skipping the sub-example")

#%%############################################################################
# Optimize
# --------
if tf is None:
    pass
else:
    # we'll take distance with respect to these coefficients to use as loss
    Sx_original = sc(x_original)
    # primitive normalize
    div = tf.reduce_max(Sx_original)
    Sx_original /= div

    # start our signal to reconstruct from random noise
    tf.random.set_seed(0)
    x_recovered = tf.Variable(tf.random.normal([N]), trainable=True)

    # create optimizer & loss function
    n_iters = 100
    optimizer = tf.keras.optimizers.SGD(lr=40000, momentum=.9, nesterov=True)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # optimize, track losses
    print("TensorFlow: optimizing over %s iterations" % n_iters, flush=True)
    losses, recon_errs = [], []

    for _ in range(n_iters):
        with tf.GradientTape(persistent=True) as g:
            g.watch(x_recovered)
            Sx_recovered = sc(x_recovered)
            loss = loss_fn(Sx_original, Sx_recovered)

        grads = g.gradient(loss, x_recovered)
        optimizer.apply_gradients(zip([grads], [x_recovered]))

        xon = x_original.numpy()
        xrn = x_recovered.numpy()
        losses.append(float(loss.numpy()))
        recon_errs.append(reconstruction_error(xon, xrn))
        print(end='.', flush=True)

#%%############################################################################
# Plot results
# ------------
if tf is None:
    pass
else:
    x_recovered = x_recovered.numpy()
    viz(x_recovered, x_original_npy, losses, recon_errs)

#%%############################################################################
# Jax: create scattering object
# -----------------------------
try:
    import jax
    import jax.numpy as jnp
    import optax

    # create scattering object
    sc = Scattering1D(N, J, Q, T=T, frontend='jax')

    # handle device
    if TRY_GPU and backend_has_gpu('jax'):
        device = 'gpu'
        sc.gpu()
    else:
        device = 'cpu'

    # handle signal array format
    x_original = jax.device_put(jnp.asarray(x_original_npy),
                                device=jax.devices(device)[0])

except ImportError:
    jax = None
    # in ReadTheDocs this is done to reduce build time
    warnings.warn("Couldn't import jax, skipping the sub-example")

#%%############################################################################
# Optimize
# --------
if jax is None:
    pass
else:
    # we'll take distance with respect to these coefficients to use as loss
    Sx_original = sc(x_original)
    # primitive normalize
    div = Sx_original.max()
    Sx_original /= div

    # start our signal to reconstruct from random noise
    seed = jax.random.PRNGKey(0)
    x_recovered = jax.random.normal(seed, jnp.atleast_1d(N), dtype='float32')
    x_recovered = jax.device_put(x_recovered, device=jax.devices(device)[0])

    # create optimizer & loss function
    n_iters = 100
    optimizer = optax.sgd(learning_rate=40000, momentum=.9, nesterov=True,
                          accumulator_dtype=x_recovered.dtype)
    params = {'x_recovered': x_recovered}
    opt_state = optimizer.init(params)

    def compute_and_track_loss(params, Sx_original, div, losses):
        loss = 2 * optax.l2_loss(sc(params['x_recovered']) / div,
                                 Sx_original).mean()
        losses.append(float(loss.aval.val))
        return loss

    # optimize, track losses
    print("Jax: optimizing over %s iterations" % n_iters, flush=True)
    losses, recon_errs = [], []

    for _ in range(n_iters):
        grads = jax.grad(compute_and_track_loss)(params, Sx_original, div, losses)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        xon, xrn = np.array(x_original), np.array(params['x_recovered'])
        recon_errs.append(reconstruction_error(xon, xrn))
        print(end='.', flush=True)

#%%############################################################################
# Plot results
# ------------
if jax is None:
    pass
else:
    x_recovered = np.array(params['x_recovered'])
    viz(x_recovered, x_original_npy, losses, recon_errs)

#%%############################################################################
# Note
# ----
# Results can be improved significantly with better optimization procedures,
# including warmup, swapping to L1 loss, and normalization. See the references.
# Also, pointwise distance measures directly in time domain have serious flaws
# and aren't necessarily reflective of reconstruction quality.
