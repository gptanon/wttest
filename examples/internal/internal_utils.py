# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as dtime


def timeit(fn, n_iters=10):
    """For CPU benchmarking."""
    t0 = dtime()
    for _ in range(n_iters):
        fn()
    return (dtime() - t0) / n_iters


def run_benchmarks(bench_fns, n_iters=10, verbose=True):
    # warmup - caching, internal reusables, etc
    for _ in range(2):
        for bench_fn in bench_fns.values():
            _ = bench_fn()

    # bench
    times = {}
    for name, bench_fn in bench_fns.items():
        t_avg = timeit(bench_fn, n_iters)
        times[name] = t_avg
        if verbose:
            print("{} {:.3g} sec".format(name, t_avg))
    return times


def viz_benchmarks(times, title=''):
    # data
    libraries = list(times)
    time_values = np.array(list(times.values()))
    y_pos = np.arange(len(libraries))
    bar_labels = np.array(["x{:.2g}".format(time_values.max() / n)
                           for n in time_values])

    # plot
    fig, ax = plt.subplots(figsize=(6*1.5, 2.5*1.5))
    blue = np.array([0., 74., 173.]) / 255
    red = np.array([173., 30., 30.]) / 255
    color = [red] + [blue] * (len(libraries) - 1)
    bars = ax.barh(y_pos, time_values, align='center', height=.65, color=color)
    ax.set_yticks(y_pos, labels=libraries, fontsize=15)
    ax.tick_params(axis='x', which='both', labelsize=15)
    ax.set_xlabel('sec', fontsize=15)
    ax.set_title(title, fontsize=16)

    # styling
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.grid(visible=True, axis='x')
    ax.set_axisbelow(True)
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    ax.bar_label(bars, bar_labels, padding=5,
                 fontsize=15)
    plt.show()


def scattering1d(x, pad_fn, backend, log2_T, psi1_f, psi2_f, phi_f,
                 paths_include_n2n1, ind_start=None, ind_end=None,
                 oversampling=0, max_order=2, average=True, out_type='array',
                 average_global=None, vectorized=None, vectorized_early_U_1=None,
                 psi1_f_stacked=None):
    """
    Main function implementing the 1-D scattering transform.
    See `help(wavespin.scattering1d.frontend.Scattering1D)`.

    Computes same coefficients as `wavespin.Scattering1D.scattering1d`, but
    in a different order. Equality is tested in
    `tests/scattering1d/test_legacy_scattering1d.py`.

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    frontend/core.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    fft = backend.fft

    for _ in range(100):
        U_1_hat = fft(x.reshape(32, -1).mean(axis=0))
