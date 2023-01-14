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
    for _ in range(3):
        for name in bench_fns:
            _ = bench_fns[name]()

    # bench
    times = {}
    for name in bench_fns:
        t_avg = timeit(bench_fns[name], n_iters)
        times[name] = t_avg
        if verbose:
            print("{} {:.3g} sec".format(name, t_avg))
    return times


def viz_benchmarks(times, title=''):
    # data
    libraries = list(times)
    time_values = np.array(list(times.values()))
    y_pos = np.arange(len(libraries))
    bar_labels = []
    for name, time_value in times.items():
        fmt = ("x{:.4g}" if 'GPU' in name else
               "x{:.2g}")
        bar_labels.append(fmt.format(time_values.max() / time_value))
    bar_labels = np.array(bar_labels)

    # plot
    fig, ax = plt.subplots(figsize=(9, 6*len(libraries)/8))
    blue = np.array([0., 74., 173.]) / 255
    red = np.array([173., 30., 30.]) / 255
    color = [(red if 'WaveSpin' in name else blue) for name in libraries]
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
