# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Miscellaneous tools."""
import numpy as np
import warnings
from scipy.fft import fft, ifft
from itertools import zip_longest, chain

from ...utils.gen_utils import _infer_backend, ExtendedUnifiedBackend
from ...utils.measures import compute_analyticity


def fft_upsample(xf, factor=2, time_to_time=False, analytic=None,
                 anti_analytic=None, real=None):
    """Sinc-interpolates `xf`. Perfect recovery if `xf` is unaliased.

    For real-valued `x`, having Nyquist bin loses an imaginary bin. For
    complex-valued, non-analytic or non-anti-analytic `x`, having Nyquist bin
    completely loses one bin. Either way, it's aliasing.

    See https://dsp.stackexchange.com/q/83781/50076

    Parameters
    ----------
    xf : tensor
        1D input in frequency domain, or in time if `time_to_time=True`.
        `len(xf)` must be even.

    factor : int >= 2
        Upsampling factor, linear. Must be a power of 2 and >=2.

    time_to_time : bool (default False)
        True:  assumes `xf` is in time domain, and returns time-domain output.
        False: assumes `xf` is in freq domain, and returns freq-domain output.

    analytic : None
        Whether `xf` is strictly analytic (negative freqs zero).
        Determined automatically if None for `len(x) > 2`.

    anti_analytic : None
        Whether `xf` is strictly anti-analytic (positive freqs zero).
        Determined automatically if None for `len(x) > 2`.

    real : None
        Whether `xf` is real-valued in time.
        Determined automatically if None.

    Returns
    -------
    out : tensor
        `xf` upsampled by `factor`.
    """
    # handle inputs ##########################################################
    assert xf.ndim == 1, xf.shape
    N = len(xf)
    assert N == 1 or N % 2 == 0, N
    assert factor >= 2, factor
    assert np.log2(factor).is_integer(), np.log2(factor)

    eps = np.finfo(xf.dtype).eps * 10
    if real is None:
        xt = xf if time_to_time else ifft(xf, workers=-1)
        real = bool(np.abs(xt.imag).max() < eps)

    if time_to_time:
        xf = fft(xf, workers=-1)

    if analytic is None and anti_analytic is None and N != 1:
        if N == 2:
            warnings.warn("Cannot automatically determine analyticity for "
                          "input length 2; will default to `False`.")
            analytic = anti_analytic = False
        else:
            # for upsampling purposes, `2` is useless
            analyticity = compute_analyticity(xf)
            analytic = bool(analyticity == 1)
            anti_analytic = bool(analyticity == -1)
    any_analytic = bool(analytic or anti_analytic)

    # compute ################################################################
    nyq = xf[N//2]
    n_to_add = N * factor - N
    xf = xf * factor

    if abs(nyq) > eps and not (any_analytic or real) and N > 1:
        # also aliases imaginary bin for real case, but don't warn since it's
        # too common. it's covered in docstring
        warnings.warn("Non-zero Nyquist bin for non-analytic and non-real input; "
                      "irrecoverable aliasing, will zero.")
        nyq *= 0

    if N == 1:
        out = np.hstack([xf, np.zeros(n_to_add)])
    else:
        zeros = (np.zeros(n_to_add + N//2 - 1) if any_analytic else
                 np.zeros(n_to_add - 1))
        if analytic:
            out = np.hstack([xf[:N//2 + 1], zeros])
        elif anti_analytic:
            out = np.hstack([xf[0], zeros, xf[-N//2:]])
        else:
            out = np.hstack([xf[:N//2],
                             nyq/2, zeros, np.conj(nyq)/2,
                             xf[-(N//2 - 1):] if N != 2 else []])

    if time_to_time:
        out = ifft(out, workers=-1)
    if real:
        out = out.real
    return out


def tensor_padded(seq, pad_value=0, init_fn=None, cast_fn=None, ref_shape=None,
                  left_pad_axis=None, general=True):
    """Make tensor from variable-length `seq` (e.g. nested list) padded with
    `fill_value`.

    Parameters
    ----------
    seq : list[tensor]
        Nested list of tensors.

    pad_value : float
        Value to pad with.

    init_fn : function / None
        Instantiates packing tensor, e.g. `lambda shape: torch.zeros(shape)`.
        Defaults to `backend.full`.

    cast_fn : function / None
        Function to cast tensor values before packing, e.g.
        `lambda x: torch.tensor(x)`.

    ref_shape : tuple[int] / None
        Tensor output shape to pack into, instead of what's inferred for `seq`,
        as long as >= that shape. Shape inferred from `seq` is the minimal size
        determined from longest list at each nest level, i.e. we can't go lower
        without losing elements.

        Tuple can contain `None`: `(None, 3)` will pad dim0 per `seq` and dim1
        to 3, unless `seq`'s dim1 is 4, then will pad to 4.

        Recommended to pass this argument if applying `tensor_padded` multiple
        times, as time to infer shape is significant, especially relative to
        GPU computation.

    left_pad_axis : int / tuple[int] / None
        Along these axes, will pad from left instead of the default right.
        Not implemented for dim0 (`0 in left_pad_axis`).

    general : bool (default True)
        If `False`, will use a much faster routine that's for JTFS.

    Not implemented for TensorFlow: will convert to numpy array then rever to
    TF tensor.

    Code borrows from: https://stackoverflow.com/a/27890978/10133797
    """
    iter_axis = [0]
    prev_axis = [iter_axis[0]]

    def fill_tensor(arr, seq, fill_value=0):
        if iter_axis[0] != prev_axis[0]:
            prev_axis[0] = iter_axis[0]

        if arr.ndim == 1:
            try:
                len_ = len(seq)
            except TypeError:
                len_ = 0

            if len_ == 0:
                pass
            elif len(shape) not in left_pad_axis:  # right pad
                arr[:len_] = cast_fn(seq)
            else:  # left pad
                arr[-len_:] = cast_fn(seq)
        else:
            iter_axis[0] += 1

            left_pad = bool(iter_axis[0] in left_pad_axis)
            if left_pad:
                seq = _IterWithDelay(seq, len(arr) - len(seq), fillvalue=())

            for subarr, subseq in zip_longest(arr, seq, fillvalue=()):
                fill_tensor(subarr, subseq, fill_value)
                if subarr.ndim != 1:
                    iter_axis[0] -= 1

    # infer `init_fn` and `cast_fn` from `seq`, if not provided ##############
    backend, backend_name = _infer_backend(seq, get_name=True)
    is_tf = bool(backend_name == 'tensorflow')
    if is_tf:
        tf = backend
        backend = np
        backend_name = 'numpy'

    # infer dtype & whether on GPU
    sq = seq
    while isinstance(sq, list):
        sq = sq[0]
    dtype = sq.dtype if hasattr(sq, 'dtype') else type(sq)
    if backend_name == 'torch':
        device = sq.device
    else:
        device = None

    if init_fn is None:
        if backend_name in ('numpy', 'jaxlib', 'jax'):
            if is_tf:
                dtype = dtype.name
            init_fn = lambda s: np.full(s, pad_value, dtype=dtype)
        elif backend_name == 'torch':
            init_fn = lambda s: backend.full(s, pad_value, dtype=dtype,
                                             device=device)

    if cast_fn is None:
        if is_tf:
            cast_fn = lambda x: x.numpy()
        elif backend_name in ('numpy', 'jaxlib', 'jax'):
            cast_fn = lambda x: x
        elif backend_name == 'torch':
            cast_fn = lambda x: (backend.tensor(x, device=device)
                                 if not isinstance(x, backend.Tensor) else x)

    ##########################################################################
    # infer shape if not provided
    if ref_shape is None or any(s is None for s in ref_shape):
        shape = list(find_shape(seq, fast=not general))
        # override shape with `ref_shape` where provided
        if ref_shape is not None:
            for i, s in enumerate(ref_shape):
                if s is not None and s >= shape[i]:
                    shape[i] = s
    else:
        shape = ref_shape
    shape = tuple(shape)

    # handle `left_pad_axis`
    if left_pad_axis is None:
        left_pad_axis = ()
    elif isinstance(left_pad_axis, int):
        left_pad_axis = [left_pad_axis]
    elif isinstance(left_pad_axis, tuple):
        left_pad_axis = list(left_pad_axis)
    # negatives -> positives
    for i, a in enumerate(left_pad_axis):
        if a < 0:
            # +1 since counting index depth which goes `1, 2, ...`
            left_pad_axis[i] = (len(shape) + 1) + a
    if 0 in left_pad_axis:
        raise NotImplementedError("`0` in `left_pad_axis`")

    # fill
    arr = init_fn(shape)
    fill_tensor(arr, seq, fill_value=pad_value)

    # revert if needed
    if is_tf:
        arr = tf.convert_to_tensor(arr)
    return arr


def find_shape(seq, fast=False):
    """Finds shape to pad a variably nested list to.
    `fast=True` uses an implementation optimized for JTFS.
    """
    if fast:
        """Assumes 4D/5D and only variable length lists are in dim3."""
        flat = chain.from_iterable
        try:
            dim4 = len(seq[0][0][0][0])  # 5D
            dims = (len(seq), len(seq[0]), len(seq[0][0]),
                    max(map(len, flat(flat(seq)))), dim4)
        except:
            dims = (len(seq), len(seq[0]),
                    max(map(len, flat(seq))), len(seq[0][0][0]))
    else:
        dims = _find_shape_gen(seq)
    return dims


# metrics ####################################################################
def energy(x, axis=None, kind='l2', keepdims=False):
    """Compute energy. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)` (so actually L2^2).
    """
    x = x['coef'] if isinstance(x, dict) else x
    B = ExtendedUnifiedBackend(x)
    ckw = dict(x=x, axis=axis, keepdims=keepdims)
    out = (B.norm(**ckw, ord=1) if kind == 'l1' else
           B.norm(**ckw, ord=2)**2)
    if np.prod(out.shape) == 1:
        out = float(out)
    return out


def l2(x, axis=None, keepdims=True):
    """`sqrt(sum(abs(x)**2))`."""
    B = ExtendedUnifiedBackend(x)
    out = B.norm(x, ord=2, axis=axis, keepdims=keepdims)
    if np.prod(tuple(out.shape)) == 1:
        out = out[0]
    return out

def rel_l2(x0, x1, axis=None, adj=False):
    """Relative Euclidean distance, i.e. distance w.r.t. own norm.
    Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    ref = l2(x0, axis) if not adj else (l2(x0, axis) + l2(x1, axis)) / 2
    return l2(x1 - x0, axis) / ref


def l1(x, axis=None, keepdims=True):
    """`sum(abs(x))`."""
    B = ExtendedUnifiedBackend(x)
    out = B.norm(x, ord=1, axis=axis, keepdims=keepdims)
    if np.prod(tuple(out.shape)) == 1:
        out = out[0]
    return out

def rel_l1(x0, x1, adj=False, axis=None):
    ref = l1(x0, axis) if not adj else (l1(x0, axis) + l1(x1, axis)) / 2
    return l1(x1 - x0, axis) / ref


def rel_ae(x0, x1, eps=None, ref_both=True):
    """Relative absolute error."""
    B = ExtendedUnifiedBackend(x0)
    if ref_both:
        if eps is None:
            eps = make_eps(x0, x1)
        ref = (B.abs(x0) + B.abs(x1))/2 + eps
    else:
        if eps is None:
            eps = make_eps(x0)
        ref = B.abs(x0) + eps
    return B.abs(x0 - x1) / ref


def make_eps(x0, x1=None):
    """Make epsilon from maximum of `x0` or maxima of `x0` and `x1`, less
    conservative alternative to machine epsilon.
    """
    B = ExtendedUnifiedBackend(x0)
    if x1 is None:
        eps = B.abs(x0).max() / 1000
    else:
        eps = (B.abs(x0).max() + B.abs(x1).max()) / 2000
    eps = max(eps, 10 * np.finfo(B.numpy(x0).dtype).eps)
    return eps


# helpers ####################################################################
def _find_shape_gen(seq):
    """Code borrows from https://stackoverflow.com/a/27890978/10133797"""
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [_find_shape_gen(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in
                           zip_longest(*shapes, fillvalue=1))


class _IterWithDelay():
    """Allows implementing left padding by delaying iteration of a sequence.
    Used by `toolkit.tensor_padded`.
    """
    def __init__(self, x, delay, fillvalue=()):
        self.x = x
        self.delay = delay
        self.fillvalue = fillvalue

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx > self.delay - 1:
            idx = self.idx - self.delay
            if idx < len(self.x):
                out = self.x[idx]
            else:
                raise StopIteration
        else:
            out = self.fillvalue
        self.idx += 1
        return out
