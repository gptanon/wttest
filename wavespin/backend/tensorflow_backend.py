# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from .numpy_backend import NumPyBackend
import tensorflow as tf


class TensorFlowBackend(NumPyBackend):
    """
    TensorFlow general backend. For docstrings, see NumPy backend.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/backend/
    tensorflow_backend.py

    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    name = 'tensorflow'

    @classmethod
    def concatenate(cls, arrays, axis=-2, keep_cat_dim=False):
        fn = (tf.stack if keep_cat_dim else
              tf.concat)
        return fn(arrays, axis=axis)

    @staticmethod
    def modulus(x):
        norm = tf.abs(x)

        return norm

    @classmethod
    def sqrt(cls, x, dtype=None):
        if isinstance(x, (int, float)):
            x = tf.constant(x, dtype=dtype)
        elif dtype is not None:
            x = tf.cast(x, dtype=dtype)
        return tf.math.sqrt(x)

    @classmethod
    def mean(cls, x, axis=-1, keepdims=True):
        return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

    @classmethod
    def conj(cls, x, inplace=False):
        if inplace:
            raise Exception("TensorFlow doesn't support `out=`")
        return (tf.math.conj(x) if cls._is_complex(x) else
                x)

    @classmethod
    def zeros_like(cls, ref, shape=None):
        shape = shape if shape is not None else ref.shape
        return tf.zeros(shape, dtype=ref.dtype)

    @classmethod
    def reshape(cls, x, shape):
        return tf.reshape(x, shape)

    @classmethod
    def transpose(cls, x, axes):
        return tf.transpose(x, axes)

    @classmethod
    def assign_slice(cls, x, x_slc, slc, axis=-1):
        """Directly implemented only for indexing into last axis, for non-last
        will `transpose`.
        """
        # handle non-last axis
        ndim = x.ndim
        if axis < 0:
            axis = ndim + axis
        if axis != ndim - 1:
            # move target axis to last, assign to last, then move back
            tp_axes = list(range(ndim))
            tp_axes[axis] = ndim - 1
            tp_axes[ndim - 1] = axis
            o = tf.transpose(x, tp_axes)
            x_slc = tf.transpose(x_slc, tp_axes)
            o = cls.assign_slice(o, x_slc, slc, axis=-1)
            o = tf.transpose(o, tp_axes)
            return o

        slc_name = type(slc).__name__
        if slc_name == 'tuple':
            axis = 0
            while slc[axis].start is None and slc[axis].stop is None:
                axis += 1
            slc = slc[axis]
            slc_name = type(slc).__name__
        else:
            # default to last axis
            axis = -1

        if slc_name == 'list':
            pass
        elif slc_name == 'range':
            slc = list(slc)
        elif slc_name == 'slice':
            slc = list(range(slc.start or 0, slc.stop or x.shape[-1],
                             slc.step or 1))
        elif slc_name == 'int':
            slc = [slc]
        else:
            raise TypeError("`slc` must be list, range, slice, or int "
                            "(got %s)" % slc_name)

        if x_slc.ndim < x.ndim:
            x_slc = tf.expand_dims(x_slc, -1)

        slc_tf = tf.ones(x_slc.shape)
        slc_tf = tf.where(slc_tf).numpy().reshape(*x_slc.shape, x_slc.ndim)
        slc_tf[..., axis] = slc

        x = tf.tensor_scatter_nd_update(x, slc_tf, x_slc)
        return x

    @classmethod
    def cast(cls, x, dtype):
        return tf.cast(x, dtype)

    @classmethod
    def ensure_dtype(cls, x, dtype):
        if not str(x.dtype).endswith(dtype):
            x = cls.cast(x, dtype)
        return x
