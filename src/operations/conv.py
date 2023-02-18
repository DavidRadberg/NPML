import numpy as np
from numpy.lib.stride_tricks import as_strided


def conv2d(imgs: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    h, w, d, n = imgs.shape
    kh, kw, kd, kn = kernels.shape

    oh = h - kh + 1
    ow = w - kw + 1

    assert d == kd

    group_shape = (oh, ow, n)
    patch_shape = (kh, kw, d)
    stride_shape = group_shape + patch_shape

    oh_stride = imgs.strides[0]
    ow_stride = imgs.strides[1]
    n_stride = imgs.strides[3]

    kh_stride = oh_stride
    kw_stride = ow_stride
    d_stride = imgs.strides[2]

    strides = (oh_stride, ow_stride, n_stride, kh_stride, kw_stride, d_stride)

    patches = as_strided(imgs, shape=stride_shape, strides=strides)

    patches_2d = np.reshape(patches, (oh * ow * n, -1))
    kernels_2d = np.reshape(kernels, (-1, kn))

    result_2d = np.matmul(patches_2d, kernels_2d)

    result = np.reshape(result_2d, (oh, ow, n, kn))
    return np.transpose(result, [0, 1, 3, 2])


def conv2d_sum(imgs: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    h, w, n = imgs.shape
    kh, kw, kn = kernels.shape

    assert n == kn

    oh = h - kh + 1
    ow = w - kw + 1

    group_shape = (oh, ow)
    patch_shape = (kh, kw, n)
    patches_shape = group_shape + patch_shape

    oh_stride = imgs.strides[0]
    ow_stride = imgs.strides[1]

    kh_stride = oh_stride
    kw_stride = ow_stride
    n_stride = imgs.strides[2]

    strides = (oh_stride, ow_stride, kh_stride, kw_stride, n_stride)

    patches = as_strided(imgs, shape=patches_shape, strides=strides)

    patches_2d = np.reshape(patches, (oh * ow, -1))
    kernels_2d = np.reshape(kernels, (-1))

    results_2d = np.matmul(patches_2d, kernels_2d)

    return np.reshape(results_2d, (oh, ow))
