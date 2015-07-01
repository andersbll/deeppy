import itertools
from random import shuffle
import numpy as np
import cudarray as ca
import deeppy as dp
from deeppy.feedforward.convnet_layers import padding
from test_layers import check_grad, check_params


batch_sizes = [1, 5, 10]
n_channels = [1, 3, 8]
img_shapes = [(1, 6), (6, 1), (7, 7), (8, 8), (9, 15)]


def shuffled(l):
    l = list(l)
    shuffle(l)
    return l


def img_out_shape(img_shape, win_shape, stride, border_mode):
    pad = padding(win_shape, border_mode)
    h = (img_shape[0] + 2*pad[0] - win_shape[0]) // stride[0] + 1
    w = (img_shape[1] + 2*pad[1] - win_shape[1]) // stride[1] + 1
    return h, w


def test_convolution():
    n_filters = [1, 5, 10]
    win_shapes = [(1, 1), (3, 3), (5, 5)]
    strides = [(1, 1), (2, 2), (3, 3)]
    border_modes = ['valid', 'same', 'full']
    confs = itertools.product(batch_sizes, n_channels, img_shapes, n_filters,
                              win_shapes, strides, border_modes)

    # Sample random parameter configurations to reduce workload.
    confs = shuffled(confs)[:100]

    for (batch_size, n_channel, img_shape, n_filter, win_shape, stride,
         border_mode) in confs:
        if img_shape[0] < win_shape[0] or img_shape[1] < win_shape[1]:
            continue
        print('Convolution: batch_size=%i, n_channel=%i, img_shape=%s, '
              'n_filter=%i, win_shape=%s, stride=%s, border_mode=%s'
              % (batch_size, n_channel, str(img_shape), n_filter,
                 str(win_shape), str(stride), border_mode))
        x_shape = (batch_size, n_channel) + img_shape
        w_shape = (n_filter, n_channel) + win_shape
        x = np.random.normal(size=x_shape).astype(ca.float_)
        w = np.random.normal(size=w_shape).astype(ca.float_)*1e-4
        b = np.random.normal(size=(1, n_filter, 1, 1)).astype(ca.float_)*1e-4
        layer = dp.Convolution(n_filter, win_shape, weights=w, bias=b,
                               strides=stride, border_mode=border_mode)
        layer._setup(x_shape)
        y_img_shape = img_out_shape(img_shape, win_shape, stride, border_mode)
        assert layer.y_shape(x_shape) == (batch_size, n_filter) + y_img_shape

        check_grad(layer, x)
        check_params(layer)


def test_pool():
    win_shapes = [(1, 1), (2, 2), (3, 3)]
    strides = [(1, 1), (2, 2), (3, 3)]
    border_modes = ['valid', 'same', 'full']
    methods = ['max', 'avg']
    confs = itertools.product(batch_sizes, n_channels, img_shapes,
                              win_shapes, strides, border_modes, methods)

    # Sample random parameter configurations to reduce workload.
    confs = shuffled(confs)[:100]

    for (batch_size, n_channel, img_shape, win_shape, stride, border_mode,
         method) in confs:
        if img_shape[0] < win_shape[0] or img_shape[1] < win_shape[1]:
            continue
        if border_mode != 'valid' and \
           (win_shape[0] != 1 or win_shape[1] != 1) and \
           ca._backend == 'cuda' and \
           ca.nnet.pool._default_impl == 'cudnn':
            # Bug: I think CUDArray/DeepPy calculates the padding in a manner
            # inconsistent with cuDNN
            continue
        if method == 'avg' and \
           ca._backend == 'cuda' and \
           ca.nnet.pool._default_impl == 'masked':
            # Not implemented yet
            continue
        print('Pool: batch_size=%i, n_channel=%i, img_shape=%s, win_shape=%s, '
              'stride=%s, border_mode=%s, method=%s'
              % (batch_size, n_channel, str(img_shape), str(win_shape),
                 str(stride), border_mode, method))

        x_shape = (batch_size, n_channel) + img_shape
        x = np.random.normal(size=x_shape).astype(ca.float_)
        layer = dp.Pool(win_shape=win_shape, method=method, strides=stride,
                        border_mode=border_mode)
        layer._setup(x_shape)
        y_img_shape = img_out_shape(img_shape, win_shape, stride, border_mode)
        assert layer.y_shape(x_shape) == (batch_size, n_channel) + y_img_shape

        check_grad(layer, x)
