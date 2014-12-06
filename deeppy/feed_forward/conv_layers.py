import numpy as np
from .layers import Layer, ParamMixin
from ..base import parameter
import cudarray as ca


def padding(win_shape, border_mode):
    if border_mode == 'valid':
        return (0, 0)
    elif border_mode == 'same':
        return (win_shape[0]//2, win_shape[1]//2)
    elif border_mode == 'full':
        return (win_shape[0]-1, win_shape[1]-1)
    else:
        raise ValueError('invalid mode: "%s"' % border_mode)


class Convolutional(Layer, ParamMixin):
    def __init__(self, n_filters, filter_shape, weights, bias=0.0,
                 strides=(1, 1), border_mode='valid', weight_decay=0.0):
        self.name = 'conv'
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.W = parameter(weights)
        self.b = parameter(bias)
        pad = padding(filter_shape, border_mode)
        self.conv_op = ca.nnet.ConvBC01(pad, strides)

    def _setup(self, input_shape):
        n_channels = input_shape[1]
        W_shape = (self.n_filters, n_channels) + self.filter_shape
        b_shape = (1, self.n_filters, 1, 1)
        self.W._setup(W_shape)
        if not self.W.name:
            self.W.name = self.name + '_W'
        self.b._setup(b_shape)
        if not self.b.name:
            self.b.name = self.name + '_b'

    def fprop(self, x, phase):
        self.last_x = x
        convout = self.conv_op.fprop(x, self.W.array)
        return convout + self.b.array

    def bprop(self, y_grad):
        _, x_grad = self.conv_op.bprop(self.last_x, self.W.array,
                                       y_grad, filters_d=self.W.grad_array)
        ca.sum(ca.sum(y_grad, axis=(2, 3), keepdims=True), axis=0,
               keepdims=True, out=self.b.grad_array)
        return x_grad

    def params(self):
        return self.W, self.b

    def set_params(self, params):
        self.W, self.b = params

    def output_shape(self, input_shape):
        return self.conv_op.output_shape(input_shape, self.n_filters,
                                         self.filter_shape)


class Pool(Layer):
    def __init__(self, win_shape=(3, 3), method='max', strides=(1, 1),
                 border_mode='valid'):
        self.name = 'pool'
        pad = padding(win_shape, border_mode)
        self.pool_op = ca.nnet.PoolB01(win_shape, pad, strides, method)

    def fprop(self, x, phase):
        self.last_img_shape = x.shape[2:]
        poolout = self.pool_op.fprop(x)
        return poolout

    def bprop(self, y_grad):
        x_grad = self.pool_op.bprop(self.last_img_shape, y_grad)
        return x_grad

    def output_shape(self, input_shape):
        return self.pool_op.output_shape(input_shape)


class LocalResponseNormalization(Layer):
    def __init__(self, alpha=1e-4, beta=0.75, n=5, k=1):
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.k = k

    def fprop(self, input, phase):
        input = ca.lrnorm_bc01(input, N=self.n, alpha=self.alpha,
                               beta=self.beta, k=self.k)
        return input

    def bprop(self, Y_grad):
        return Y_grad

    def output_shape(self, input_shape):
        return input_shape


class Flatten(Layer):
    def fprop(self, x, phase):
        self.name = 'flatten'
        self.last_x_shape = x.shape
        return ca.reshape(x, self.output_shape(x.shape))

    def bprop(self, y_grad):
        return ca.reshape(y_grad, self.last_x_shape)

    def output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))
