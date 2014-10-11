import numpy as np
from .layers import Layer, ParamMixin
from ..fillers import filler
from ..base import Parameter
import cudarray as ca


def padding(win_shape, border_mode):
    if border_mode == 'valid':
        return (0, 0)
    elif border_mode == 'same':
        return (win_shape[0]//2, win_shape[1]//2)
    elif border_mode == 'full':
        return (win_shape[0]-1, win_shape[1]-1)
    else:
        raise ValueError('invalid mode: "%s"' % mode)


def convout_shape(img_shape, filter_shape, padding, strides):
    return ((img_shape[0] + 2*padding[0] - filter_shape[0])/strides[0] + 1,
            (img_shape[1] + 2*padding[1] - filter_shape[1])/strides[1] + 1)


class Convolutional(Layer, ParamMixin):
    def __init__(self, n_filters, filter_shape, weights, bias=0.0,
                 strides=(1, 1), border_mode='valid', weight_decay=0.0):
        self.name = 'conv'
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.strides = strides
        self.padding = padding(filter_shape, border_mode)
        self.weight_filler = filler(weights)
        self.bias_filler = filler(bias)
        self.weight_decay = weight_decay

    def _setup(self, input_shape):
        n_channels = input_shape[1]
        W_shape = (self.n_filters, n_channels) + self.filter_shape
        b_shape = (1, self.n_filters, 1, 1)
        self.W = ca.array(self.weight_filler.array(W_shape))
        self.b = ca.array(self.bias_filler.array(b_shape))
        self.W_grad = ca.empty_like(self.W)
        self.b_grad = ca.empty_like(self.b)
        if self.weight_decay > 0.0:
            def penalty_fun():
                return 2*self.weight_decay*self.W
        else:
            penalty_fun = None
        self.W_param = Parameter(self.W, gradient=self.W_grad, name='W',
                                 penalty_fun=penalty_fun, monitor=True)
        self.b_param = Parameter(self.b, gradient=self.b_grad, name='b')

    def fprop(self, input, phase):
        self.last_input = input
        self.last_input_shape = input.shape
        convout = ca.empty(self.output_shape(input.shape))
        ca.nnet.conv_bc01(input, self.W, padding=self.padding,
                          strides=self.strides, convout=convout)
        return convout + self.b

    def bprop(self, Y_grad):
        img_shape = self.last_input_shape[2:]
        input_grad = ca.empty(self.last_input_shape)
        ca.nnet.conv_bc01_bprop_filters(
            self.last_input, Y_grad, self.filter_shape, self.padding,
            self.strides, self.W_grad
        )
        ca.nnet.conv_bc01_bprop_imgs(self.W, Y_grad, img_shape, self.padding,
                                     self.strides, input_grad)
        ca.sum(ca.sum(Y_grad, axis=(2, 3), keepdims=True), axis=0,
               keepdims=True, out=self.b_grad)
        return input_grad

    def params(self):
        return self.W_param, self.b_param

    def output_shape(self, input_shape):
        b, _, img_h, img_w = input_shape
        out_shape = convout_shape((img_h, img_w), self.filter_shape,
                                  self.padding, self.strides)
        return (b, self.n_filters) + out_shape


class Pool(Layer):
    def __init__(self, win_shape=(3, 3), method='max', strides=(1, 1),
                 border_mode='valid'):
        self.name = 'pool'
        self.win_shape = win_shape
        self.strides = strides
        self.method = method
        self.padding = padding(win_shape, border_mode)

    def fprop(self, input, phase):
        self.last_input_shape = input.shape
        poolout, mask = ca.nnet.pool_b01(
            imgs=input, win_shape=self.win_shape, padding=self.padding,
            strides=self.strides, method=self.method
        )
        self.last_mask = mask
        return poolout

    def bprop(self, output_grad):
        input_grad = ca.nnet.pool_b01_bprop(
            output_grad, self.last_mask, self.last_input_shape[2:],
            padding=self.padding, win_shape=self.win_shape,
            strides=self.strides, method=self.method
        )
        return input_grad

    def output_shape(self, input_shape):
        b, c, img_h, img_w = input_shape
        out_shape = convout_shape((img_h, img_w), self.filter_shape,
                                  self.padding, self.strides)
        return (b, c) + out_shape


class Flatten(Layer):
    def fprop(self, input, phase):
        self.name = 'flatten'
        self.last_input_shape = input.shape
        return ca.reshape(input, self.output_shape(input.shape))

    def bprop(self, Y_grad):
        return ca.reshape(Y_grad, self.last_input_shape)

    def output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))
