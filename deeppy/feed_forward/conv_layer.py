import numpy as np

from .layers import Layer, ParamMixin
from cudarray import conv_bc01, bprop_conv_bc01


class Convolutional(Layer, ParamMixin):
    def __init__(self, n_feats, filter_shape, strides, weight_scale,
                 weight_decay=0.0, padding_mode='same', border_mode='nearest'):
        self.n_feats = n_feats
        self.filter_shape = filter_shape
        self.strides = strides
        self.weight_scale = weight_scale
        self.weight_decay = weight_decay
        self.padding_mode = padding_mode
        self.border_mode = border_mode

    def _setup(self, input_shape, rng):
        n_channels = input_shape[1]
        W_shape = (n_channels, self.n_feats) + self.filter_shape
        self.W = rng.normal(size=W_shape, scale=self.weight_scale)
        self.b = np.zeros(self.n_feats)

    def fprop(self, input):
        self.last_input = input
        self.last_input_shape = input.shape
        convout = np.empty(self.output_shape(input.shape))
        conv_bc01(input, self.W, convout)
        return convout + self.b[np.newaxis, :, np.newaxis, np.newaxis]

    def bprop(self, output_grad):
        input_grad = np.empty(self.last_input_shape)
        self.dW = np.empty(self.W.shape)
        bprop_conv_bc01(self.last_input, output_grad, self.W, input_grad,
                        self.dW)
        n_imgs = output_grad.shape[0]
        self.db = np.sum(output_grad, axis=(0, 2, 3)) / (n_imgs)
        self.dW -= self.weight_decay*self.W
        return input_grad

    def params(self):
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def param_grads(self):
        # undo weight decay
        gW = self.dW+self.weight_decay*self.W
        return gW, self.db

    def output_shape(self, input_shape):
        if self.padding_mode == 'same':
            h = input_shape[2]
            w = input_shape[3]
        elif self.padding_mode == 'full':
            h = input_shape[2]-self.filter_shape[1]+1
            w = input_shape[3]-self.filter_shape[2]+1
        else:
            h = input_shape[2]+self.filter_shape[1]-1
            w = input_shape[3]+self.filter_shape[2]-1
        shape = (input_shape[0], self.n_feats, h, w)
        return shape
