import numpy as np
from .layers_seg import Layer_seg, ParamMixin_seg
from ..base import parameter
import cudarray as ca


def padding_seg(win_shape, border_mode):
    if border_mode == 'valid':
        return (0, 0)
    elif border_mode == 'same':
        return (win_shape[0]//2, win_shape[1]//2)
    elif border_mode == 'full':
        return (win_shape[0]-1, win_shape[1]-1)
    else:
        raise ValueError('invalid mode: "%s"' % mode)


class Convolutional_seg(Layer_seg, ParamMixin_seg):
    def __init__(self, n_filters, filter_shape, weights, bias=0.0,
                 border_mode='valid', weight_decay=0.0):
        self.name = 'conv'
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.W = parameter(weights)
        self.b = parameter(bias)
        pad = padding_seg(filter_shape, border_mode)
        self.conv_op = ca.nsnet.ConvBC01()
        self.indexing_shape = None

    def _setup(self, input_shape):
        self.input_shape = input_shape
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

    def bprop(self, y_grad, to_x=True):
        _, x_grad = self.conv_op.bprop(
            self.last_x, self.W.array, y_grad, to_imgs=to_x,
            filters_d=self.W.grad_array
        )
        ca.sum(ca.sum(y_grad, axis=(2, 3), keepdims=True), axis=0,
               keepdims=True, out=self.b.grad_array)
        return x_grad


    def params(self):
        return self.W, self.b

    def output_shape(self, input_shape):
        return self.conv_op.output_shape(input_shape, self.n_filters)

    def output_index(self, input_index):
        if input_index == None:
            if self.input_shape[0] != 1:
                raise ValueError('Must start with full image in one fragment')
            img_h, img_w = self.input_shape[-2:]
            input_index = np.arange(img_h*img_w, dtype=np.float)
            input_index = input_index.reshape((1,)+self.input_shape[-2:])

        return input_index

    def print_info(self):
        print self.name
        print "n_filters : %f" % self.n_filters
        print "filtershape :%s" % (self.filter_shape,)
        print "wight sigma : %f" % self.W.filler.sigma
        if self.W.penalty != None:
            print " L2 pen : %f" % self.W._l2_penalty

class Pool_seg(Layer_seg):
    def __init__(self, win_shape=(2, 2), strides=None):
        self.name = 'pool'
        self.pool_op = ca.nsnet.PoolB01(win_shape, strides)

    def fprop(self, x, phase):
        poolout = self.pool_op.fprop(x)
        return poolout

    def bprop(self, y_grad):
        x_grad = self.pool_op.bprop(y_grad)
        return x_grad

    def output_shape(self, input_shape):
        return self.pool_op.output_shape(input_shape)

    def output_index(self, input_index):
        return self.pool_op.output_index(input_index)

    def print_info(self):
        print self.name
        print "win_shape :%s" % (self.pool_op.win_shape,)
        print "strides : %s" % (self.pool_op.strides, )


class Flatten_seg(Layer_seg):
    def __init__(self, win_shape=(1, 1)):
        self.name = 'flatten'
        self.flatten_op = ca.nsnet.FlattenBC01(win_shape)

    def fprop(self, x, phase):
        flattenOut = self.flatten_op.fprop(x)
        return flattenOut

    def bprop(self, y_grad):
        x_grad = self.flatten_op.bprop(y_grad)
        return x_grad

    def output_shape(self, input_shape):
        return self.flatten_op.output_shape(input_shape)

    def output_index(self, input_index):
        self.flatten_op.output_index(input_index)
        return None

    def print_info(self):
        print self.name
        print "window shape :%s" % (self.flatten_op.win_shape,)
