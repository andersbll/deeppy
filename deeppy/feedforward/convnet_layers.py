import numpy as np
from .layers import Layer
from ..base import ParamMixin
from ..parameter import Parameter
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


class Convolution(Layer, ParamMixin):
    def __init__(self, n_filters, filter_shape, weights, bias=0.0,
                 strides=(1, 1), border_mode='valid'):
        self.name = 'conv'
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.W = Parameter.from_any(weights)
        self.b = Parameter.from_any(bias)
        pad = padding(filter_shape, border_mode)
        self.conv_op = ca.nnet.ConvBC01(pad, strides)

    def _setup(self, x_shape):
        n_channels = x_shape[1]
        W_shape = (self.n_filters, n_channels) + self.filter_shape
        b_shape = (1, self.n_filters, 1, 1)
        self.W._setup(W_shape)
        if not self.W.name:
            self.W.name = self.name + '_W'
        self.b._setup(b_shape)
        if not self.b.name:
            self.b.name = self.name + '_b'

    def fprop(self, x, phase):
        self._tmp_last_x = x
        convout = self.conv_op.fprop(x, self.W.array)
        return convout + self.b.array

    def bprop(self, y_grad, to_x=True):
        _, x_grad = self.conv_op.bprop(
            self._tmp_last_x, self.W.array, y_grad, to_imgs=to_x,
            filters_d=self.W.grad_array
        )
        ca.sum(ca.sum(y_grad, axis=(2, 3), keepdims=True), axis=0,
               keepdims=True, out=self.b.grad_array)
        return x_grad

    @property
    def _params(self):
        return self.W, self.b

    @_params.setter
    def _params(self, params):
        self.W, self.b = params

    def y_shape(self, x_shape):
        return self.conv_op.output_shape(x_shape, self.n_filters,
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

    def bprop(self, y_grad, to_x=True):
        if to_x:
            x_grad = self.pool_op.bprop(self.last_img_shape, y_grad)
            return x_grad

    def y_shape(self, x_shape):
        return self.pool_op.output_shape(x_shape)


class LocalResponseNormalization(Layer):
    def __init__(self, alpha=1e-4, beta=0.75, n=5, k=1):
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.k = k

    def fprop(self, x, phase):
        x = ca.lrnorm_bc01(x, N=self.n, alpha=self.alpha, beta=self.beta,
                           k=self.k)
        return x

    def bprop(self, y_grad, to_x=True):
        return y_grad

    def y_shape(self, x_shape):
        return x_shape


class LocalContrastNormalization(Layer):
    @staticmethod
    def gaussian_kernel(sigma, size=None):
        if size is None:
            size = int(np.ceil(sigma*2.))
            if size % 2 == 0:
                size += 1
        xs = np.linspace(-size/2., size/2., size)
        kernel = 1/(np.sqrt(2*np.pi))*np.exp(-xs**2/(2*sigma**2))/sigma
        return kernel/np.sum(kernel)

    def __init__(self, kernel, eps=0.1, strides=(1, 1)):
        self.name = 'lcn'
        self.eps = eps
        if kernel.ndim == 1:
            kernel = np.outer(kernel, kernel)
        if kernel.shape[-2] % 2 == 0 or kernel.shape[-1] % 2 == 0:
            raise ValueError('only odd kernel sizes are supported')
        self.kernel = kernel
        pad = padding(kernel.shape[-2:], 'same')
        self.conv_op = ca.nnet.ConvBC01(pad, strides)

    def _setup(self, x_shape):
        n_channels = x_shape[1]
        if self.kernel.ndim == 2:
            self.kernel = np.repeat(self.kernel[np.newaxis, np.newaxis, ...],
                                    n_channels, axis=1)
        elif self.kernel.ndim == 3:
            self.kernel = self.kernel[np.newaxis, :]
        self.ca_kernel = ca.array(self.kernel)

    def fprop(self, x, phase):
        n_channels = x.shape[1]

        # Calculate local mean
        tmp = self.conv_op.fprop(x, self.ca_kernel)
        if n_channels > 1:
            ca.divide(tmp, n_channels, tmp)

        # Center input with local mean
        centered = ca.subtract(x, tmp)

        # Calculate local standard deviation
        tmp = ca.power(centered, 2)
        tmp = self.conv_op.fprop(tmp, self.ca_kernel)
        if n_channels > 1:
            ca.divide(tmp, n_channels, tmp)
        ca.sqrt(tmp, tmp)

        # Scale centered input with standard deviation
        return centered / (tmp + self.eps)

    def bprop(self, y_grad, to_x=True):
        raise NotImplementedError('LocalContrastNormalization supports only '
                                  'usage as a preprocessing layer.')

    def y_shape(self, x_shape):
        return x_shape


class Flatten(Layer):
    def __init__(self):
        self.name = 'flatten'

    def fprop(self, x, phase):
        self.last_x_shape = x.shape
        return ca.reshape(x, self.y_shape(x.shape))

    def bprop(self, y_grad, to_x=True):
        return ca.reshape(y_grad, self.last_x_shape)

    def y_shape(self, x_shape):
        return (x_shape[0], np.prod(x_shape[1:]))
