
import cudarray as ca
from ..base import ParamMixin
from ..parameter import Parameter
from .layers import Layer


class Activation(Layer):
    _tmp_x = None

    @classmethod
    def from_any(cls, arg):
        if isinstance(arg, Activation):
            return arg
        if isinstance(arg, tuple):
            method, args = arg
        else:
            method = arg
            args = ()
        if method == 'leaky_relu':
            return LeakyReLU(*args)
        if method == 'parametric_relu':
            return ParametricReLU(*args)
        if method == 'relu':
            return ReLU(*args)
        if method == 'sigmoid':
            return Sigmoid(*args)
        if method == 'softmax':
            return Softmax(*args)
        if method == 'softplus':
            return Softplus(*args)
        if method == 'tanh':
            return Tanh(*args)
        raise ValueError('Invalid input arguments')

    def y_shape(self, x_shape):
        return x_shape


class LeakyReLU(Activation):
    def __init__(self, a=0.25):
        self.a = a

    def fprop(self, x):
        self._tmp_x = x
        pos = ca.maximum(x, 0)
        neg = self.a * ca.minimum(x, 0)
        return pos + neg

    def bprop(self, y_grad):
        pos = ca.nnet.relu_d(self._tmp_x)
        neg = self.a * (self._tmp_x < 0)
        return (pos + neg) * y_grad


class ParametricReLU(Activation, ParamMixin):
    def __init__(self, a=0.25):
        self.a = Parameter.from_any(a)

    def setup(self, x_shape):
        self.a.setup((1, 1))

    @property
    def params(self):
        return [self.a]

    @params.setter
    def params(self, params):
        self.a = params[0]

    def fprop(self, x):
        self._tmp_x = x
        pos = ca.maximum(x, 0)
        neg = self.a.array * ca.minimum(x, 0)
        return pos + neg

    def bprop(self, y_grad):
        pos = ca.nnet.relu_d(self._tmp_x)
        neg_mask = self._tmp_x < 0
        a_grad = neg_mask * self._tmp_x * y_grad
        ca.sum(a_grad, keepdims=True, out=self.a.grad_array)
        return (pos + self.a.array * neg_mask) * y_grad


class ReLU(Activation):
    def fprop(self, x):
        self._tmp_x = x
        return ca.nnet.relu(x)

    def bprop(self, y_grad):
        ca.nnet.relu_d(self._tmp_x, self._tmp_x)
        return self._tmp_x * y_grad


class Sigmoid(Activation):
    def fprop(self, x):
        self._tmp_x = x
        return ca.nnet.sigmoid(x)

    def bprop(self, y_grad):
        ca.nnet.sigmoid_d(self._tmp_x, self._tmp_x)
        return self._tmp_x * y_grad


class Softmax(Activation):
    def fprop(self, x):
        self._tmp_x = x
        return ca.nnet.softmax(x)

    def bprop(self, y_grad):
        raise NotImplementedError()


class Softplus(Activation):
    def fprop(self, x):
        self._tmp_x = x
        return ca.log(1.0 + ca.exp(x))

    def bprop(self, y_grad):
        return 1.0/(1.0 + ca.exp(-self._tmp_x)) * y_grad


class Tanh(Activation):
    def fprop(self, x):
        self._tmp_x = x
        return ca.tanh(x)

    def bprop(self, y_grad):
        ca.nnet.tanh_d(self._tmp_x, self._tmp_x)
        return self._tmp_x * y_grad
