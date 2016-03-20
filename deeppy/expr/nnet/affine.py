import cudarray as ca
from ...base import ParamMixin
from ...parameter import Parameter
from ..base import Unary


class Affine(Unary, ParamMixin):
    def __init__(self, n_out, weights, bias=0.0):
        self.n_out = n_out
        self.weights = Parameter.from_any(weights)
        if bias is not None:
            bias = Parameter.from_any(bias)
        self.bias = bias

    def __call__(self, x):
        super(Affine, self).__call__(x)
        self.bpropable = True
        return self

    def setup(self):
        x_shape = self.x.shape
        self.shape = (x_shape[0], self.n_out)
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)
        self.weights.setup((x_shape[1], self.n_out))
        if self.bias is not None:
            self.bias.setup(self.n_out)

    def fprop(self):
        ca.dot(self.x.array, self.weights.array, out=self.array)
        if self.bias is not None:
            self.array += self.bias.array

    def bprop(self):
        ca.dot(self.x.array.T, self.grad_array, out=self.weights.grad_array)
        ca.dot(self.grad_array, self.weights.array.T, out=self.x.grad_array)
        if self.bias is not None:
            ca.sum(self.grad_array, axis=0, out=self.bias.grad_array)

    @property
    def params(self):
        if self.bias is None:
            return self.weights,
        else:
            return self.weights, self.bias

    @params.setter
    def params(self, params):
        if self.bias is None:
            self.weights, = params
        else:
            self.weights, self.bias = params


class OneHot(Unary):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def setup(self):
        self.shape = self.x.shape + (self.n_classes,)
        self.array = ca.zeros(self.shape)

    def fprop(self):
        ca.nnet.one_hot_encode(self.x.array, self.n_classes, self.array)
