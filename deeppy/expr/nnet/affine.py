import cudarray as ca
from ...base import ParamMixin
from ...parameter import Parameter
from ..base import Unary


class Linear(Unary, ParamMixin):
    def __init__(self, n_out, weights):
        self.n_out = n_out
        self.weights = Parameter.from_any(weights)

    def __call__(self, x):
        super(Linear, self).__call__(x)
        self.bpropable = True
        return self

    def setup(self):
        batch_size, n_in = self.x.shape
        self.shape = (batch_size, self.n_out)
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)
        self.weights.setup((n_in, self.n_out))

    def fprop(self):
        ca.dot(self.x.array, self.weights.array, self.array)

    def bprop(self):
        ca.dot(self.x.array.T, self.grad_array, self.weights.grad_array)
        ca.dot(self.grad_array, self.weights.array.T, self.x.grad_array)

    @property
    def params(self):
        return self.weights,

    @params.setter
    def params(self, params):
        self.weights, = params


class Affine(Linear):
    def __init__(self, n_out, weights, bias=0.0):
        super(Affine, self).__init__(n_out, weights)
        self.bias = Parameter.from_any(bias)

    def setup(self):
        super(Affine, self).setup()
        self.bias.setup((1, self.n_out))

    def fprop(self):
        super(Affine, self).fprop()
        self.array += self.bias.array

    def bprop(self):
        super(Affine, self).bprop()
        ca.sum(self.grad_array, axis=0, keepdims=True,
               out=self.bias.grad_array)

    @property
    def params(self):
        return self.weights, self.bias

    @params.setter
    def params(self, params):
        self.weights, self.bias = params
