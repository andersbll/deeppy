import cudarray as ca
from ..base import UnaryElementWise


class LeakyReLU(UnaryElementWise):
    def __init__(self, a=0.2):
        self.a = a

    def fprop(self):
        ca.minimum(self.x.array, 0, out=self.array)
        self.array *= self.a
        self.array += ca.maximum(self.x.array, 0)

    def bprop(self):
        self.x.grad_array = ca.less(self.x.array, 0) * self.a
        pos = ca.nnet.relu_d(self.x.array)
        self.x.grad_array += pos
        self.x.grad_array *= self.grad_array


class ReLU(UnaryElementWise):
    def fprop(self):
        ca.nnet.relu(self.x.array, self.array)

    def bprop(self):
        ca.nnet.relu_d(self.x.array, out=self.x.grad_array)
        self.x.grad_array *= self.grad_array


class Sigmoid(UnaryElementWise):
    def fprop(self):
        ca.nnet.sigmoid(self.x.array, self.array)

    def bprop(self):
        ca.nnet.sigmoid_d(self.x.array, out=self.x.grad_array)
        self.x.grad_array *= self.grad_array


class Softmax(UnaryElementWise):
    def fprop(self):
        # e_i = exp(x_i - max(x))
        # y = e_i / sum(e)
        tmp1 = ca.amax(self.x.array, axis=1, keepdims=True)
        ca.subtract(self.x.array, tmp1, self.array)
        ca.exp(self.array, self.array)
        ca.sum(self.array, axis=1, keepdims=True, out=tmp1)
        self.array /= tmp1

    def bprop(self):
        # y_i * (y_grad_i - sum(y_grad * y))
        ca.multiply(self.array, self.grad_array, self.x.grad_array)
        tmp1 = ca.sum(self.x.grad_array, axis=1, keepdims=True)
        ca.subtract(self.grad_array, tmp1, self.x.grad_array)
        self.x.grad_array *= self.array


class Softplus(UnaryElementWise):
    def fprop(self):
        ca.nnet.softplus(self.x.array, self.array)

    def bprop(self):
        ca.nnet.softplus_d(self.x.array, out=self.x.grad_array)
        self.x.grad_array *= self.grad_array


def leaky_relu(x):
    return LeakyReLU()(x)


def relu(x):
    return ReLU()(x)


def sigmoid(x):
    return Sigmoid()(x)


def softmax(x):
    return Softmax()(x)


def softplus(x):
    return Softplus()(x)
