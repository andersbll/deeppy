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
        self.array = ca.nnet.softmax(self.x.array)

    def bprop(self, y_grad):
        raise NotImplementedError()


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
