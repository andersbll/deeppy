import cudarray as ca
from ..base import UnaryElementWise


class LeakyReLU(UnaryElementWise):
    def __init__(self, a=0.2):
        self.a = a

    def fprop(self):
        ca.minimum(self.x.out, 0, out=self.out)
        self.out *= self.a
        self.out += ca.maximum(self.x.out, 0)

    def bprop(self):
        self.x.out_grad = ca.less(self.x.out, 0) * self.a
        pos = ca.nnet.relu_d(self.x.out)
        self.x.out_grad += pos
        self.x.out_grad *= self.out_grad


class ReLU(UnaryElementWise):
    def fprop(self):
        ca.nnet.relu(self.x.out, self.out)

    def bprop(self):
        ca.nnet.relu_d(self.x.out, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


class Sigmoid(UnaryElementWise):
    def fprop(self):
        ca.nnet.sigmoid(self.x.out, self.out)

    def bprop(self):
        ca.nnet.sigmoid_d(self.x.out, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


class Softmax(UnaryElementWise):
    def fprop(self):
        self.out = ca.nnet.softmax(self.x.out)

    def bprop(self, y_grad):
        raise NotImplementedError()


class Softplus(UnaryElementWise):
    def fprop(self):
        ca.nnet.softplus(self.x.out, self.out)

    def bprop(self):
        ca.nnet.softplus_d(self.x.out, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


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
