import numpy as np
import cudarray as ca

from .base import UnaryElementWise, BinaryElementWise


class Absolute(UnaryElementWise):
    def fprop(self):
        ca.fabs(self.x.array, out=self.array)

    def bprop(self):
        ca.nnet.relu_d(self.x.array, self.x.grad_array)
        self.x.grad_array *= 2.0
        self.x.grad_array -= 1.0
        ca.multiply(self.x.grad_array, self.grad_array, out=self.x.grad_array)


class Clip(UnaryElementWise):
    def __init__(self, a_min, a_max, keepgrads=True):
        self.a_min = a_min
        self.a_max = a_max
        self.keepgrads = keepgrads

    def fprop(self):
        ca.clip(self.x.array, self.a_min, self.a_max, out=self.array)

    def bprop(self):
        if self.keepgrads:
            self.x.grad_array = self.grad_array
        else:
            ca.multiply(self.grad_array, self.x.array > self.a_min,
                        self.x.grad_array)
            self.x.grad_array *= self.x.array < self.a_max


class Negative(UnaryElementWise):
    def fprop(self):
        ca.negative(self.x.array, out=self.array)

    def bprop(self):
        ca.negative(self.grad_array, out=self.x.grad_array)


class Log(UnaryElementWise):
    def fprop(self):
        ca.log(self.x.array, out=self.array)

    def bprop(self):
        ca.divide(1.0, self.x.array, out=self.x.grad_array)
        self.x.grad_array *= self.grad_array


class Exp(UnaryElementWise):
    def fprop(self):
        ca.exp(self.x.array, out=self.array)

    def bprop(self):
        ca.exp(self.x.array, out=self.x.grad_array)
        self.x.grad_array *= self.grad_array


class Tanh(UnaryElementWise):
    def fprop(self):
        ca.tanh(self.x.array, self.array)

    def bprop(self):
        ca.nnet.tanh_d(self.x.array, out=self.x.grad_array)
        self.x.grad_array *= self.grad_array


class Add(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            return 2*lhs
        if isinstance(lhs, np.ScalarType) and lhs == 0:
            return rhs
        if isinstance(rhs, np.ScalarType) and rhs == 0:
            return lhs
        return super(Add, self).__call__(lhs, rhs)

    def fprop(self):
        ca.add(self.lhs.array, self.rhs.array, out=self.array)

    def bprop(self):
        if self.lhs.bpropable:
            self.lhs.grad_array = self.grad_array
        if self.rhs.bpropable:
            self.rhs.grad_array = self.grad_array


class Subtract(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            return 0.0
        if isinstance(lhs, np.ScalarType) and lhs == 0:
            return -rhs
        if isinstance(rhs, np.ScalarType) and rhs == 0:
            return lhs
        return super(Subtract, self).__call__(lhs, rhs)

    def fprop(self):
        ca.subtract(self.lhs.array, self.rhs.array, out=self.array)

    def bprop(self):
        if self.lhs.bpropable:
            self.lhs.grad_array = self.grad_array
        if self.rhs.bpropable:
            ca.negative(self.grad_array, out=self.rhs.grad_array)


class Multiply(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            return lhs**2
        if isinstance(lhs, np.ScalarType) and lhs == 1:
            return rhs
        if isinstance(rhs, np.ScalarType) and rhs == 1:
            return lhs
        return super(Multiply, self).__call__(lhs, rhs)

    def fprop(self):
        ca.multiply(self.lhs.array, self.rhs.array, out=self.array)

    def bprop(self):
        if self.lhs.bpropable:
            ca.multiply(self.grad_array, self.rhs.array, self.lhs.grad_array)
        if self.rhs.bpropable:
            ca.multiply(self.grad_array, self.lhs.array, self.rhs.grad_array)


class Divide(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            return 1.0
        if isinstance(rhs, np.ScalarType) and rhs == 1:
            return lhs
        return super(Divide, self).__call__(lhs, rhs)

    def fprop(self):
        ca.divide(self.lhs.array, self.rhs.array, out=self.array)

    def bprop(self):
        if self.lhs.bpropable:
            ca.divide(self.grad_array, self.rhs.array, out=self.lhs.grad_array)
        if self.rhs.bpropable:
            ca.multiply(self.grad_array, self.array, out=self.rhs.grad_array)
            self.rhs.grad_array /= self.rhs.array
            ca.negative(self.rhs.grad_array, out=self.rhs.grad_array)


class Power(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            raise NotImplementedError()
        return super(Power, self).__call__(lhs, rhs)

    def fprop(self):
        ca.power(self.lhs.array, self.rhs.array, out=self.array)

    def bprop(self):
        if self.lhs.bpropable:
            tmp = self.rhs.array - 1
            ca.power(self.lhs.array, tmp, out=self.lhs.grad_array)
            self.lhs.grad_array *= self.rhs.array
            self.lhs.grad_array *= self.grad_array
        if self.rhs.bpropable:
            ca.log(self.lhs.array, out=self.rhs.grad_array)
            self.rhs.grad_array *= self.array
            self.rhs.grad_array *= self.grad_array


class Maximum(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            return lhs
        return super(Maximum, self).__call__(lhs, rhs)

    def fprop(self):
        ca.maximum(self.lhs.array, self.rhs.array, out=self.array)

    def bprop(self):
        if self.lhs.bpropable:
            tmp = ca.equal(self.lhs.array, self.array)
            ca.multiply(self.grad_array, tmp, self.lhs.grad_array)
        if self.rhs.bpropable:
            ca.equal(self.rhs.array, self.array, self.rhs.grad_array)
            self.rhs.grad_array *= self.grad_array


class Minimum(Maximum):
    # Inherits bprop from Maximum
    def fprop(self):
        ca.minimum(self.lhs.array, self.rhs.array, out=self.array)


def clip(a, a_min, a_max):
    return Clip(a_min, a_max)(a)


def absolute(x):
    return Absolute()(x)


fabs = absolute


def negative(x):
    return Negative()(x)


def exp(x):
    return Exp()(x)


def log(x):
    return Log()(x)


def add(lhs, rhs):
    return Add()(lhs, rhs)


def subtract(lhs, rhs):
    return Subtract()(lhs, rhs)


def maximum(lhs, rhs):
    return Maximum()(lhs, rhs)


def minimum(lhs, rhs):
    return Minimum()(lhs, rhs)


def multiply(lhs, rhs):
    return Multiply()(lhs, rhs)


def divide(lhs, rhs):
    return Divide()(lhs, rhs)


def power(lhs, rhs):
    return Power()(lhs, rhs)


def tanh(x):
    return Tanh()(x)
