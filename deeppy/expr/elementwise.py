import numpy as np
import cudarray as ca

from .base import UnaryElementWise, BinaryElementWise


class Absolute(UnaryElementWise):
    def fprop(self):
        ca.fabs(self.x.out, out=self.out)

    def bprop(self):
        ca.nnet.relu_d(self.x.out, self.x.out_grad)
        self.x.out_grad *= 2.0
        self.x.out_grad -= 1.0
        ca.multiply(self.x.out_grad, self.out_grad, out=self.x.out_grad)


class Clip(UnaryElementWise):
    def __init__(self, a_min, a_max, keepgrads=True):
        self.a_min = a_min
        self.a_max = a_max
        self.keepgrads = keepgrads

    def fprop(self):
        ca.clip(self.x.out, self.a_min, self.a_max, out=self.out)

    def bprop(self):
        if self.keepgrads:
            self.x.out_grad = self.out_grad
        else:
            ca.multiply(self.out_grad, self.x.out > self.a_min,
                        self.x.out_grad)
            self.x.out_grad *= self.x.out < self.a_max


class Negative(UnaryElementWise):
    def fprop(self):
        ca.negative(self.x.out, out=self.out)

    def bprop(self):
        ca.negative(self.out_grad, out=self.x.out_grad)


class Log(UnaryElementWise):
    def fprop(self):
        ca.log(self.x.out, out=self.out)

    def bprop(self):
        ca.divide(1.0, self.x.out, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


class Exp(UnaryElementWise):
    def fprop(self):
        ca.exp(self.x.out, out=self.out)

    def bprop(self):
        ca.exp(self.x.out, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


class Tanh(UnaryElementWise):
    def fprop(self):
        ca.tanh(self.x.out, self.out)

    def bprop(self):
        ca.nnet.tanh_d(self.x.out, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


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
        ca.add(self.lhs.out, self.rhs.out, out=self.out)

    def bprop(self):
        if self.lhs_bprop:
            self.lhs.out_grad = self.out_grad
        if self.rhs_bprop:
            self.rhs.out_grad = self.out_grad


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
        ca.subtract(self.lhs.out, self.rhs.out, out=self.out)

    def bprop(self):
        if self.lhs_bprop:
            self.lhs.out_grad = self.out_grad
        if self.rhs_bprop:
            ca.negative(self.out_grad, out=self.rhs.out_grad)


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
        ca.multiply(self.lhs.out, self.rhs.out, out=self.out)

    def bprop(self):
        if self.lhs_bprop:
            ca.multiply(self.out_grad, self.rhs.out, out=self.lhs.out_grad)
        if self.rhs_bprop:
            ca.multiply(self.out_grad, self.lhs.out, out=self.rhs.out_grad)


class Divide(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            return 1.0
        if isinstance(rhs, np.ScalarType) and rhs == 1:
            return lhs
        return super(Divide, self).__call__(lhs, rhs)

    def fprop(self):
        ca.divide(self.lhs.out, self.rhs.out, out=self.out)

    def bprop(self):
        if self.lhs_bprop:
            ca.divide(self.out_grad, self.rhs.out, out=self.lhs.out_grad)
        if self.rhs_bprop:
            ca.multiply(self.out_grad, self.out, out=self.rhs.out_grad)
            self.rhs.out_grad /= self.rhs.out
            ca.negative(self.rhs.out_grad, out=self.rhs.out_grad)


class Power(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            raise NotImplementedError()
        return super(Power, self).__call__(lhs, rhs)

    def fprop(self):
        ca.power(self.lhs.out, self.rhs.out, out=self.out)

    def bprop(self):
        if self.lhs_bprop:
            tmp = self.rhs.out - 1
            ca.power(self.lhs.out, tmp, out=self.lhs.out_grad)
            self.lhs.out_grad *= self.rhs.out
            self.lhs.out_grad *= self.out_grad
        if self.rhs_bprop:
            ca.log(self.lhs.out, out=self.rhs.out_grad)
            self.rhs.out_grad *= self.out
            self.rhs.out_grad *= self.out_grad


class Maximum(BinaryElementWise):
    def __call__(self, lhs, rhs):
        if lhs is rhs:
            return lhs
        return super(Maximum, self).__call__(lhs, rhs)

    def fprop(self):
        ca.maximum(self.lhs.out, self.rhs.out, out=self.out)

    def bprop(self):
        if self.lhs_bprop:
            tmp = ca.equal(self.lhs.out, self.out)
            ca.multiply(self.out_grad, tmp, self.lhs.out_grad)
        if self.rhs_bprop:
            ca.equal(self.rhs.out, self.out, self.rhs.out_grad)
            self.rhs.out_grad *= self.out_grad


class Minimum(Maximum):
    # Inherits bprop from Maximum
    def fprop(self):
        ca.minimum(self.lhs.out, self.rhs.out, out=self.out)


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
