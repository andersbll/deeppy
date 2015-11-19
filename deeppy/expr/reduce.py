import numpy as np
import cudarray as ca
from .base import Unary


class Reduce(Unary):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def setup(self):
        self.out = ca.sum(self.x.out, axis=self.axis, keepdims=self.keepdims)
        self.out_shape = self.out.shape
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)


class Mean(Reduce):
    def setup(self):
        super(Mean, self).setup()
        self.n = np.prod(self.x.out_shape)

    def fprop(self):
        ca.mean(self.x.out, axis=self.axis, out=self.out,
                keepdims=self.keepdims)

    def bprop(self):
        self.x.out_grad.fill(1.0/self.n)
        self.x.out_grad *= self.out_grad


class Sum(Reduce):
    def fprop(self):
        ca.sum(self.x.out, axis=self.axis, out=self.out,
               keepdims=self.keepdims)

    def bprop(self):
        self.x.out_grad.fill(1.0)
        self.x.out_grad *= self.out_grad


def mean(x, axis=None, keepdims=False):
    return Mean(axis, keepdims)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
