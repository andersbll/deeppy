import numpy as np
import cudarray as ca
from .base import Unary


class Reduce(Unary):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def setup(self):
        self.shape = ca.sum(self.x.array, axis=self.axis,
                            keepdims=self.keepdims).shape
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)


class Mean(Reduce):
    def setup(self):
        super(Mean, self).setup()
        self.n = np.prod(self.x.shape)
        self.scale = 1.0/np.prod(self.x.shape[self.axis])

    def fprop(self):
        ca.mean(self.x.array, axis=self.axis, out=self.array,
                keepdims=self.keepdims)

    def bprop(self):
        self.x.grad_array.fill(self.scale)
        self.x.grad_array *= self.grad_array


class Sum(Reduce):
    def fprop(self):
        ca.sum(self.x.array, axis=self.axis, out=self.array,
               keepdims=self.keepdims)

    def bprop(self):
        self.x.grad_array.fill(1.0)
        self.x.grad_array *= self.grad_array


def mean(x, axis=None, keepdims=False):
    return Mean(axis, keepdims)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
