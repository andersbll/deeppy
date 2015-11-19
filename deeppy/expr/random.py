import cudarray as ca
from .base import Expr, NoBPropMixin


class Normal(Expr, NoBPropMixin):
    bpropable = False

    def __init__(self, shape, mu=0.0, sigma=1.0):
        self.out_shape = shape
        self.mu = mu
        self.sigma = sigma
        self.out = ca.empty(self.out_shape)

    def fprop(self):
        self.out = ca.random.normal(loc=self.mu, scale=self.sigma,
                                    size=self.out_shape)


class Uniform(Expr, NoBPropMixin):
    bpropable = False

    def __init__(self, shape, low=0.0, high=1.0):
        self.out_shape = shape
        self.low = low
        self.high = high
        self.out = ca.empty(self.out_shape)

    def fprop(self):
        self.out = ca.random.uniform(low=self.low, high=self.high,
                                     size=self.out_shape)


def normal(loc=0, scale=1.0, size=None):
    return Normal(size, loc, scale)


def uniform(low=0, high=1.0, size=None):
    return Uniform(size, low, high)
