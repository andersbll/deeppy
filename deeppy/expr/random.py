import cudarray as ca
from .base import Op, NoBPropMixin


class Normal(Op, NoBPropMixin):
    bpropable = False

    def __init__(self, shape, mu=0.0, sigma=1.0):
        self.shape = shape
        self.mu = mu
        self.sigma = sigma
        self.array = ca.empty(self.shape)

    def fprop(self):
        self.array = ca.random.normal(loc=self.mu, scale=self.sigma,
                                      size=self.shape)


class Uniform(Op, NoBPropMixin):
    bpropable = False

    def __init__(self, shape, low=0.0, high=1.0):
        self.shape = shape
        self.low = low
        self.high = high
        self.array = ca.empty(self.shape)

    def fprop(self):
        self.array = ca.random.uniform(low=self.low, high=self.high,
                                       size=self.shape)


def normal(loc=0, scale=1.0, size=None):
    return Normal(size, loc, scale)


def uniform(low=0, high=1.0, size=None):
    return Uniform(size, low, high)
