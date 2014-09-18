import numpy as np


class Filler(object):
    def __init__(self):
        raise NotImplementedError()

    def array(self, shape):
        raise NotImplementedError()


class ConstantFiller(Filler):
    def __init__(self, c=0.0):
        self.c = c

    def array(self, shape):
        return np.ones(shape)*self.c


class NormalFiller(Filler):
    def __init__(self, sigma=1.0, mu=0.0, rng=None):
        self.sigma = sigma
        self.mu = mu
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def array(self, shape):
        return self.rng.normal(loc=self.mu, scale=self.sigma, size=shape)


class UniformFiller(Filler):
    def __init__(self, low, high, rng=None):
        self.low = low
        self.high = high
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def array(self, shape):
        return self.rng.uniform(low=self.low, high=self.high, size=shape)


class CopyFiller(Filler):
    def __init__(self, array):
        self.arr = np.array(array)

    def array(self, shape):
        if self.arr.shape != shape:
            raise ValueError('Requested filler shape does not match.')
        return np.array(self.arr)


def filler(arg):
    if isinstance(arg, Filler):
        return arg
    elif isinstance(arg, (int, float)):
        return ConstantFiller(arg)
    elif isinstance(arg, np.ndarray):
        return CopyFiller(arg)
    elif isinstance(arg, tuple):
        if len(arg) == 2:
            if arg[0] == 'normal':
                return NormalFiller(**arg[1])
            elif arg[0] == 'uniform':
                return UniformFiller(**arg[1])
    raise ValueError('Invalid fillter arguments')
