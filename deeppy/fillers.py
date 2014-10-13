import numpy as np
import cudarray as ca


class Filler(object):
    def __init__(self):
        raise NotImplementedError()

    def array(self, shape):
        raise NotImplementedError()


class ConstantFiller(Filler):
    def __init__(self, c=0.0):
        self.c = c

    def array(self, shape):
        return ca.ones(shape)*self.c


class NormalFiller(Filler):
    def __init__(self, sigma=1.0, mu=0.0):
        self.sigma = sigma
        self.mu = mu

    def array(self, shape):
        return ca.random.normal(loc=self.mu, scale=self.sigma, size=shape)


class UniformFiller(Filler):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def array(self, shape):
        return ca.random.uniform(low=self.low, high=self.high, size=shape)


class CopyFiller(Filler):
    def __init__(self, np_array):
        self.arr = array

    def array(self, shape):
        if self.arr.shape != shape:
            raise ValueError('Requested filler shape does not match.')
        return ca.array(self.arr)


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
