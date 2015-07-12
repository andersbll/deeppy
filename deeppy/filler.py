import numpy as np
import cudarray as ca


class Filler(object):
    def __init__(self):
        raise NotImplementedError()

    @classmethod
    def from_any(cls, arg):
        if isinstance(arg, Filler):
            return arg
        elif isinstance(arg, (int, float)):
            return ConstantFiller(arg)
        elif isinstance(arg, (np.ndarray, ca.ndarray)):
            return CopyFiller(arg)
        elif isinstance(arg, tuple):
            if len(arg) == 2:
                if arg[0] == 'normal':
                    return NormalFiller(**arg[1])
                elif arg[0] == 'uniform':
                    return UniformFiller(**arg[1])
        raise ValueError('Invalid fillter arguments')

    def array(self, shape):
        raise NotImplementedError()


class ConstantFiller(Filler):
    def __init__(self, value=0.0):
        self.value = value

    def array(self, shape):
        return ca.ones(shape)*self.value


class NormalFiller(Filler):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

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
        self.arr = np_array

    def array(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        if self.arr.shape != shape:
            raise ValueError('Shape mismatch: expected %s but got %s'
                             % (str(self.arr.shape), str(shape)))
        return ca.array(self.arr)


class AutoFiller(Filler):
    def __init__(self, gain=1.0):
        self.gain = gain

    def array(self, shape):
        ndim = len(shape)
        if ndim == 2:
            # FullyConnected weights
            scale = 1.0 / np.sqrt(shape[0])
        elif ndim == 4:
            # Convolution filter
            scale = 1.0 / np.sqrt(np.prod(shape[1:]))
        else:
            raise ValueError('AutoFiller does not support ndim %i' % ndim)
        scale = self.gain * scale / np.sqrt(3)
        return ca.random.uniform(low=-scale, high=scale, size=shape)
