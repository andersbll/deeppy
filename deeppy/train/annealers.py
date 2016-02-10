import numpy as np


class Annealer(object):
    def value(self, t):
        raise NotImplementedError()

    @classmethod
    def from_any(cls, arg):
        if arg is None:
            return ZeroAnnealer()
        elif isinstance(arg, Annealer):
            return arg
        raise ValueError('Invalid arguments.')


class ZeroAnnealer(Annealer):
    def __init__(self, val_start):
        self.val_start = val_start

    def value(self, t):
        return self.val_start


class DecayAnnealer(Annealer):
    def __init__(self, val_start, decay=0.1):
        self.val_start = val_start
        self.decay = decay

    def value(self, t):
        return self.val_start/(1.0 + self.decay*t)


class GammaAnnealer(Annealer):
    def __init__(self, val_start, val_stop, t_max, gamma=1.0):
        self.val_start = val_start
        self.val_stop = val_stop
        self.gamma = gamma
        self.t_max = t_max

    def value(self, t):
        if t > 0:
            t = t / float(self.t_max)
        val = self.val_start - t**(self.gamma)*(self.val_start - self.val_stop)
        if t > self.t_max and val <= 0.0 and self.val_stop > 0.0:
            raise ValueError('time > t_max and val has become negative')
        return val
