import numpy as np
import cudarray as ca
from .fillers import filler, Filler


class Parameter(object):
    def __init__(self, fill, name='', learn_rate=1.0, penalty=None,
                 norm=None, monitor=False):
        self.filler = filler(fill)
        self.name = name
        self.learn_rate = learn_rate
        self.monitor = monitor
        if penalty is None:
            self.penalty_fun = None
        elif isinstance(penalty, tuple):
            if len(penalty) == 2:
                if penalty[0] == 'l2':
                    self.penalty = 2*penalty[1]
                    self.penalty_fun = self.l2_penalty
                else:
                    raise ValueError('invalid penalty type: %s' % arg[0])
        if norm is None:
            self.norm_fun = None
        self.values = None
        self._grad = None

    def _setup(self, shape):
        self.values = self.filler.array(shape)

    @property
    def grad(self):
        if self._grad is None:
            self._grad = ca.empty_like(self.values)
        return self._grad

    def l2_penalty(self):
        return self.penalty*self.values


def parameter(arg):
    if isinstance(arg, Parameter):
        return arg
    elif isinstance(arg, (int, float, np.ndarray, Filler)):
        return Parameter(arg)
    raise ValueError('Invalid parameter arguments')
