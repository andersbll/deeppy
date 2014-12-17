import numpy as np
import cudarray as ca
from .fillers import filler, Filler


bool_ = ca.bool_
int_ = ca.int_
float_ = ca.float_


class Parameter(object):
    def __init__(self, fill, name='', learn_rate=1.0, penalty=None,
                 norm=None, monitor=False):
        self.filler = filler(fill)
        self.name = name
        self.learn_rate = learn_rate
        self.monitor = monitor
        if penalty is None:
            self.penalty = None
        elif isinstance(penalty, tuple):
            if len(penalty) == 2:
                if penalty[0] == 'l2':
                    self._l2_penalty = 2*penalty[1]
                    self.penalty = self.l2_penalty
                else:
                    raise ValueError('invalid penalty type: %s' % penalty[0])
        if norm is None:
            self.norm_fun = None
        self.values = None
        self._grad = None
        self.shares = []

    def _setup(self, shape):
        self.values = self.filler.array(shape)

    @property
    def array(self):
        return self.values

    @property
    def grad_array(self):
        ''' Returns the gradient array. '''
        if self._grad is None:
            self._grad = ca.empty_like(self.array)
        return self._grad

    def grad(self):
        ''' Returns a parameter step calculated from the gradient.
        This differs from grad_array() as the parameter may be shared such
        that its gradient has multiple sources. '''
        grad = self.grad_array
        for param in self.shares:
            grad += param.grad_array
        return grad

    def step(self, step):
        ''' Update the parameter values according to the given step. '''
        self.values += step

    def l2_penalty(self):
        return self._l2_penalty * self.values

    def share(self):
        param = SharedParameter(self)
        self.shares.append(param)
        return param


class SharedParameter(Parameter):
    def __init__(self, parent):
        self.parent = parent
        self._grad = None

    def _setup(self, shape):
        raise RuntimeError('_setup() should not be called for SharedParameter')

    @property
    def name(self):
        return self.parent.name

    @property
    def learn_rate(self):
        return self.parent.learn_rate

    @property
    def monitor(self):
        return self.parent.monitor

    @property
    def array(self):
        return self.parent.array

    def grad(self):
        raise RuntimeError('grad() should not be called for SharedParameter.')

    def l2_penalty(self):
        return self.parent.l2_penalty()

    def share(self):
        return self.parent.share()


def parameter(arg):
    if isinstance(arg, Parameter):
        return arg
    elif isinstance(arg, (int, float, np.ndarray, Filler)):
        return Parameter(arg)
    raise ValueError('Invalid parameter arguments')
