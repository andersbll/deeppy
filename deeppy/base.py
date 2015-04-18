import numpy as np
import cudarray as ca
import logging
from .fillers import filler, Filler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)


bool_ = ca.bool_
int_ = ca.int_
float_ = ca.float_


class Parameter(object):
    def __init__(self, fill, name='', learn_rate=1.0, weight_decay=0.0,
                 monitor=False):
        self.filler = filler(fill)
        self.name = name
        self.learn_rate = learn_rate
        self._monitor = monitor
        self.weight_decay = weight_decay
        self._array = None
        self._grad_array = None
        self._last_step = None
        self.shares = []

    @classmethod
    def from_any(cls, arg):
        if isinstance(arg, Parameter):
            return arg
        elif isinstance(arg, (int, float, np.ndarray, Filler)):
            return cls(arg)
        raise ValueError('Invalid parameter arguments')

    def _setup(self, shape):
        self._array = self.filler.array(shape)

    @property
    def array(self):
        return self._array

    @property
    def grad_array(self):
        ''' Returns the gradient array. '''
        if self._grad_array is None:
            self._grad_array = ca.empty_like(self.array)
        return self._grad_array

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
        if self._monitor:
            self._last_step = step
        self._array += step

    def penalty(self):
        if self.weight_decay == 0.0:
            return None
        else:
            return 2*self.weight_decay * self._array

    def monitor(self):
        if not self._monitor:
            return
        val_mean_abs = np.array(ca.mean(ca.fabs(self._array)))
        grad_mean_abs = np.array(ca.mean(ca.fabs(self._grad_array)))
        step_mean_abs = np.array(ca.mean(ca.fabs(self._last_step)))
        logger.info('%s:\t%.1e  [%.1e, %.1e]'
                    % (self.name, val_mean_abs, grad_mean_abs, step_mean_abs))

    def share(self):
        param = SharedParameter(self)
        self.shares.append(param)
        return param


class SharedParameter(Parameter):
    def __init__(self, parent):
        self.parent = parent
        self._grad_array = None

    def _setup(self, shape):
        pass

    @property
    def name(self):
        return self.parent.name

    @property
    def learn_rate(self):
        return self.parent.learn_rate

    @property
    def monitor(self):
        self.parent.monitor()

    @property
    def array(self):
        return self.parent.array

    def grad(self):
        raise RuntimeError('grad() should not be called for SharedParameter.')

    def share(self):
        return self.parent.share()
