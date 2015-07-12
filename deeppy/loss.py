import numpy as np
import cudarray as ca
from .base import PickleMixin


_FLT_MIN = np.finfo(ca.float_).tiny


class Loss(PickleMixin):
    # abll: I suspect that this interface is not ideal. It would be more
    # elegant if Loss only provided loss() and grad(). However, where should
    # we place the logic from fprop()?

    @classmethod
    def from_any(cls, arg):
        if isinstance(arg, Loss):
            return arg
        elif isinstance(arg, str):
            if arg == 'softmaxce':
                return SoftmaxCrossEntropy()
            elif arg == 'bce':
                return BinaryCrossEntropy()
            elif arg == 'mse':
                return MeanSquaredError()
        raise ValueError('Invalid constructor arguments: %s' % arg)

    def _setup(self, x_shape):
        pass

    def fprop(self, x):
        return x

    def loss(self, target, x):
        """ Returns the loss calculated from the target and the input. """
        raise NotImplementedError()

    def grad(self, target, x):
        """ Returns the input gradient. """
        raise NotImplementedError()

    def y_shape(self, x_shape):
        return x_shape


class SoftmaxCrossEntropy(Loss):
    """
    Softmax + cross entropy (aka. multinomial logistic loss)
    """

    def __init__(self):
        self.name = 'softmaxce'
        self._tmp_x = None
        self._tmp_y = None
        self._tmp_target = None
        self._tmp_one_hot = None
        self.n_classes = None

    def _setup(self, x_shape):
        self.n_classes = x_shape[1]

    def _softmax(self, x):
        # caching wrapper
        if self._tmp_x is not x:
            self._tmp_y = ca.nnet.softmax(x)
            self._tmp_x = x
        return self._tmp_y

    def _one_hot(self, target):
        # caching wrapper
        if self._tmp_target is not target:
            self._tmp_one_hot = ca.nnet.one_hot_encode(target, self.n_classes)
            self._tmp_target = target
        return self._tmp_one_hot

    def fprop(self, x):
        return ca.nnet.one_hot_decode(self._softmax(x))

    def loss(self, target, x):
        y = self._softmax(x)
        target = self._one_hot(target)
        return ca.nnet.categorical_cross_entropy(y_pred=y, y_true=target)

    def grad(self, target, x):
        y = self._softmax(x)
        target = self._one_hot(target)
        return -(target - y)

    def y_shape(self, x_shape):
        return (x_shape[0],)


class BinaryCrossEntropy(Loss):
    def __init__(self):
        self.name = 'bce'

    def loss(self, y, y_pred):
        y_pred = ca.maximum(y_pred, _FLT_MIN)
        return -ca.mean(y*ca.log(y_pred) + (1 - y)*ca.log(1 - y_pred), axis=1)

    def grad(self, y, y_pred):
        y_pred = ca.maximum(y_pred, _FLT_MIN)
        return -(y/y_pred - (1-y)/(1-y_pred))


class MeanSquaredError(Loss):
    def __init__(self):
        self.name = 'mse'
        self.n_targets = None

    def _setup(self, x_shape):
        self.n_targets = x_shape[1]

    def loss(self, y, y_pred):
        return ca.mean((y-y_pred)**2, axis=1)

    def grad(self, y, y_pred):
        return 2.0 / self.n_targets * (y_pred - y)
