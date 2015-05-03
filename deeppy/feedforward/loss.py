import cudarray as ca
from ..base import PickleMixin


class Loss(PickleMixin):
    # abll: I suspect that this interface is not ideal. It would be more
    # elegant if Loss only provided loss() and grad(). However,
    #
    # - Some loss functions benefit from adding extra functionality.
    #   E.g. MultinomialLogReg that incorporates the softmax operation to
    #   allow for a simple and stable gradient expression.
    #
    # - Where should we place the logic from predict()?

    @classmethod
    def from_any(cls, arg):
        if isinstance(arg, Loss):
            return arg
        elif isinstance(arg, str):
            if arg == 'logreg':
                return MultinomialLogReg()
            elif arg == 'mse':
                return MeanSquaredError()
            elif arg == 'bce':
                return BinaryCrossEntropy()
        raise ValueError('Invalid constructor arguments: %s' % arg)

    def _setup(self, x_shape):
        pass

    def predict(self, x):
        return x

    def loss(self, target, x):
        """ Returns the loss calculated from the target and the prediction. """
        raise NotImplementedError()

    def grad(self, target, x):
        """ Returns the input gradient calculated from the target and the
        prediction. """
        raise NotImplementedError()

    def y_shape(self, x_shape):
        return x_shape


class MultinomialLogReg(Loss):
    """ Multinomial logistic regression with a cross-entropy loss function. """
    def __init__(self):
        self.name = 'logreg'
        self._tmp_last_x = None
        self._tmp_last_y = None
        self._tmp_last_target = None
        self._tmp_last_target_one_hot = None

    def _setup(self, x_shape):
        self.n_classes = x_shape[1]

    def _softmax(self, x):
        if self._tmp_last_x is not x:
            self._tmp_last_y = ca.nnet.softmax(x)
            self._tmp_last_x = x
        return self._tmp_last_y

    def _one_hot(self, target):
        if self._tmp_last_target is not target:
            self._target_one_hot = ca.nnet.one_hot_encode(target,
                                                          self.n_classes)
            self._tmp_last_target = target
        return self._target_one_hot

    def predict(self, x):
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


class MeanSquaredError(Loss):
    def __init__(self):
        self.name = 'mse'

    def _setup(self, x_shape):
        self.n_targets = x_shape[1]

    def grad(self, y, y_pred):
        return 2.0 / self.n_targets * (y_pred - y)

    def loss(self, y, y_pred):
        return ca.mean((y-y_pred)**2, axis=1)


class BinaryCrossEntropy(Loss):
    def __init__(self, eps=1e-10):
        self.name = 'bce'
        self.eps = eps

    def grad(self, y, y_pred):
        return -(y/(y_pred+self.eps) - (1-y)/(1-y_pred+self.eps))

    def loss(self, y, y_pred):
        return ca.mean(-ca.sum(y*ca.log(y_pred+self.eps) +
                               (1-y) * ca.log(1-y_pred+self.eps), axis=1))
