import cudarray as ca
from ..base import Op
from .activation import Softmax
from .one_hot import OneHot


class Loss(Op):
    axis = None
    bcast_shape = None

    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        self.inputs = [pred, target]
        return self

    def setup_from_shape(self, in_shape):
        batch_size = in_shape[0]
        self.shape = (batch_size,)
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.ones(self.shape)
        self.axis = tuple(range(1, len(in_shape)))
        self.bcast_shape = tuple([batch_size, ] + [1, ]*len(self.axis))

    def setup(self):
        self.setup_from_shape(self.pred.shape)


class SquareError(Loss):
    def fprop(self):
        diff = self.pred.array - self.target.array
        diff **= 2.0
        ca.sum(diff, axis=self.axis, out=self.array)

    def bprop(self):
        ca.subtract(self.pred.array, self.target.array, self.pred.grad_array)
        self.pred.grad_array *= 2
        self.pred.grad_array *= ca.reshape(self.grad_array, self.bcast_shape)


class BinaryCrossEntropy(Loss):
    def __init__(self):
        self.eps = 1e-12

    def fprop(self):
        # -log(1 - pred)*(1 - target) - log(pred)*target
        tmp1 = 1 - self.pred.array
        tmp1 += self.eps
        ca.log(tmp1, tmp1)
        tmp2 = 1 - self.target.array
        ca.multiply(tmp1, tmp2, tmp1)
        ca.add(self.pred.array, self.eps, tmp2)
        ca.log(tmp2, tmp2)
        tmp2 *= self.target.array
        ca.add(tmp1, tmp2, tmp1)
        tmp1 *= -1
        ca.sum(tmp1, axis=1, out=self.array)

    def bprop(self):
        # -(target/pred - (1 - target)/(1 - pred))
        tmp1 = 1 - self.target.array
        tmp2 = 1 - self.pred.array
        tmp2 += self.eps
        ca.divide(tmp1, tmp2, tmp1)
        ca.add(self.pred.array, self.eps, tmp2)
        ca.divide(self.target.array, tmp2, out=tmp2)
        ca.subtract(tmp1, tmp2, self.pred.grad_array)
        self.pred.grad_array *= ca.reshape(self.grad_array, self.bcast_shape)


class _SoftmaxIdentityBProp(Softmax):
    def bprop(self):
        self.x.grad_array = self.grad_array


class SoftmaxCrossEntropy(Loss):
    def __init__(self, n_classes=None, eps=1e-8):
        self.n_classes = n_classes
        self.eps = eps

    def __call__(self, pred, target):
        if isinstance(pred, Softmax) and \
           not isinstance(pred, _SoftmaxIdentityBProp):
            pred = pred.x
        pred = _SoftmaxIdentityBProp()(pred)
        if self.n_classes is not None:
            target = OneHot(self.n_classes)(target)
        return super(SoftmaxCrossEntropy, self).__call__(pred, target)

    def fprop(self):
        # -target * log(pred)
        tmp1 = self.pred.array + self.eps
        ca.log(tmp1, tmp1)
        tmp1 *= self.target.array
        ca.sum(tmp1, axis=1, out=self.array)
        ca.negative(self.array, self.array)

    def bprop(self):
        ca.subtract(self.pred.array, self.target.array, self.pred.grad_array)
        self.pred.grad_array *= ca.reshape(self.grad_array, self.bcast_shape)
