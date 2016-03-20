import numpy as np
import cudarray as ca
from ..base import Op
from .activation import Softmax
from .one_hot import OneHot
_FLT_MIN = np.finfo(ca.float_).tiny


class Loss(Op):
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        self.inputs = [pred, target]
        return self

    def setup(self):
        self.shape = (self.pred.shape[0], 1)
        self.array = ca.empty(self.shape)
        self.grad_array = ca.ones(self.shape)


class SquareError(Loss):
    def fprop(self):
        diff = self.pred.array - self.target.array
        diff **= 2.0
        ca.sum(diff, axis=1, keepdims=True, out=self.array)

    def bprop(self):
        ca.subtract(self.pred.array, self.target.array, self.pred.grad_array)
        self.pred.grad_array *= 2
        self.pred.grad_array *= self.grad_array


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
        ca.sum(tmp1, axis=1, keepdims=True, out=self.array)

    def bprop(self):
        # -(target/pred - (1 - target)/(1 - pred))
        tmp1 = 1 - self.target.array
        tmp2 = 1 - self.pred.array
        tmp2 += self.eps
        ca.divide(tmp1, tmp2, tmp1)
        ca.add(self.pred.array, self.eps, tmp2)
        ca.divide(self.target.array, tmp2, out=tmp2)
        ca.subtract(tmp1, tmp2, self.pred.grad_array)
        self.pred.grad_array *= self.grad_array


class SoftmaxCrossEntropy(Loss):
    class SoftmaxIdentityBProp(Softmax):
        def bprop(self):
            self.x.grad_array = self.grad_array

    def __init__(self, n_classes, one_hot_targets=True):
        self.n_classes = n_classes
        self.one_hot_targets = one_hot_targets

    def __call__(self, pred, target):
        if self.one_hot_targets:
            target = OneHot(self.n_classes)(target)
        return super(SoftmaxCrossEntropy, self).__call__(pred, target)

    def fprop(self):
        self.array = ca.nnet.categorical_cross_entropy(
            y_pred=self.pred.array, y_true=self.target.array
        )

    def bprop(self):
        self.pred.grad_array = -(self.target.array - self.pred.array)
