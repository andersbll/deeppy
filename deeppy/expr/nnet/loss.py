import numpy as np
import cudarray as ca
from ..base import Expr
from .activation import Softmax
from .one_hot import OneHot
_FLT_MIN = np.finfo(ca.float_).tiny


class Loss(Expr):
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        self.inputs = [pred, target]
        return self

    def setup(self):
        self.out_shape = (self.pred.out_shape[0], 1)
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.ones(self.out_shape)


class SquareError(Loss):
    def fprop(self):
        diff = self.pred.out - self.target.out
        diff **= 2.0
        ca.sum(diff, axis=1, keepdims=True, out=self.out)

    def bprop(self):
        ca.subtract(self.pred.out, self.target.out, self.pred.out_grad)
        self.pred.out_grad *= 2
        self.pred.out_grad *= self.out_grad


class BinaryCrossEntropy(Loss):
    def __init__(self):
        self.eps = 1e-12

    def fprop(self):
        # -log(1 - pred)*(1 - target) - log(pred)*target
        tmp1 = 1 - self.pred.out
        tmp1 += self.eps
        ca.log(tmp1, tmp1)
        tmp2 = 1 - self.target.out
        ca.multiply(tmp1, tmp2, tmp1)
        ca.add(self.pred.out, self.eps, tmp2)
        ca.log(tmp2, tmp2)
        tmp2 *= self.target.out
        ca.add(tmp1, tmp2, tmp1)
        tmp1 *= -1
        ca.sum(tmp1, axis=1, keepdims=True, out=self.out)

    def bprop(self):
        # -(target/pred - (1 - target)/(1 - pred))
        tmp1 = 1 - self.target.out
        tmp2 = 1 - self.pred.out
        tmp2 += self.eps
        ca.divide(tmp1, tmp2, tmp1)
        ca.add(self.pred.out, self.eps, tmp2)
        ca.divide(self.target.out, tmp2, out=tmp2)
        ca.subtract(tmp1, tmp2, self.pred.out_grad)
        self.pred.out_grad *= self.out_grad


class SoftmaxCrossEntropy(Loss):
    class SoftmaxIdentityBProp(Softmax):
        def bprop(self):
            self.x.out_grad = self.out_grad

    def __init__(self, n_classes, one_hot_targets=True):
        self.n_classes = n_classes
        self.one_hot_targets = one_hot_targets

    def __call__(self, pred, target):
        if self.one_hot_targets:
            target = OneHot(self.n_classes)(target)
        return super(SoftmaxCrossEntropy, self).__call__(pred, target)

    def fprop(self):
        self.out = ca.nnet.categorical_cross_entropy(y_pred=self.pred.out,
                                                     y_true=self.target.out)

    def bprop(self):
        self.pred.out_grad = -(self.target.out - self.pred.out)
