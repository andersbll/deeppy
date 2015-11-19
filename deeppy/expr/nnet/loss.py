import numpy as np
import cudarray as ca
from ..base import Expr, NoBPropMixin
from .activation import Softmax
from .one_hot import OneHot
_FLT_MIN = np.finfo(ca.float_).tiny


class BinaryCrossEntropy(Expr):
    def __init__(self, clip=True):
        self.clip = clip

    def __call__(self, x, target):
        self.x = x
        self.target = target
        self.inputs = [x, target]
        return self

    def setup(self):
        self.out_shape = (1,)
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        pred = self.x.out
        target = self.target.out
        if self.clip:
            ca.clip(pred, _FLT_MIN, .9999999, pred)
        self.out = -ca.sum(target*ca.log(pred) + (1 - target)*ca.log(1 - pred))

    def bprop(self):
        self.grad()
        self.x.out_grad *= self.out_grad

    def grad(self):
        pred = self.x.out
        target = self.target.out
        if self.clip:
            ca.clip(pred, _FLT_MIN, .9999999, pred)
        self.x.out_grad = -(target/pred - (1-target)/(1-pred))


class SoftmaxCrossEntropy(Expr):
    class SoftmaxIdentityBProp(Softmax, NoBPropMixin):
        def bprop(self):
            self.x.out_grad = self.out_grad

    def __init__(self, n_classes, one_hot_encode=True):
        self.n_classes = n_classes
        self.one_hot_encode = one_hot_encode

    def __call__(self, pred, target):
        pred = self.SoftmaxIdentityBProp()(pred)
        if self.one_hot_encode:
            target = OneHot(self.n_classes)(target)
        self.pred = pred
        self.target = target
        self.inputs = [pred, target]
        return self

    def setup(self):
        self.out_shape = (self.pred.out_shape[0],)
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        self.out = ca.nnet.categorical_cross_entropy(y_pred=self.pred.out,
                                                     y_true=self.target.out)

    def bprop(self):
        self.grad()
        self.pred.out_grad *= self.out_grad

    def grad(self):
        self.pred.out_grad = -(self.target.out - self.pred.out)
        self.pred.out_grad *= self.out_grad
