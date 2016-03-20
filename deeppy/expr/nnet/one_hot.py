import cudarray as ca
from ..base import Unary
from ...base import int_


class OneHot(Unary):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def setup(self):
        self.shape = self.x.shape + (self.n_classes,)
        self.array = ca.empty(self.shape)

    def fprop(self):
        ca.nnet.one_hot_encode(self.x.array, self.n_classes, self.array)


class OneHotDecode(Unary):
    def setup(self):
        self.shape = self.x.shape[:1]
        self.array = ca.empty(self.shape, dtype=int_)

    def fprop(self):
        ca.nnet.one_hot_decode(self.x.array, self.array)
