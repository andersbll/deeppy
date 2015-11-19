import cudarray as ca
from ..base import Unary


class OneHot(Unary):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def setup(self):
        self.out_shape = self.x.out_shape + (self.n_classes,)
        self.out = ca.empty(self.out_shape)

    def fprop(self):
        ca.nnet.one_hot_encode(self.x.out, self.n_classes, self.out)
