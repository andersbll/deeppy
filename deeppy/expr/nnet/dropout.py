import cudarray as ca
from ...base import PhaseMixin
from ..base import UnaryElementWise


class Dropout(UnaryElementWise, PhaseMixin):
    def __init__(self, dropout=0.5):
        self.dropout = dropout
        self._tmp_mask = None
        self.phase = 'train'

    def __call__(self, x):
        if self.dropout == 0.0:
            return x
        return super(Dropout, self).__call__(x)

    def setup(self):
        super(Dropout, self).setup()
        self.mask_shape = self.shape
        self._tmp_mask = ca.zeros(self.mask_shape, dtype=ca.int_)

    def fprop(self):
        if self.phase == 'train':
            ca.less(self.dropout, ca.random.uniform(size=self.mask_shape),
                    self._tmp_mask)
            ca.multiply(self.x.array, self._tmp_mask, self.array)
            self.array *= 1.0/(1.0-self.dropout)
        elif self.phase == 'test':
            self.array = self.x.array
        else:
            raise ValueError('Invalid phase: %s' % self.phase)

    def bprop(self):
        ca.multiply(self.grad_array, self._tmp_mask, self.x.grad_array)


class SpatialDropout(Dropout):
    def setup(self):
        super(SpatialDropout, self).setup()
        self.mask_shape = self.shape[:2] + (1, 1)
        self._tmp_mask = ca.zeros(self.mask_shape, dtype=ca.int_)
