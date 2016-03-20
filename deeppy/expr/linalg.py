import cudarray as ca
from .base import Binary


class Dot(Binary):
    def setup(self):
        try:
            # XXX: don't be lazy
            self.shape = ca.dot(self.lhs.array, self.rhs.array).shape
        except ValueError:
            raise ValueError('Shape mismatch: %s and %s for %s. LHS: %s RHS: '
                             '%s.' % (self.lhs.shape, self.rhs.shape,
                                      self, self.lhs, self.rhs))
        self.array = ca.empty(self.shape)
        self.grad_array = ca.empty(self.shape)

    def fprop(self):
        ca.dot(self.lhs.array, self.rhs.array, out=self.array)

    def bprop(self):
        if self.lhs_bprop:
            ca.dot(self.grad_array, self.rhs.array.T, out=self.lhs.grad_array)
        if self.rhs_bprop:
            ca.dot(self.lhs.array.T, self.grad_array, out=self.rhs.grad_array)


def dot(lhs, rhs):
    return Dot()(lhs, rhs)
