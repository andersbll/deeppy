import cudarray as ca
from .base import Binary


class Dot(Binary):
    def setup(self):
        try:
            # XXX: don't be lazy
            self.out_shape = ca.dot(self.lhs.out, self.rhs.out).shape
        except ValueError:
            raise ValueError('Shape mismatch: %s and %s for %s. LHS: %s RHS: '
                             '%s.' % (self.lhs.out.shape, self.rhs.out.shape,
                                      self, self.lhs, self.rhs))
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        ca.dot(self.lhs.out, self.rhs.out, out=self.out)

    def bprop(self):
        if self.lhs_bprop:
            ca.dot(self.out_grad, self.rhs.out.T, out=self.lhs.out_grad)
        if self.rhs_bprop:
            ca.dot(self.lhs.out.T, self.out_grad, out=self.rhs.out_grad)


def dot(lhs, rhs):
    return Dot()(lhs, rhs)
