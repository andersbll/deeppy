import numpy as np
import cudarray as ca
import deeppy

from ..base import PickleMixin


def _require_expr(x):
    if isinstance(x, Expr):
        return x
    elif isinstance(x, ca.ndarray):
        return Constant(ca.array(x))
    else:
        return Constant(x)


class Expr(PickleMixin):
    _pickle_ignore = ['_tmp_', 'out', 'out_grad']
    out_shape = None
    out = None
    out_grad = None
    bpropable = True
    inputs = []

    def setup(self):
        pass

    def fprop(self):
        raise NotImplementedError()

    def bprop(self):
        raise NotImplementedError()

    def __add__(self, other):
        return deeppy.expr.Add()(self, other)

    def __radd__(self, other):
        return deeppy.expr.Add()(other, self)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return deeppy.expr.Subtract()(self, other)

    def __rsub__(self, other):
        return deeppy.expr.Subtract()(other, self)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return deeppy.expr.Multiply()(self, other)

    def __rmul__(self, other):
        return deeppy.expr.Multiply()(other, self)

    def __imul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return deeppy.expr.Divide()(self, other)

    def __rdiv__(self, other):
        return deeppy.expr.Divide()(other, self)

    def __idiv__(self, other):
        return self.__div__(other)

    def __truediv__(self, other):
        return deeppy.expr.Divide()(self, other)

    def __rtruediv__(self, other):
        return deeppy.expr.Divide()(other, self)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other):
        return deeppy.expr.Power()(self, other)

    def __rpow__(self, other):
        return deeppy.expr.Power()(other, self)

    def __ipow__(self, other):
        return self.__pow__(other)

    def __neg__(self):
        return deeppy.expr.Negative()(self)

    @property
    def T(self):
        return deeppy.expr.transpose(self)


class NoBPropMixin(object):
    def bprop(self):
        raise ValueError('NoBPropMixin should not bprop().')


class NoFPropMixin(object):
    def fprop(self):
        raise ValueError('NoFPropMixin should not bprop().')


class SplitMixin(object):
    pass


class Identity(Expr):
    def __call__(self, x):
        self.x = x
        self.inputs = [x]
        self.bpropable = x.bpropable
        return self

    def setup(self):
        self.out_shape = self.x.out_shape
        self.out = self.x.out
        if self.bpropable:
            self.out_grad = self.x.out_grad

    def fprop(self):
        self.out = self.x.out

    def bprop(self):
        self.x.out_grad = self.out_grad


class Constant(Expr, NoBPropMixin, NoFPropMixin):
    bpropable = False

    def __init__(self, value):
        if isinstance(value, np.ndarray):
            value = ca.array(value)
        self.value = value
        self.out = value
        if isinstance(value, (float, int)):
            self.out_shape = (1,)
        else:
            self.out_shape = self.out.shape


class Output(Expr, NoBPropMixin, NoFPropMixin):
    def __call__(self, x):
        self.inputs = [x]
        return self

    def setup(self):
        pass


class Unary(Expr):
    x = None

    def __call__(self, x):
        x = _require_expr(x)
        self.x = x
        if isinstance(x, Constant):
            # Propagate constant.
            self.setup()
            self.fprop()
            return Constant(self.out)
        self.bpropable = x.bpropable
        self.inputs = [x]
        return self


class UnaryElementWise(Unary):
    def setup(self):
        self.out_shape = self.x.out_shape
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)


class Binary(Expr):
    lhs = None
    rhs = None
    lhs_bprop = True
    rhs_bprop = True

    def __call__(self, lhs, rhs):
        lhs = _require_expr(lhs)
        rhs = _require_expr(rhs)
        self.lhs = lhs
        self.rhs = rhs
        if isinstance(lhs, Constant) and isinstance(rhs, Constant):
            # Propagate constant
            self.setup()
            self.fprop()
            return Constant(self.out)
        self.lhs_bprop = lhs.bpropable
        self.rhs_bprop = rhs.bpropable
        self.inputs = [lhs, rhs]
        return self


class Broadcast(Unary):
    def __init__(self, shape, broadcast_shape):
        self.out_shape = shape
        self.broadcast_shape = broadcast_shape
        # XXX: axis and keepdims are not determined correctly
        self.axis = []
        for axis, (a_dim, b_dim) in enumerate(zip(shape, broadcast_shape)):
            if a_dim != b_dim:
                self.axis.append(axis)
        self.axis = tuple(self.axis)
        self.keepdims = True

    def setup(self):
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.broadcast_shape)

    def fprop(self):
        self.out = self.x.out

    def bprop(self):
        ca.sum(self.out_grad, axis=self.axis, keepdims=self.keepdims,
               out=self.x.out_grad)


class BinaryElementWise(Binary):
    def setup(self):
        try:
            self.out_shape = np.add(np.zeros_like(self.lhs.out),
                                    np.zeros_like(self.rhs.out)).shape
            if self.lhs_bprop and np.prod(self.out_shape) > self.lhs.out.size:
                self.lhs = Broadcast(self.lhs.out_shape,
                                     self.out_shape)(self.lhs)
                self.inputs = [self.lhs, self.rhs]
                self.lhs.setup()
            if self.rhs_bprop and np.prod(self.out_shape) > self.rhs.out.size:
                self.rhs = Broadcast(self.rhs.out_shape,
                                     self.out_shape)(self.rhs)
                self.rhs_broadcast = True
                self.inputs = [self.lhs, self.rhs]
                self.rhs.setup()
        except ValueError:
            raise
            raise ValueError('Shape mismatch: %s and %s for %s. LHS: %s RHS: '
                             '%s.' % (self.lhs.out.shape, self.rhs.out.shape,
                                      self, self.lhs, self.rhs))
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)


class Source(Expr, NoBPropMixin, NoFPropMixin):
    bpropable = False

    def __init__(self, shape):
        self.out_shape = shape

    def setup(self):
        if not (isinstance(self.out, ca.ndarray)
                and self.out.shape == self.out_shape):
            self.out = ca.empty(self.out_shape)
        if not (isinstance(self.out_grad, ca.ndarray)
                and self.out_grad.shape == self.out_shape):
            self.out_grad = ca.empty(self.out_shape)


class Variable(Expr):
    def __init__(self, parameter):
        self.parameter = parameter

    def setup(self):
        self.out_shape = self.parameter.array.shape
        self.out = self.parameter.array
        self.out_grad = self.parameter.grad_array

    def fprop(self):
        self.out = self.parameter.array

    def bprop(self):
        ca.copyto(self.parameter.grad_array, self.out_grad)
