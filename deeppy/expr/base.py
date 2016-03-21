import numpy as np
import cudarray as ca
import deeppy

from ..base import PickleMixin


def _require_op(x):
    if isinstance(x, Op):
        return x
    if isinstance(x, (np.ndarray, ca.ndarray, int, float)):
        return Constant(x)
    else:
        raise ValueError('Cannot process %s' % str(x))


class Op(PickleMixin):
    _pickle_ignore = ['_tmp_', 'array', 'grad_array']
    inputs = []
    bpropable = True
    shape = None
    array = None
    grad_array = None

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


class Identity(Op):
    def __call__(self, x):
        self.x = x
        self.inputs = [x]
        self.bpropable = x.bpropable
        return self

    def setup(self):
        self._shape = self.x.shape
        self.array = self.x.array
        if self.bpropable:
            self.grad_array = self.x.grad_array

    def fprop(self):
        self.array = self.x.array

    def bprop(self):
        self.x.grad_array = self.grad_array


class Source(Op, NoBPropMixin, NoFPropMixin):
    bpropable = False

    def __init__(self, shape):
        self.shape = shape
        self.array = ca.zeros(shape)

    @classmethod
    def from_array(cls, array):
        if isinstance(array, np.ndarray):
            array = ca.array(array)
        obj = cls(array.shape)
        obj.array = array
        return obj


class Constant(Op, NoBPropMixin, NoFPropMixin):
    bpropable = False

    def __init__(self, value):
        if isinstance(value, np.ndarray):
            value = ca.array(value)
        self.array = value
        if isinstance(value, (float, int)):
            self.shape = (1,)
        else:
            self.shape = self.array.shape


class Output(Op, NoBPropMixin, NoFPropMixin):
    def __call__(self, x):
        self.inputs = [x]
        return self

    def setup(self):
        pass


class Unary(Op):
    x = None

    def __call__(self, x):
        x = _require_op(x)
        self.x = x
        if isinstance(x, Constant):
            # Propagate constant.
            self.setup()
            self.fprop()
            return Constant(self.array)
        self.bpropable = x.bpropable
        self.inputs = [x]
        return self


class UnaryElementWise(Unary):
    def setup(self):
        self.shape = self.x.shape
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)


class Binary(Op):
    lhs = None
    rhs = None

    def __call__(self, lhs, rhs):
        lhs = _require_op(lhs)
        rhs = _require_op(rhs)
        self.lhs = lhs
        self.rhs = rhs
        if isinstance(lhs, Constant) and isinstance(rhs, Constant):
            # Propagate constant
            self.setup()
            self.fprop()
            return Constant(self.array)
        self.inputs = [lhs, rhs]
        return self


class Broadcast(Unary):
    def __init__(self, shape, broadcast_shape):
        self.shape = shape
        self.broadcast_shape = broadcast_shape
        # XXX: axis and keepdims are not determined correctly
        self.axis = []
        for axis, (a_dim, b_dim) in enumerate(zip(shape, broadcast_shape)):
            if a_dim != b_dim:
                self.axis.append(axis)
        self.axis = tuple(self.axis)
        self.keepdims = True

    def setup(self):
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.broadcast_shape)

    def fprop(self):
        self.array = self.x.array

    def bprop(self):
        ca.sum(self.grad_array, axis=self.axis, keepdims=self.keepdims,
               out=self.x.grad_array)


class BinaryElementWise(Binary):
    def __call__(self, lhs, rhs):
        super(BinaryElementWise, self).__call__(lhs, rhs)
        self.lhs_orig = self.lhs
        self.rhs_orig = self.rhs
        return self

    def setup(self):
        self.shape = np.add(np.zeros(self.lhs.shape),
                            np.zeros(self.rhs.shape)).shape
        size = np.prod(self.shape)
        if self.lhs_orig.bpropable and size > self.lhs_orig.array.size:
            self.lhs = Broadcast(self.lhs_orig.shape,
                                 self.shape)(self.lhs_orig)
            self.inputs[0] = self.lhs
            self.lhs.setup()
        if self.rhs_orig.bpropable and size > self.rhs_orig.array.size:
            self.rhs = Broadcast(self.rhs_orig.shape,
                                 self.shape)(self.rhs_orig)
            self.inputs[1] = self.rhs
            self.rhs.setup()
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)


class Variable(Op):
    def __init__(self, parameter):
        self.parameter = parameter

    def setup(self):
        self.shape = self.parameter.array.shape
        self.array = self.parameter.array
        self.grad_array = self.parameter.grad_array

    def fprop(self):
        self.array = self.parameter.array

    def bprop(self):
        ca.copyto(self.parameter.grad_array, self.grad_array)
