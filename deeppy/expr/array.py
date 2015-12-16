import numpy as np
import cudarray as ca
from .base import Expr, Output, SplitMixin, Unary


class Reshape(Unary):
    def __init__(self, newshape):
        if not isinstance(newshape, tuple):
            if hasattr(newshape, '__iter__'):
                newshape = tuple(newshape)
            else:
                newshape = (newshape,)
        if not 0 <= list(newshape).count(-1) <= 1:
            raise ValueError('invalid newshape: %s' % newshape)
        self.newshape = newshape

    def setup(self):
        size = self.x.out.size
        newshape_size = np.prod(self.newshape)
        self.out_shape = tuple(d if d != -1 else -size//newshape_size
                               for d in self.newshape)
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        self.out = ca.reshape(self.x.out, self.out_shape)

    def bprop(self):
        self.x.out_grad = ca.reshape(self.out_grad, self.x.out_shape)


class Slices(Expr, SplitMixin):
    def __init__(self, splits):
        self.splits = splits

    def __call__(self, x):
        self.x = x
        self.inputs = [x]
        self.outputs = [Output()(self) for _ in range(len(self.splits)+1)]
        self.bpropable = x.bpropable
        return self.outputs

    def setup(self):
        splits = [0] + self.splits + [self.x.out_shape[0]]
        self.slices = []
        for i in range(len(splits) - 1):
            self.slices.append((splits[i], splits[i+1]))
        for i, (start, end) in enumerate(self.slices):
            out_shape = (end-start,) + self.x.out_shape[1:]
            self.outputs[i].out_shape = out_shape
            self.outputs[i].out = self.x.out[start:end, :]
            self.outputs[i].bpropable = self.bpropable
            if self.bpropable:
                self.outputs[i].out_grad = ca.zeros(out_shape)

    def fprop(self):
        for i, (start, end) in enumerate(self.slices):
            self.outputs[i].out = self.x.out[start:end, :]

    def bprop(self):
        for i, (start, end) in enumerate(self.slices):
            ca.copyto(self.x.out_grad[start:end, :], self.outputs[i].out_grad)


class Transpose(Unary):
    def __init__(self, contiguous=False):
        self.contiguous = contiguous

    def setup(self):
        if len(self.x.out_shape) == 1:
            self.out_shape = (1,) + self.x.out_shape
        elif len(self.x.out_shape) == 2:
            self.out_shape = tuple(reversed(self.x.out_shape))
        else:
            raise ValueError('invalid shape for transpose: %s'
                             % str(self.x.shape))
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        self.out = ca.transpose(self.x.out)
        if self.contiguous:
            self.out = ca.ascontiguousarray(self.out)

    def bprop(self):
        self.x.out_grad = ca.transpose(self.out_grad)
        if self.contiguous:
            self.out = ca.ascontiguousarray(self.x.out_grad)


class VSplit(Expr, SplitMixin):
    def __init__(self, len_x):
        self.n_splits = len_x

    def __call__(self, x):
        self.x = x
        self.inputs = [x]
        self.bpropable = x.bpropable
        self.outputs = [Output()(self) for i in range(self.n_splits)]
        return self.outputs

    def setup(self):
        out_shape = self.x.out_shape[1:]
        for i in range(self.n_splits):
            self.outputs[i].out_shape = out_shape
            self.outputs[i].out = self.x.out[i]
            self.outputs[i].out_grad = ca.empty(out_shape)

    def fprop(self):
        for i in range(self.n_splits):
            self.outputs[i].out = self.x.out[i]

    def bprop(self):
        for i in range(self.n_splits):
            ca.copyto(self.x.out_grad[i], self.outputs[i].out_grad)


class VStack(Expr):
    def __call__(self, *xs):
        self.n_sources = len(xs)
        self.inputs = xs
        return self

    def setup(self):
        shape = self.inputs[0].out_shape
        for expr in self.inputs:
            if shape != expr.out_shape:
                raise ValueError('shape mismatch: %s and %s'
                                 % (shape, expr.out_shape))
        self.out_shape = (self.n_sources,) + shape
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        for i in range(self.n_sources):
            ca.copyto(self.out[i], self.inputs[i].out)

    def bprop(self):
        for i in range(self.n_sources):
            ca.copyto(self.inputs[i].out_grad, self.out_grad[i])


class Concatenate(Expr):
    def __init__(self, axis):
        self.axis = axis
        self.a_size = -1

    def __call__(self, a, b):
        self.a = a
        self.b = b
        self.bpropable = a.bpropable or b.bpropable
        self.inputs = [a, b]
        return self

    def setup(self):
        a_shp = self.a.out_shape
        b_shp = self.b.out_shape
        concat_size = a_shp[self.axis] + b_shp[self.axis]
        self.a_size = a_shp[self.axis]
        self.out_shape = (a_shp[:self.axis] + (concat_size,) +
                          a_shp[self.axis+1:])
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        ca.extra.concatenate(self.a.out, self.b.out, axis=self.axis,
                             out=self.out)

    def bprop(self):
        ca.extra.split(self.out_grad, a_size=self.a_size, axis=self.axis,
                       out_a=self.a.out_grad, out_b=self.b.out_grad)


def transpose(x):
    return Transpose()(x)
