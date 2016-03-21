import numpy as np
import cudarray as ca
from .base import Op, Output, SplitMixin, Unary


class Flatten(Unary):
    def setup(self):
        shape = self.x.shape
        self.shape = (shape[0], np.prod(shape[1:]))
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)

    def fprop(self):
        self.array = ca.reshape(self.x.array, self.shape)

    def bprop(self):
        self.x.grad_array = ca.reshape(self.grad_array, self.x.shape)


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
        size = self.x.array.size
        newshape_size = np.prod(self.newshape)
        self.shape = tuple(d if d != -1 else -size//newshape_size
                           for d in self.newshape)
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)

    def fprop(self):
        self.array = ca.reshape(self.x.array, self.shape)

    def bprop(self):
        self.x.grad_array = ca.reshape(self.grad_array, self.x.shape)


class Slices(Op, SplitMixin):
    def __init__(self, splits):
        self.splits = splits
        self.outputs = [Output()(self) for _ in range(len(self.splits)+1)]

    def __call__(self, x):
        self.x = x
        self.inputs = [x]
        self.bpropable = x.bpropable
        for output in self.outputs:
            output.bpropable = self.bpropable
        return self.outputs

    def setup(self):
        splits = [0] + self.splits + [self.x.shape[0]]
        self.slices = []
        for i in range(len(splits) - 1):
            self.slices.append((splits[i], splits[i+1]))
        for i, (start, end) in enumerate(self.slices):
            shape = (end-start,) + self.x.shape[1:]
            self.outputs[i].shape = shape
            self.outputs[i].array = self.x.array[start:end]
            if self.bpropable:
                self.outputs[i].grad_array = ca.zeros(shape)

    def fprop(self):
        for i, (start, end) in enumerate(self.slices):
            self.outputs[i].array = self.x.array[start:end]

    def bprop(self):
        for i, (start, end) in enumerate(self.slices):
            ca.copyto(self.x.grad_array[start:end], self.outputs[i].grad_array)


class Transpose(Unary):
    def __init__(self, contiguous=False):
        self.contiguous = contiguous

    def setup(self):
        if len(self.x.shape) == 1:
            self.shape = (1,) + self.x.shape
        elif len(self.x.shape) == 2:
            self.shape = tuple(reversed(self.x.shape))
        else:
            raise ValueError('invalid shape for transpose: %s'
                             % str(self.x.shape))
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)

    def fprop(self):
        self.array = ca.transpose(self.x.array)
        if self.contiguous:
            self.array = ca.ascontiguousarray(self.array)

    def bprop(self):
        self.x.grad_array = ca.transpose(self.grad_array)
        if self.contiguous:
            self.array = ca.ascontiguousarray(self.x.grad_array)


class VSplit(Op, SplitMixin):
    def __init__(self, len_x):
        self.n_splits = len_x

    def __call__(self, x):
        self.x = x
        self.inputs = [x]
        self.bpropable = x.bpropable
        self.outputs = [Output()(self) for i in range(self.n_splits)]
        return self.outputs

    def setup(self):
        shape = self.x.shape[1:]
        for i in range(self.n_splits):
            self.outputs[i].shape = shape
            self.outputs[i].array = self.x.array[i]
            self.outputs[i].grad_array = ca.zeros(shape)

    def fprop(self):
        for i in range(self.n_splits):
            self.outputs[i].array = self.x.array[i]

    def bprop(self):
        for i in range(self.n_splits):
            ca.copyto(self.x.grad_array[i], self.outputs[i].grad_array)


class VStack(Op):
    def __call__(self, *xs):
        self.n_sources = len(xs)
        self.inputs = xs
        return self

    def setup(self):
        shape = self.inputs[0].shape
        for expr in self.inputs:
            if shape != expr.shape:
                raise ValueError('shape mismatch: %s and %s'
                                 % (shape, expr.shape))
        self.shape = (self.n_sources,) + shape
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)

    def fprop(self):
        for i in range(self.n_sources):
            ca.copyto(self.array[i], self.inputs[i].array)

    def bprop(self):
        for i in range(self.n_sources):
            ca.copyto(self.inputs[i].grad_array, self.grad_array[i])


class Concatenate(Op):
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
        a_shp = self.a.shape
        b_shp = self.b.shape
        concat_size = a_shp[self.axis] + b_shp[self.axis]
        self.a_size = a_shp[self.axis]
        self.shape = (a_shp[:self.axis] + (concat_size,) + a_shp[self.axis+1:])
        self.array = ca.zeros(self.shape)
        self.grad_array = ca.zeros(self.shape)

    def fprop(self):
        ca.extra.concatenate(self.a.array, self.b.array, axis=self.axis,
                             out=self.array)

    def bprop(self):
        ca.extra.split(self.grad_array, a_size=self.a_size, axis=self.axis,
                       out_a=self.a.grad_array, out_b=self.b.grad_array)


def transpose(x):
    return Transpose()(x)
