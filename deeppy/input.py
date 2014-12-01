import numpy as np
import cudarray as ca


class Input(object):
    def __init__(self, x, batch_size=0):
        self.x = x
        self.batch_size = batch_size if batch_size > 0 else x.shape[0]
        self.n_samples = x.shape[0]
        self.n_batches = int(np.ceil(float(self.n_samples) / self.batch_size))

    def _batch_slices(self):
        for b in range(self.n_batches):
            batch_start = b * self.batch_size
            batch_stop = min(self.n_samples, batch_start + self.batch_size)
            yield batch_start, batch_stop

    def batches(self):
        for batch_start, batch_stop in self._batch_slices():
            yield ca.array(self.x[batch_start:batch_stop])

    @property
    def x_shape(self):
        return (self.batch_size,) + self.x.shape[1:]


class SupervisedMixin(object):
    def supervised_batches(self):
        raise NotImplementedError()

    @property
    def y_shape(self):
        raise NotImplementedError()


class SupervisedInput(Input, SupervisedMixin):
    def __init__(self, x, y, batch_size=0):
        super(SupervisedInput, self).__init__(x, batch_size)
        if x.shape[0] != y.shape[0]:
            raise ValueError('shape mismatch between x and y')
        self.y = y

    def supervised_batches(self):
        for batch_start, batch_stop in self._batch_slices():
            x_batch = ca.array(self.x[batch_start:batch_stop])
            y_batch = ca.array(self.y[batch_start:batch_stop])
            yield x_batch, y_batch

    @property
    def y_shape(self):
        return (self.batch_size,) + self.y.shape[1:]


def to_input(arg):
    if isinstance(arg, Input):
        return arg
    elif isinstance(arg, np.ndarray):
        return Input(arg)
    elif isinstance(arg, tuple):
        return SupervisedInput(arg[0], arg[1])
    raise ValueError('Invalid input arguments')
