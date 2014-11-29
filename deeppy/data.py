import numpy as np
import cudarray as ca


class Data(object):
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


class SupervisedData(Data):
    def __init__(self, x, y, batch_size=0):
        super(SupervisedData, self).__init__(x, batch_size)
        if x.shape[0] != y.shape[0]:
            raise ValueError('shape mismatch between x and y')
        self.y = y

    def batches(self):
        for batch_start, batch_stop in self._batch_slices():
            x_batch = ca.array(self.x[batch_start:batch_stop])
            y_batch = ca.array(self.y[batch_start:batch_stop])
            yield x_batch, y_batch

    @property
    def y_shape(self):
        return (self.batch_size,) + self.y.shape[1:]


def to_data(arg):
    if isinstance(arg, Data):
        return arg
    elif isinstance(arg, np.ndarray):
        return Data(arg)
    elif isinstance(arg, tuple):
        return SupervisedData(arg[0], arg[1])
    raise ValueError('Invalid data arguments')
