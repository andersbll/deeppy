import numpy as np
import cudarray as ca


class Feed(object):
    def __init__(self, x, batch_size=None, epoch_size=None):
        self.x = x
        self.n_samples = x.shape[0]
        if batch_size is None:
            batch_size = 128 if self.n_samples > 512 else self.n_samples
        if epoch_size is None:
            epoch_size = int(np.ceil(float(self.n_samples) / batch_size))
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.epoch_idx = 0

    @classmethod
    def from_any(cls, arg):
        if isinstance(arg, Feed):
            arg.reset()
            return arg
        elif isinstance(arg, np.ndarray):
            return cls(arg)
        elif isinstance(arg, tuple):
            return SupervisedFeed(arg[0], arg[1])
        raise ValueError('Invalid arguments.')

    def reset(self):
        self.epoch_idx = 0

    def _batch_slices(self):
        for b in range(self.epoch_size):
            start = (self.epoch_idx * self.epoch_size + b) * self.batch_size
            start = start % self.n_samples
            stop = (start + self.batch_size) % self.n_samples
            yield start, stop
        self.epoch_idx += 1

    def batches(self):
        x = ca.empty(self.x_shape, dtype=self.x.dtype)
        for start, stop in self._batch_slices():
            if stop > start:
                x_np = self.x[start:stop]
            else:
                x_np = np.concatenate((self.x[start:], self.x[:stop]))
            ca.copyto(x, x_np)
            yield {'x': x}

    @property
    def x_shape(self):
        return (self.batch_size,) + self.x.shape[1:]

    @property
    def shapes(self):
        return {'x_shape': self.x_shape}


class SupervisedFeed(Feed):
    def __init__(self, x, y, batch_size=None, epoch_size=None):
        super(SupervisedFeed, self).__init__(x, batch_size, epoch_size)
        if x.shape[0] != y.shape[0]:
            raise ValueError('shape mismatch between x and y')
        self.y = y

    def batches(self):
        x = ca.empty(self.x_shape, dtype=self.x.dtype)
        y = ca.empty(self.y_shape, dtype=self.y.dtype)
        for start, stop in self._batch_slices():
            if stop > start:
                x_np = self.x[start:stop]
                y_np = self.y[start:stop]
            else:
                x_np = np.concatenate((self.x[start:], self.x[:stop]))
                y_np = np.concatenate((self.y[start:], self.y[:stop]))
            ca.copyto(x, x_np)
            ca.copyto(y, y_np)
            yield {'x': x, 'y': y}

    @property
    def y_shape(self):
        return (self.batch_size,) + self.y.shape[1:]

    @property
    def shapes(self):
        return {'x_shape': self.x_shape, 'y_shape': self.y_shape}
