import numpy as np
import cudarray as ca
from ..feed import Feed


class SiameseFeed(Feed):
    def __init__(self, x1, x2, batch_size=None, epoch_size=None):
        super(SiameseFeed, self).__init__(x1, batch_size, epoch_size)
        if x1.shape[0] != x2.shape[0]:
            raise ValueError('shape mismatch between x1 and x2')
        self.x2 = x2

    def batches(self):
        x1 = ca.empty(self.x_shape, dtype=self.x.dtype)
        x2 = ca.empty_like(x1)
        for start, stop in self._batch_slices():
            if stop > start:
                x1_np = self.x[start:stop]
                x2_np = self.x2[start:stop]
            else:
                x1_np = np.concatenate((self.x[start:], self.x[:stop]))
                x2_np = np.concatenate((self.x[start:], self.x[:stop]))
            ca.copyto(x1, x1_np)
            ca.copyto(x2, x2_np)
            yield x1, x2


class SupervisedSiameseFeed(SiameseFeed):
    def __init__(self, x1, x2, y, batch_size=None, epoch_size=None):
        super(SupervisedSiameseFeed, self).__init__(x1, x2, batch_size,
                                                    epoch_size)
        if x1.shape[0] != y.shape[0]:
            raise ValueError('shape mismatch between x and y')
        self.y = y

    def batches(self):
        x1 = ca.empty(self.x_shape, dtype=self.x.dtype)
        x2 = ca.empty_like(x1)
        y = ca.empty(self.y_shape, dtype=self.y.dtype)
        for start, stop in self._batch_slices():
            if stop > start:
                x1_np = self.x[start:stop]
                x2_np = self.x2[start:stop]
                y_np = self.y[start:stop]
            else:
                x1_np = np.concatenate((self.x[start:], self.x[:stop]))
                x2_np = np.concatenate((self.x[start:], self.x[:stop]))
                y_np = np.concatenate((self.y[start:], self.y[:stop]))
            ca.copyto(x1, x1_np)
            ca.copyto(x2, x2_np)
            ca.copyto(y, y_np)
            yield x1, x2, y

    @property
    def y_shape(self):
        return (self.batch_size,) + self.y.shape[1:]

    @property
    def shapes(self):
        return self.x_shape, self.x_shape, self.y_shape
