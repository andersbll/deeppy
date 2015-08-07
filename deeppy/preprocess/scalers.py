import numpy as np


class StandardScaler(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self._x_mean = None
        self._x_std = None

    def fit(self, x):
        self._x_mean = np.mean(x, dtype=np.float64).astype(x.dtype)
        self._x_std = np.std(x, dtype=np.float64).astype(x.dtype)

    def fit_transform(self, x, copy=True):
        self.fit(x)
        return self.transform(x, copy)

    def transform(self, x, copy=True):
        if copy:
            x = np.copy(x)
        x -= self._x_mean
        x *= self.std / self._x_std
        x += self.mean
        return x

    def inverse_transform(self, x, copy=True):
        if copy:
            x = np.copy(x)
        x -= self.mean
        x /= self.std / self._x_std
        x += self._x_mean
        return x


class UniformScaler(object):
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high
        self._min = None
        self._max = None

    def fit(self, x):
        self._min = np.min(x)
        self._max = np.max(x)

    def fit_transform(self, x, copy=True):
        self.fit(x)
        return self.transform(x, copy)

    def transform(self, x, copy=True):
        if copy:
            x = np.copy(x)
        x -= self._min
        x *= (self.high - self.low) / (self._max - self._min)
        x += self.low
        return x

    def inverse_transform(self, x, copy=True):
        if copy:
            x = np.copy(x)
        x -= self.low
        x /= (self.high - self.low) / (self._max - self._min)
        x += self._min
        return x
