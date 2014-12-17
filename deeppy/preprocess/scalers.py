import numpy as np


class UniformScaler:
    def __init__(self, low=0.0, high=1.0, zero_mean=True, feature_wise=False):
        self.zero_mean = zero_mean
        self.feature_wise = feature_wise
        self._mean = None
        self._low = low
        self._high = high

    def fit(self, x):
        if self.zero_mean:
            if self.feature_wise:
                self._mean = np.mean(x, axis=0, keepdims=True)
            else:
                self._mean = np.mean(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        if self.zero_mean:
            x -= self._mean
        x /= self._high - self._low
        return x
