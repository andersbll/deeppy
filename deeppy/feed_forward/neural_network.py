import numpy as np
import cudarray as ca
import itertools
from .layers import ParamMixin


class NeuralNetwork:
    def __init__(self, layers):
        self._initialized = False
        self.layers = layers
        self.bprop_until = next(idx for idx, layer in enumerate(layers)
                                if isinstance(layer, ParamMixin))

    def _setup(self, X, Y):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = X.shape
        for layer in self.layers:
            layer._setup(next_shape)
            next_shape = layer.output_shape(next_shape)
        if next_shape != Y.shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, Y.shape))
        self._initialized = True

    def _params(self):
        all_params = [layer.params() for layer in self.layers
                      if isinstance(layer, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    def _bprop(self, X, Y):
        # Forward propagation
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next, 'train')
        Y_pred = X_next

        # Back propagation of partial derivatives
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        layers = self.layers[self.bprop_until:-1]
        for layer in reversed(layers):
            next_grad = layer.bprop(next_grad)

        return self.layers[-1].loss(Y, Y_pred)

    def _loss(self, X, Y):
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next, 'test')
        Y_pred = X_next
        return self.layers[-1].loss(Y, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = ca.array(X)
        for layer in self.layers[:-1]:
            X_next = layer.fprop(X_next, 'test')
        Y_pred = self.layers[-1].predict(X_next)
        return np.array(Y_pred)

    def error(self, X, Y, batch_size=128):
        """ Calculate error on the given data. """
        self._setup(X, Y)
        Y_pred = self.predict(X)
        error = Y_pred != Y
        return np.mean(error)
