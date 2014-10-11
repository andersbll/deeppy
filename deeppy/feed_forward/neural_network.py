import numpy as np
import cudarray as ca
import itertools
from .layers import ParamMixin
from ..helpers import one_hot_encode, one_hot_decode


class NeuralNetwork:
    def __init__(self, layers):
        self._initialized = False
        self.layers = layers

    def _setup(self, X, Y):
        if self._initialized:
            return
        # Setup layers sequentially
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
        for layer in reversed(self.layers[:-1]):
            next_grad = layer.bprop(next_grad)

        return self.layers[-1].loss(Y, Y_pred)

    def _loss(self, X, Y_one_hot):
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next, 'test')
        Y_pred = X_next
        return self.layers[-1].loss(Y_one_hot, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = ca.array(X)
        for layer in self.layers:
            X_next = layer.fprop(X_next, 'test')
        Y_pred = np.array(X_next)
        Y_pred = one_hot_decode(Y_pred)
        return Y_pred

    def error(self, X, Y):
        """ Calculate error on the given data. """
        Y_pred = self.predict(X)
        error = Y_pred != Y
        return np.mean(error)
