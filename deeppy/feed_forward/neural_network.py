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

    def _output_shape(self, input_shape):
        for layer in self.layers:
            input_shape = layer.output_shape(input_shape)
        return input_shape

    def predict(self, X, batch_size=0):
        """ Calculate an output Y for the given input X. """
        if batch_size == 0:
            batch_size = X.shape[0]
        n_samples = X.shape[0]
        n_batches = int(np.ceil(float(n_samples) / batch_size))
        Y = np.empty(self._output_shape(X.shape))
        for b in range(n_batches):
            batch_begin = min(b * batch_size, n_samples-batch_size)
            batch_end = batch_begin + batch_size
            X_next = ca.array(X[batch_begin:batch_end])
            for layer in self.layers[:-1]:
                X_next = layer.fprop(X_next, 'test')
            Y_batch = np.array(self.layers[-1].predict(X_next))
            Y[batch_begin:batch_end, ...] = Y_batch
        return Y

    def error(self, X, Y, batch_size=0):
        """ Calculate error on the given data. """
        Y_pred = self.predict(X, batch_size)
        error = Y_pred != Y
        return np.mean(error)
