import numpy as np
import cudarray as ca
import itertools
from .layers_seg import ParamMixin


class NeuralNetwork:
    def __init__(self, layers):
        self._initialized = False
        self.layers = layers
        self.bprop_until = next(idx for idx, layer in enumerate(layers)
                                if isinstance(layer, ParamMixin))

    def _setup(self, X, Y):
        # Setup layers sequentially
        X = np.reshape(X, X.shape[1:])
        Y = np.reshape(Y, Y.shape[1:])
        if self._initialized:
            return

        next_shape = X.shape
        indexing_shape = None

        for layer in self.layers:
            layer._setup(next_shape)
            next_shape = layer.output_shape(next_shape)
            indexing_shape = layer.output_index(indexing_shape)

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
        X = np.reshape(X, X.shape[1:])
        Y = np.reshape(Y, Y.shape[1:])
        # Forward propagation
        print "X"
        print X.shape
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next, 'train')
            print X_next.shape
        Y_pred = X_next

        # Back propagation of partial derivatives
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        layers = self.layers[self.bprop_until:-1]
        for layer in reversed(layers):
            next_grad = layer.bprop(next_grad)

        print "return"
        print self.layers[-1].loss(Y, Y_pred).shape
        return self.layers[-1].loss(Y, Y_pred)

    def _loss(self, X, Y):
        X = np.reshape(X, X.shape[1:])
        Y = np.reshape(Y, Y.shape[1:])
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next, 'test')
        Y_pred = X_next
        return self.layers[-1].loss(Y, Y_pred)

    def predict(self, X, batch_size=0):
        """ Calculate an output Y for the given input X. """
        if batch_size == 0:
            batch_size = X.shape[0]
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        Y_pred = []
        for b in range(n_batches):
            batch_begin = b * batch_size
            batch_end = batch_begin + batch_size
            X_next = ca.array(X[batch_begin:batch_end])
            for layer in self.layers[:-1]:
                X_next = layer.fprop(X_next, 'test')
            Y_pred_batch = self.layers[-1].predict(X_next)
            Y_pred.append(Y_pred_batch)
        Y_pred = np.concatenate(Y_pred)
        return Y_pred

    def error(self, X, Y, batch_size=0):
        X = np.reshape(X, X.shape[1:])
        Y = np.reshape(Y, Y.shape[1:])
        """ Calculate error on the given data. """
        Y_pred = self.predict(X, batch_size)
        error = Y_pred != Y
        return np.mean(error)
