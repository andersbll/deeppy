import numpy as np
import cudarray as ca
from .layers import ParamMixin
from ..helpers import one_hot_encode, one_hot_decode


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def _setup(self, X, Y):
        # Setup layers sequentially
        next_shape = X.shape
        for layer in self.layers:
            layer._setup(next_shape)
            next_shape = layer.output_shape(next_shape)
        if next_shape != Y.shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, Y.shape))

    def fit(self, X, Y, learning_rate=0.1, max_iter=10, batch_size=64):
        """ Train network on the given data. """
        n_samples = Y.shape[0]
        n_batches = n_samples // batch_size
        Y_one_hot = one_hot_encode(Y)
        self._setup(X, Y_one_hot)
        iter = 0
        # Stochastic gradient descent with mini-batches
        while iter < max_iter:
            iter += 1
            for b in range(n_batches):
                batch_begin = b*batch_size
                batch_end = batch_begin+batch_size
                X_batch = ca.array(X[batch_begin:batch_end])
                Y_batch = ca.array(Y_one_hot[batch_begin:batch_end])

                # Forward propagation
                X_next = X_batch
                for layer in self.layers:
                    X_next = layer.fprop(X_next, 'train')
                Y_pred = X_next

                # Back propagation of partial derivatives
                next_grad = self.layers[-1].input_grad(Y_batch, Y_pred)
                for layer in reversed(self.layers[:-1]):
                    next_grad = layer.bprop(next_grad)

                # Update parameters
                for layer in self.layers:
                    if isinstance(layer, ParamMixin):
                        for param, inc in zip(layer.params(),
                                              layer.param_incs()):
                            param -= learning_rate*inc

            # Output training status
            loss = np.mean(self._loss(ca.array(X), ca.array(Y_one_hot)))
            error = self.error(X, Y)
            print('iter %i, loss %.4f, train error %.4f' % (iter, loss, error))

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
