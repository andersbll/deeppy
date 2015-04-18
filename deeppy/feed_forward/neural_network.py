import numpy as np
import itertools
from .layers import ParamMixin
from ..input import Input


class NeuralNetwork:
    def __init__(self, layers):
        self._initialized = False
        self.layers = layers
        self.bprop_until = next((idx for idx, l in enumerate(self.layers)
                                 if isinstance(l, ParamMixin)),
                                len(self.layers))

    def _setup(self, input):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = input.x_shape
        for layer in self.layers:
            layer._setup(next_shape)
            next_shape = layer.output_shape(next_shape)
        if next_shape != input.y_shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, input.y_shape))
        self._initialized = True

    def _params(self):
        all_params = [layer.params() for layer in self.layers
                      if isinstance(layer, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    def _update(self, batch):
        # Forward propagation
        x, y = batch
        x_next = x
        for layer in self.layers:
            x_next = layer.fprop(x_next, 'train')
        y_pred = x_next

        # Back propagation of partial derivatives
        next_grad = self.layers[-1].input_grad(y, y_pred)
        layers = self.layers[self.bprop_until:-1]
        for layer in reversed(layers[1:]):
            next_grad = layer.bprop(next_grad)
        layers[0].bprop(next_grad, to_x=False)
        return self.layers[-1].loss(y, y_pred)

    def _output_shape(self, input_shape):
        for layer in self.layers:
            input_shape = layer.output_shape(input_shape)
        return input_shape

    def predict(self, input):
        """ Calculate the output for the given input x. """
        input = Input.from_any(input)
        y = np.empty(self._output_shape(input.x.shape))
        y_offset = 0
        for x_batch in input.batches():
            x_next = x_batch
            for layer in self.layers[:-1]:
                x_next = layer.fprop(x_next, 'test')
            y_batch = np.array(self.layers[-1].predict(x_next))
            batch_size = x_batch.shape[0]
            y[y_offset:y_offset+batch_size, ...] = y_batch
            y_offset += batch_size
        return y

    def error(self, input):
        input = Input.from_any(input)
        """ Calculate error on the given input. """
        y_pred = self.predict(input)
#        print(y_pred)
        # XXX: this only works for classification
        # TODO: support regression
        error = y_pred != input.y
        return np.mean(error)
