import numpy as np
import itertools
from .layers import ParamMixin
from ..data import to_data, Data


class NeuralNetwork:
    def __init__(self, layers):
        self._initialized = False
        self.layers = layers
        self.bprop_until = next(idx for idx, layer in enumerate(layers)
                                if isinstance(layer, ParamMixin))

    def _setup(self, data):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = data.x_shape
        for layer in self.layers:
            layer._setup(next_shape)
            next_shape = layer.output_shape(next_shape)
        if next_shape != data.y_shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, data.y_shape))
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
        for layer in reversed(layers):
            next_grad = layer.bprop(next_grad)
        return self.layers[-1].loss(y, y_pred)

    def _output_shape(self, input_shape):
        for layer in self.layers:
            input_shape = layer.output_shape(input_shape)
        return input_shape

    def predict(self, data):
        """ Calculate the output for the given input x. """
        data = to_data(data)
        y = np.empty(self._output_shape(data.x.shape))
        y_offset = 0
        for x_batch in data.batches():
            x_next = x_batch
            for layer in self.layers[:-1]:
                x_next = layer.fprop(x_next, 'test')
            y_batch = np.array(self.layers[-1].predict(x_next))
            batch_size = x_batch.shape[0]
            y[y_offset:y_offset+batch_size, ...] = y_batch
            y_offset += batch_size
        return y

    def error(self, data):
        data = to_data(data)
        """ Calculate error on the given data. """
        y_pred = self.predict(Data(data.x, data.batch_size))
#        print(y_pred)
        # XXX: this only works for classification
        # TODO: support regression
        error = y_pred != data.y
        return np.mean(error)
