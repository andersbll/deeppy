import numpy as np
import itertools
from ..base import Model, ParamMixin, PhaseMixin
from ..input import Input


class NeuralNetwork(Model, PhaseMixin):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.bprop_until = next((idx for idx, l in enumerate(self.layers)
                                 if isinstance(l, ParamMixin)), 0)
        self.layers[self.bprop_until].bprop_to_x = False
        self._initialized = False

    def _setup(self, input):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = input.x_shape
        for layer in self.layers + [self.loss]:
            layer._setup(next_shape)
            next_shape = layer.y_shape(next_shape)
        if next_shape != input.y_shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, input.y_shape))
        self._initialized = True

    @property
    def _params(self):
        all_params = [layer._params for layer in self.layers
                      if isinstance(layer, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    @PhaseMixin.phase.setter
    def phase(self, phase):
        if self._phase == phase:
            return
        self._phase = phase
        for layer in self.layers:
            if isinstance(layer, PhaseMixin):
                layer.phase = phase

    def _update(self, batch):
        self.phase = 'train'

        # Forward propagation
        x, y = batch
        x_next = x
        for layer in self.layers:
            x_next = layer.fprop(x_next)
        y_pred = x_next

        # Back propagation of partial derivatives
        next_grad = self.loss.grad(y, y_pred)
        layers = self.layers[self.bprop_until:]
        for layer in reversed(layers[1:]):
            next_grad = layer.bprop(next_grad)
        layers[0].bprop(next_grad)
        return self.loss.loss(y, y_pred)

    def _output_shape(self, input_shape):
        next_shape = input_shape
        for layer in self.layers + [self.loss]:
            next_shape = layer.y_shape(next_shape)
        return next_shape

    def predict(self, input):
        """ Calculate the output for the given input x. """
        self.phase = 'test'
        input = Input.from_any(input)
        y = np.empty(self._output_shape(input.x.shape))
        y_offset = 0
        for x_batch in input.batches():
            x_next = x_batch
            for layer in self.layers:
                x_next = layer.fprop(x_next)
            y_batch = np.array(self.loss.fprop(x_next))
            batch_size = x_batch.shape[0]
            y[y_offset:y_offset+batch_size, ...] = y_batch
            y_offset += batch_size
        return y
