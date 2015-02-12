import warnings
import numpy as np
import cudarray as ca
import itertools
from .layers_seg import ParamMixin_seg
from ..input import to_input

import logging
logger = logging.getLogger(__name__)


class NeuralNetwork_seg:
    def __init__(self, layers):
        self._initialized = False
        self.layers = layers
        self.bprop_until = next((idx for idx, l in enumerate(self.layers)
                                 if isinstance(l, ParamMixin_seg)),
                                len(self.layers))

    def _setup(self, input):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = input.x_shape

        img_h, img_w = input.x_shape[-2:]
        input_index = np.arange(img_h*img_w, dtype=np.int_)
        input_index = input_index.reshape((1,)+input.x_shape[-2:])

        for layer in self.layers:
            layer._setup(next_shape)
            #Must be befor Next Shape
            if (input_index != None):
                input_index = layer.output_index(input_index)

            next_shape = layer.output_shape(next_shape)

        if next_shape != input.y_shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, input.y_shape))
        self._initialized = True

        print "setup Done"

    def _params(self):
        all_params = [layer.params() for layer in self.layers
                      if isinstance(layer, ParamMixin_seg)]
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
        input = to_input(input)
        y = np.empty((input.x.shape[0],)+self._output_shape(input.x_shape))
        y_offset = 0
        for x_batch in input.batches():
            x_next = x_batch[0]
            for layer in self.layers[:-1]:
                x_next = layer.fprop(x_next, 'test')
            y_batch = np.array(self.layers[-1].predict(x_next))
            batch_size = x_batch.shape[0]
            y[y_offset:y_offset+batch_size, ...] = y_batch
            y_offset += batch_size
        return y

    def error(self, input):
        input = to_input(input)
        """ Calculate error on the given input. """
        y_pred = self.predict(input)

        logger.info('Test')
        logger.info('Predict: class1: %d, class2:%d' % (np.sum(y_pred), abs(y_pred.size - np.sum(y_pred))))
        logger.info('True: class1: %d, class2:%d' % (np.sum(input.y), abs(input.y.size - np.sum(input.y))))
#        print(y_pred)
        # XXX: this only works for classification
        # TODO: support regression
        error = y_pred != input.y
        return np.mean(error)
