from copy import copy
import numpy as np
import itertools
from ..feed_forward.layers import ParamMixin
from ..input import to_input
from ..base import float_


class SiameseNetwork(object):
    def __init__(self, siamese_layers, loss_layer):
        self._initialized = False
        self.layers = siamese_layers
        # Create second array of layers
        self.layers2 = [copy(layer) for layer in self.layers]
        for layer1, layer2 in zip(self.layers, self.layers2):
            if isinstance(layer1, ParamMixin):
                # Replace weights in layers2 with shared weights
                params = layer1.params()
                params = [p.share() for p in params]
                layer2.set_params(params)
        self.loss_layer = loss_layer
        self.bprop_until = next(idx for idx, layer in enumerate(self.layers)
                                if isinstance(layer, ParamMixin))

    def _setup(self, input):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = input.x_shape
        for layer in self.layers:
            layer._setup(next_shape)
            next_shape = layer.output_shape(next_shape)
        next_shape = self.loss_layer.output_shape(next_shape)
        self._initialized = True

    def _params(self):
        all_params = [layer.params() for layer in self.layers
                      if isinstance(layer, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    def _update(self, batch):
        # Forward propagation
        x1, x2, y = batch
        for layer in self.layers:
            x1 = layer.fprop(x1, 'train')
        for layer in self.layers2:
            x2 = layer.fprop(x2, 'train')
        dists = self.loss_layer.fprop(x1, x2, 'train')

        # Back propagation of partial derivatives
        grad1, grad2 = self.loss_layer.input_grad(y, dists)
        layers = self.layers[self.bprop_until:]
        for layer in reversed(layers):
            grad1 = layer.bprop(grad1)
        layers2 = self.layers2[self.bprop_until:]
        for layer in reversed(layers2):
            grad2 = layer.bprop(grad2)

        return self.loss_layer.loss(y, dists)

    def features(self, input):
        input = to_input(input)
        next_shape = input.x.shape
        for layer in self.layers:
            next_shape = layer.output_shape(next_shape)
        feats = np.empty(next_shape)
        idx = 0
        for x_batch in input.batches():
            x_next = x_batch
            for layer in self.layers:
                x_next = layer.fprop(x_next, 'test')
            feats_batch = np.array(x_next)
            batch_size = x_batch.shape[0]
            feats[idx:idx+batch_size, ...] = feats_batch
            idx += batch_size
        return feats

    def distances(self, input):
        input = to_input(input)
        dists = np.empty((input.n_samples,), dtype=float_)
        offset = 0
        for batch in input.batches():
            x1, x2 = batch
            for layer in self.layers:
                x1 = layer.fprop(x1, 'test')
            for layer in self.layers2:
                x2 = layer.fprop(x2, 'test')
            dists_batch = self.loss_layer.fprop(x1, x2, 'test')
            dists_batch = np.ravel(np.array(dists_batch))
            batch_size = x1.shape[0]
            dists[offset:offset+batch_size, ...] = dists_batch
            offset += batch_size
        return dists
