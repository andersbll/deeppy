from copy import copy
import numpy as np
from ..base import Model, CollectionMixin, ParamMixin, PhaseMixin, float_
from ..input import Input


class SiameseNetwork(Model, CollectionMixin, PhaseMixin):
    def __init__(self, siamese_layers, loss):
        self.layers = siamese_layers
        self.loss = loss
        # Create second array of layers
        self.layers2 = [copy(layer) for layer in self.layers]
        for layer1, layer2 in zip(self.layers, self.layers2):
            if isinstance(layer1, ParamMixin):
                # Replace weights in layers2 with shared weights
                layer2.params = [p.share() for p in layer1.params]
        self.bprop_until = next((idx for idx, l in enumerate(self.layers)
                                 if isinstance(l, ParamMixin)), 0)
        self.layers[self.bprop_until].bprop_to_x = False
        self.layers2[self.bprop_until].bprop_to_x = False
        self.collection = self.layers + self.layers2
        self._initialized = False

    def setup(self, x_shape, y_shape=None):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = x_shape
        for layer in self.layers:
            layer.setup(next_shape)
            next_shape = layer.y_shape(next_shape)
        next_shape = x_shape
        for layer in self.layers2:
            layer.setup(next_shape)
            next_shape = layer.y_shape(next_shape)
        next_shape = self.loss.y_shape(next_shape)
        self._initialized = True

    def update(self, x1, x2, y):
        self.phase = 'train'

        # Forward propagation
        for layer in self.layers:
            x1 = layer.fprop(x1)
        for layer in self.layers2:
            x2 = layer.fprop(x2)

        # Back propagation of partial derivatives
        grad1, grad2 = self.loss.grad(y, x1, x2)
        layers = self.layers[self.bprop_until:]
        for layer in reversed(layers[1:]):
            grad1 = layer.bprop(grad1)
        layers[0].bprop(grad1)

        layers2 = self.layers2[self.bprop_until:]
        for layer in reversed(layers2[1:]):
            grad2 = layer.bprop(grad2)
        layers2[0].bprop(grad2)

        return self.loss.loss(y, x1, x2)

    def features(self, input):
        self.phase = 'test'
        input = Input.from_any(input)
        next_shape = input.x.shape
        for layer in self.layers:
            next_shape = layer.y_shape(next_shape)
        feats = np.empty(next_shape)
        idx = 0
        for batch in input.batches():
            x_batch = batch['x']
            x_next = x_batch
            for layer in self.layers:
                x_next = layer.fprop(x_next)
            feats_batch = np.array(x_next)
            batch_size = x_batch.shape[0]
            feats[idx:idx+batch_size, ...] = feats_batch
            idx += batch_size
        return feats

    def distances(self, input):
        self.phase = 'test'
        dists = np.empty((input.n_samples,), dtype=float_)
        offset = 0
        for batch in input.batches():
            x1, x2 = batch
            for layer in self.layers:
                x1 = layer.fprop(x1)
            for layer in self.layers2:
                x2 = layer.fprop(x2)
            dists_batch = self.loss.fprop(x1, x2)
            dists_batch = np.ravel(np.array(dists_batch))
            batch_size = x1.shape[0]
            dists[offset:offset+batch_size, ...] = dists_batch
            offset += batch_size
        return dists
