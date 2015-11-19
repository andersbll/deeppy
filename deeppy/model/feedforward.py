import numpy as np
import cudarray as ca
from ..base import Model, CollectionMixin, PickleMixin
from ..input import Input
from ..loss import SoftmaxCrossEntropy
from ..expr import nnet
from ..expr.base import Source
from ..expr.graph import ExprGraph


class FeedForwardNet(Model, CollectionMixin, PickleMixin):
    _pickle_ignore = ['_graph']

    def __init__(self, expression, loss):
        self.expression = expression
        self.loss = loss
        self._graph = None
        self.collection = [expression]

    def setup(self, x_shape, y_shape=None):
        self._x_src = Source(x_shape)
        self._y_src = Source(y_shape)
        y_pred = self.expression(self._x_src)
        loss = self.loss(y_pred, self._y_src)
        self._graph = ExprGraph(loss)
        self._graph.setup()
        self._graph.out_grad = ca.array(1)

    def update(self, x, y):
        self._x_src.out = x
        self._y_src.out = y
        self._graph.fprop()
        self._graph.bprop()
        y_pred = self._graph.out
        return self.loss.loss(y_pred, y)


class ClassifierNet(FeedForwardNet):
    def fprop(self, x):
        self._src.out = x
        self._graph.fprop()
        y = self._graph.out
        y = nnet.softmax(y)
        return y

    def _predict(self, input, top_layers):
        input = Input.from_any(input)
        input.reset()
        self.phase = 'test'
        self.layers += top_layers
        y = []
        for batch in input.batches():
            x_batch = batch['x']
            y_batch = np.array(self.fprop(x_batch))
            y.append(y_batch)
        y = np.concatenate(y)[:input.n_samples]
        self.layers = self.layers[:len(self.layers)-len(top_layers)]
        return y

    def predict(self, input):
        """ Calculate the output for the given input. """
        top_layers = []
        if isinstance(self.loss, SoftmaxCrossEntropy):
            # Add softmax from SoftmaxCrossEntropy
            top_layers.append(self.loss)
        return self._predict(input, top_layers)

    def predict_proba(self, input):
        """ Calculate the output probabilities for the given input. """
        return self._predict(input, [])


class RegressorNet(ClassifierNet):
    def predict(self, input):
        """ Calculate the output for the given input. """
        return self._predict(input, [])
