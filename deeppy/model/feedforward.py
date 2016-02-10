import numpy as np
from ..base import Model, CollectionMixin, PickleMixin
from ..input import Input
from .. import expr


class FeedForwardNet(Model, CollectionMixin, PickleMixin):
    _pickle_ignore = ['_graph']

    def __init__(self, expression, loss):
        self.expression = expression
        self.loss = loss
        self._graph = None
        self.collection = [expression]

    def setup(self, x_shape, y_shape=None):
        self._x_src = expr.Source(x_shape)
        y_expr = self._fprop_expr(self._x_src)
        if y_shape is not None:
            self._y_src = expr.Source(y_shape)
            y_expr = self.loss(y_expr, self._y_src)
        self._graph = expr.ExprGraph(y_expr)
        self._graph.setup()
        self._graph.out_grad = 1.0

    def _fprop_expr(self, x):
        return self.expression(x)

    def update(self, x, y):
        self._x_src.out = x
        self._y_src.out = y
        self._graph.fprop()
        self._graph.bprop()
        return self.loss.out

    def _batchwise(self, input, expr_fun):
        input = Input.from_any(input)
        src = expr.Source(input.x_shape)
        graph = expr.ExprGraph(expr_fun(src))
        graph.setup()
        y = []
        for batch in input.batches():
            src.out = batch['x']
            graph.fprop()
            y.append(np.array(graph.out))
        y = np.concatenate(y)[:input.n_samples]
        return y

    def predict(self, input):
        """ Calculate the output for the given input. """
        return self._batchwise(input, self._fprop_expr)


class ClassifierNet(FeedForwardNet):
    def _fprop_expr(self, x):
        y_expr = super(ClassifierNet, self)._fprop_expr(x)
        if isinstance(self.loss, expr.nnet.SoftmaxCrossEntropy):
            softmax = expr.nnet.SoftmaxCrossEntropy.SoftmaxIdentityBProp()
            y_expr = softmax(y_expr)
        return y_expr

    def _predict_expr(self, x):
        y_expr = self._fprop_expr(x)
        y_expr = expr.nnet.one_hot.OneHotDecode()(y_expr)
        return y_expr

    def predict(self, input):
        """ Calculate the output for the given input. """
        return self._batchwise(input, self._predict_expr)

    def predict_proba(self, input):
        """ Calculate the output probabilities for the given input. """
        return self._batchwise(input, self._fprop_expr)


class RegressorNet(FeedForwardNet):
    def predict(self, input):
        """ Calculate the output for the given input. """
        return self._batchwise(input, self._fprop_expr)
