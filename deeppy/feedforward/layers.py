import cudarray as ca
from ..base import ParamMixin, PickleMixin
from ..parameter import Parameter


class Layer(PickleMixin):
    bprop_to_x = True

    def setup(self, x_shape):
        """ Setup layer with parameters that are unknown at __init__(). """
        pass

    def fprop(self, x):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, y_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

    def y_shape(self, x_shape):
        """ Calculate shape of this layer's output.
        x_shape[0] is the number of samples in the input.
        x_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError()


class FullyConnected(Layer, ParamMixin):
    def __init__(self, n_out, weights, bias=0.0):
        self.n_out = n_out
        self.weights = Parameter.from_any(weights)
        self.bias = Parameter.from_any(bias)
        self._tmp_x = None

    def setup(self, x_shape):
        self.weights.setup((x_shape[1], self.n_out))
        self.bias.setup(self.n_out)

    def fprop(self, x):
        self._tmp_x = x
        return ca.dot(x, self.weights.array) + self.bias.array

    def bprop(self, y_grad):
        ca.dot(self._tmp_x.T, y_grad, out=self.weights.grad_array)
        ca.sum(y_grad, axis=0, out=self.bias.grad_array)
        if self.bprop_to_x:
            return ca.dot(y_grad, self.weights.array.T)

    @property
    def params(self):
        return self.weights, self.bias

    @params.setter
    def params(self, params):
        self.weights, self.bias = params

    def y_shape(self, x_shape):
        return (x_shape[0], self.n_out)
