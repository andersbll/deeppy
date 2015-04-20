import cudarray as ca
from ..parameter import Parameter


class Layer(object):
    def _setup(self, input_shape):
        """ Setup layer with parameters that are unknown at __init__(). """
        pass

    def fprop(self, X, phase):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, Y_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

    def output_shape(self, input_shape):
        """ Calculate shape of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError()


class ParamMixin(object):
    @property
    def _params(self):
        """ Return list of the Layer's parameters. """
        raise NotImplementedError()

    @_params.setter
    def _params(self, params):
        """ Replace the Layer's parameters. """
        raise NotImplementedError()

    def bprop(self, y_grad, to_x=True):
        """ Backprop to parameters and input. """
        raise NotImplementedError()


class FullyConnected(Layer, ParamMixin):
    def __init__(self, n_output, weights, bias=0.0):
        self.name = 'fullconn'
        self.n_output = n_output
        self.W = Parameter.from_any(weights)
        self.b = Parameter.from_any(bias)

    def _setup(self, input_shape):
        n_input = input_shape[1]
        W_shape = (n_input, self.n_output)
        b_shape = self.n_output
        self.W._setup(W_shape)
        if not self.W.name:
            self.W.name = self.name + '_W'
        self.b._setup(b_shape)
        if not self.b.name:
            self.b.name = self.name + '_b'

    def fprop(self, x, phase):
        self._last_x = x
        return ca.dot(x, self.W.array) + self.b.array

    def bprop(self, y_grad, to_x=True):
        ca.dot(self._last_x.T, y_grad, out=self.W.grad_array)
        ca.sum(y_grad, axis=0, out=self.b.grad_array)
        if to_x:
            return ca.dot(y_grad, self.W.array.T)

    @property
    def _params(self):
        return self.W, self.b

    @_params.setter
    def _params(self, params):
        self.W, self.b = params

    def output_shape(self, input_shape):
        return (input_shape[0], self.n_output)


class Activation(Layer):
    def __init__(self, type):
        self.name = 'act_'
        if type == 'sigmoid':
            self.name = self.name+'sigm'
            self.fun = ca.nnet.sigmoid
            self.fun_d = ca.nnet.sigmoid_d
        elif type == 'relu':
            self.name = self.name+type
            self.fun = ca.nnet.relu
            self.fun_d = ca.nnet.relu_d
        elif type == 'tanh':
            self.name = self.name+type
            self.fun = ca.tanh
            self.fun_d = ca.nnet.tanh_d
        else:
            raise ValueError('Invalid activation function.')

    def fprop(self, x, phase):
        self._last_x = x
        return self.fun(x)

    def bprop(self, y_grad):
        self.fun_d(self._last_x, self._last_x)
        return self._last_x * y_grad

    def output_shape(self, input_shape):
        return input_shape
