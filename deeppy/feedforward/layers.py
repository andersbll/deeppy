import cudarray as ca
from ..base import ParamMixin, PickleMixin
from ..parameter import Parameter


class Layer(PickleMixin):
    def _setup(self, x_shape):
        """ Setup layer with parameters that are unknown at __init__(). """
        pass

    def fprop(self, x, phase):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, y_grad, to_x=True):
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
        self.name = 'fullconn'
        self.n_out = n_out
        self.W = Parameter.from_any(weights)
        self.b = Parameter.from_any(bias)

    def _setup(self, x_shape):
        W_shape = (x_shape[1], self.n_out)
        b_shape = self.n_out
        self.W._setup(W_shape)
        if not self.W.name:
            self.W.name = self.name + '_W'
        self.b._setup(b_shape)
        if not self.b.name:
            self.b.name = self.name + '_b'

    def fprop(self, x, phase):
        self._tmp_last_x = x
        return ca.dot(x, self.W.array) + self.b.array

    def bprop(self, y_grad, to_x=True):
        ca.dot(self._tmp_last_x.T, y_grad, out=self.W.grad_array)
        ca.sum(y_grad, axis=0, out=self.b.grad_array)
        if to_x:
            return ca.dot(y_grad, self.W.array.T)

    @property
    def _params(self):
        return self.W, self.b

    @_params.setter
    def _params(self, params):
        self.W, self.b = params

    def y_shape(self, x_shape):
        return (x_shape[0], self.n_out)


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
        self._tmp_last_x = x
        return self.fun(x)

    def bprop(self, y_grad, to_x=True):
        self.fun_d(self._tmp_last_x, self._tmp_last_x)
        return self._tmp_last_x * y_grad

    def y_shape(self, x_shape):
        return x_shape
