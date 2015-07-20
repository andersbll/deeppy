import cudarray as ca
from ..base import ParamMixin, PickleMixin
from ..parameter import Parameter


class Layer(PickleMixin):
    bprop_to_x = True

    def _setup(self, x_shape):
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
        self.name = 'fullconn'
        self.n_out = n_out
        self.weights = Parameter.from_any(weights)
        self.bias = Parameter.from_any(bias)
        self._tmp_x = None

    def _setup(self, x_shape):
        self.weights._setup((x_shape[1], self.n_out))
        if not self.weights.name:
            self.weights.name = self.name + '_w'
        self.bias._setup(self.n_out)
        if not self.bias.name:
            self.bias.name = self.name + '_b'

    def fprop(self, x):
        self._tmp_x = x
        return ca.dot(x, self.weights.array) + self.bias.array

    def bprop(self, y_grad):
        ca.dot(self._tmp_x.T, y_grad, out=self.weights.grad_array)
        ca.sum(y_grad, axis=0, out=self.bias.grad_array)
        if self.bprop_to_x:
            return ca.dot(y_grad, self.weights.array.T)

    @property
    def _params(self):
        return self.weights, self.bias

    @_params.setter
    def _params(self, params):
        self.weights, self.bias = params

    def y_shape(self, x_shape):
        return (x_shape[0], self.n_out)


class Activation(Layer):
    def __init__(self, method):
        self.name = 'act_'
        if method == 'sigmoid':
            self.name = self.name+'sigm'
            self.fun = ca.nnet.sigmoid
            self.fun_d = ca.nnet.sigmoid_d
        elif method == 'relu':
            self.name = self.name+method
            self.fun = ca.nnet.relu
            self.fun_d = ca.nnet.relu_d
        elif method == 'tanh':
            self.name = self.name+method
            self.fun = ca.tanh
            self.fun_d = ca.nnet.tanh_d
        else:
            raise ValueError('Invalid activation function.')
        self._tmp_x = None

    def fprop(self, x):
        self._tmp_x = x
        return self.fun(x)

    def bprop(self, y_grad):
        self.fun_d(self._tmp_x, self._tmp_x)
        return self._tmp_x * y_grad

    def y_shape(self, x_shape):
        return x_shape


class PReLU(Layer, ParamMixin):
    def __init__(self, a=0.25):
        self.name = 'prelu'
        self.a = Parameter.from_any(a)
        self._tmp_x = None

    def _setup(self, x_shape):
        self.a._setup((1,))
        self.a.name = self.name + '_a'

    @property
    def _params(self):
        return [self.a]

    @_params.setter
    def _params(self, params):
        self.a = params[0]

    def fprop(self, x):
        self._tmp_x = x
        pos = ca.maximum(x, 0)
        neg = self.a.array * ca.minimum(x, 0)
        return pos + neg

    def bprop(self, y_grad):
        pos = ca.nnet.relu_d(self._tmp_x)
        neg_mask = self._tmp_x < 0
        a_grad = neg_mask * self._tmp_x * y_grad
        ca.sum(a_grad, out=self.a.grad_array)
        return (pos + self.a.array * neg_mask) * y_grad

    def y_shape(self, x_shape):
        return x_shape
