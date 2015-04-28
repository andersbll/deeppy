import cudarray as ca
from ..feed_forward.layers import ParamMixin, Activation, FullyConnected
from ..feed_forward.loss import Loss
from ..base import Model, PickleMixin
from ..parameter import Parameter


class AutoencoderBase(object):
    def encode(x):
        raise NotImplementedError()

    def decode(y):
        raise NotImplementedError()

    def encode_bprop(y):
        raise NotImplementedError()

    def decode_bprop(x):
        raise NotImplementedError()


class Autoencoder(Model, AutoencoderBase, ParamMixin, PickleMixin):
    def __init__(self, n_output, weights, bias=0.0, activation='sigmoid',
                 loss='bce'):
        self.name = 'autoenc'
        self.n_output = n_output
        self.activation = Activation(activation)
        self.activation_decode = Activation(activation)
        self.loss = Loss.from_any(loss)
        self.W = Parameter.from_any(weights)
        self.b = Parameter.from_any(bias)
        self.b_prime = Parameter.from_any(bias)
        self._initialized = False

    def _setup(self, input):
        if self._initialized:
            return
        next_shape = input.x_shape
        n_input = next_shape[1]
        W_shape = (n_input, self.n_output)
        b_shape = self.n_output
        b_prime_shape = n_input
        self.W._setup(W_shape)
        if not self.W.name:
            self.W.name = self.name + '_W'
        self.b._setup(b_shape)
        if not self.b.name:
            self.b.name = self.name + '_b'
        self.b_prime._setup(b_prime_shape)
        if not self.b_prime.name:
            self.b_prime.name = self.name + '_b_prime'
        self.loss._setup((next_shape[0], self.n_output))
        self._initialized = True

    @property
    def _params(self):
        return self.W, self.b, self.b_prime

    @_params.setter
    def _params(self, params):
        self.W, self.b, self.b_prime = params

    def output_shape(self, input_shape):
        return (input_shape[0], self.n_output)

    def encode(self, x):
        self._tmp_last_x = x
        y = ca.dot(x, self.W.array) + self.b.array
        return self.activation.fprop(y, '')

    def decode(self, y_prime):
        self._tmp_last_y_prime = y_prime
        x_prime = ca.dot(y_prime, self.W.array.T) + self.b_prime.array
        return self.activation_decode.fprop(x_prime, '')

    def encode_bprop(self, y_grad):
        y_grad = self.activation.bprop(y_grad)
        # Because W's gradient has already been updated by decode_bprop() at
        # this point, we should add its contribution from the encode step.
        W_grad = self.W.grad_array
        W_grad += ca.dot(self._tmp_last_x.T, y_grad)
        ca.sum(y_grad, axis=0, out=self.b.grad_array)
        return ca.dot(y_grad, self.W.array.T)

    def decode_bprop(self, x_prime_grad):
        x_prime_grad = self.activation_decode.bprop(x_prime_grad)
        ca.dot(x_prime_grad.T, self._tmp_last_y_prime, out=self.W.grad_array)
        ca.sum(x_prime_grad, axis=0, out=self.b_prime.grad_array)
        return ca.dot(x_prime_grad, self.W.array)

    def _update(self, x):
        y_prime = self.encode(x)
        x_prime = self.decode(y_prime)
        x_prime_grad = self.loss.grad(x, x_prime)
        y_grad = self.decode_bprop(x_prime_grad)
        self.encode_bprop(y_grad)
        return self.loss.loss(x, x_prime)

    def nn_layers(self):
        return [FullyConnected(self.n_output, self.W.array, self.b.array),
                self.activation]


class DenoisingAutoencoder(Autoencoder):
    def __init__(self, n_output, weights, bias=0.0, activation='sigmoid',
                 loss='bce', corruption=0.25):
        super(DenoisingAutoencoder, self).__init__(
            n_output=n_output, weights=weights, bias=bias,
            activation=activation, loss=loss
        )
        self.corruption = corruption

    def corrupt(self, x):
        mask = ca.random.uniform(size=x.shape) < (1-self.corruption)
        return x * mask

    def _update(self, x):
        x_tilde = self.corrupt(x)
        y_prime = self.encode(x_tilde)
        x_prime = self.decode(y_prime)
        x_prime_grad = self.loss.grad(x, x_prime)
        y_grad = self.decode_bprop(x_prime_grad)
        self.encode_bprop(y_grad)
        return self.loss.loss(x, x_prime)
