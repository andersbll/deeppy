import numpy as np
import cudarray as ca
from ..feedforward.activation_layers import Activation
from ..feedforward.layers import FullyConnected
from ..loss import Loss
from ..base import Model, PickleMixin
from ..input import Input
from ..parameter import Parameter


class Autoencoder(Model, PickleMixin):
    def __init__(self, n_out, weights, bias=0.0, bias_prime=0.0,
                 activation='sigmoid', loss='bce'):
        self.n_out = n_out
        self.activation = Activation.from_any(activation)
        self.activation_decode = Activation.from_any(activation)
        self.loss = Loss.from_any(loss)
        self.weights = Parameter.from_any(weights)
        self.bias = Parameter.from_any(bias)
        self.bias_prime = Parameter.from_any(bias_prime)
        self._initialized = False
        self._tmp_x = None
        self._tmp_y = None

    def setup(self, x_shape):
        if self._initialized:
            return
        n_in = x_shape[1]
        self.weights.setup((n_in, self.n_out))
        self.bias.setup(self.n_out)
        self.bias_prime.setup(n_in)
        self.loss.setup((x_shape[0], self.n_out))
        self._initialized = True

    @property
    def params(self):
        return self.weights, self.bias, self.bias_prime

    @params.setter
    def params(self, params):
        self.weights, self.bias, self.bias_prime = params

    def output_shape(self, input_shape):
        return (input_shape[0], self.n_out)

    def encode(self, x):
        self._tmp_x = x
        y = ca.dot(x, self.weights.array) + self.bias.array
        return self.activation.fprop(y)

    def decode(self, y):
        self._tmp_y = y
        x = ca.dot(y, self.weights.array.T) + self.bias_prime.array
        return self.activation_decode.fprop(x)

    def decode_bprop(self, x_grad):
        x_grad = self.activation_decode.bprop(x_grad)
        ca.dot(x_grad.T, self._tmp_y, out=self.weights.grad_array)
        ca.sum(x_grad, axis=0, out=self.bias_prime.grad_array)
        return ca.dot(x_grad, self.weights.array)

    def encode_bprop(self, y_grad):
        y_grad = self.activation.bprop(y_grad)
        # Because the weight gradient has already been updated by
        # decode_bprop() we must add the contribution.
        w_grad = self.weights.grad_array
        w_grad += ca.dot(self._tmp_x.T, y_grad)
        ca.sum(y_grad, axis=0, out=self.bias.grad_array)
        return ca.dot(y_grad, self.weights.array.T)

    def update(self, x):
        y_prime = self.encode(x)
        x_prime = self.decode(y_prime)
        x_prime_grad = self.loss.grad(x_prime, x)
        y_grad = self.decode_bprop(x_prime_grad)
        self.encode_bprop(y_grad)
        return self.loss.loss(x_prime, x)

    def _reconstruct_batch(self, x):
        y = self.encode(x)
        return self.decode(y)

    def reconstruct(self, input):
        """ Returns the reconstructed input. """
        input = Input.from_any(input)
        x_prime = np.empty(input.x.shape)
        offset = 0
        for x_batch in input.batches():
            x_prime_batch = np.array(self._reconstruct_batch(x_batch))
            batch_size = x_prime_batch.shape[0]
            x_prime[offset:offset+batch_size, ...] = x_prime_batch
            offset += batch_size
        return x_prime

    def _embed_batch(self, x):
        return self.encode(x)

    def embed(self, input):
        """ Returns the embedding of the input. """
        input = Input.from_any(input)
        y = np.empty(self.output_shape(input.x.shape))
        offset = 0
        for x_batch in input.batches():
            y_batch = np.array(self._embed_batch(x_batch))
            batch_size = y_batch.shape[0]
            y[offset:offset+batch_size, ...] = y_batch
            offset += batch_size
        return y

    def feedforward_layers(self):
        return [FullyConnected(self.n_out, self.weights.array,
                               self.bias.array),
                self.activation]


class DenoisingAutoencoder(Autoencoder):
    def __init__(self, n_out, weights, bias=0.0, bias_prime=0.0,
                 corruption=0.25, activation='sigmoid', loss='bce'):
        super(DenoisingAutoencoder, self).__init__(
            n_out=n_out, weights=weights, bias=bias, bias_prime=bias_prime,
            activation=activation, loss=loss
        )
        self.corruption = corruption

    def corrupt(self, x):
        mask = ca.random.uniform(size=x.shape) < (1-self.corruption)
        return x * mask

    def update(self, x):
        x_tilde = self.corrupt(x)
        y_prime = self.encode(x_tilde)
        x_prime = self.decode(y_prime)
        x_prime_grad = self.loss.grad(x_prime, x)
        y_grad = self.decode_bprop(x_prime_grad)
        self.encode_bprop(y_grad)
        return self.loss.loss(x_prime, x)
