import itertools
from ..base import ParamMixin
from ..feedforward.loss import Loss
from .autoencoder import Autoencoder


class InputWrap(object):
    def __init__(self, input, x_shape):
        self.input = input
        self.x_shape = x_shape

    def __getattr__(self, attr):
        # Wrap Input methods
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ae, attr)


class StackedAutoencoderLayer(Autoencoder):
    def __init__(self, ae, prev_layers):
        self.ae = ae
        self.prev_layers = prev_layers
        self._initialized = False

    def _setup(self, input):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = input.x_shape
        for ae in self.prev_layers:
            ae._setup(InputWrap(input, next_shape))
            next_shape = ae.output_shape(next_shape)
        self.ae._setup(InputWrap(input, next_shape))
        self._initialized = True

    def _update(self, x):
        for ae in self.prev_layers:
            x = ae.encode(x)
        return self.ae._update(x)

    def _reconstruct_batch(self, x):
        for ae in self.prev_layers:
            x = ae.encode(x)
        y = self.ae.encode(x)
        x_prime = self.ae.decode(y)
        for ae in reversed(self.prev_layers):
            x_prime = ae.decode(x_prime)
        return x_prime

    def _embed_batch(self, x):
        for ae in self.prev_layers:
            x = ae.encode(x)
        return self.ae.encode(x)

    def __getattr__(self, attr):
        # Wrap non-overriden Autoencoder attributes
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ae, attr)


class StackedAutoencoder(Autoencoder):
    def __init__(self, layers, loss='bce'):
        self._initialized = False
        self.layers = layers
        self.loss = Loss.from_any(loss)

    def _setup(self, input):
        if self._initialized:
            return
        next_shape = input.x_shape
        for ae in self.layers:
            ae._setup(InputWrap(input, next_shape))
            next_shape = ae.output_shape(next_shape)
        self.loss._setup(next_shape)
        self._initialized = True

    @property
    def _params(self):
        all_params = [ae._params for ae in self.layers
                      if isinstance(ae, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    def encode(self, x):
        for ae in self.layers:
            x = ae.encode(x)
        return x

    def decode(self, y):
        for ae in reversed(self.layers):
            y = ae.decode(y)
        return y

    def decode_bprop(self, x_grad):
        for ae in self.layers:
            x_grad = ae.decode_bprop(x_grad)
        return x_grad

    def encode_bprop(self, y_grad):
        for ae in reversed(self.layers):
            y_grad = ae.encode_bprop(y_grad)
        return y_grad

    def _output_shape(self, input_shape):
        next_shape = input_shape
        for ae in self.layers:
            next_shape = ae.output_shape(next_shape)
        return next_shape

    def feedforward_layers(self):
        feedforward_layers = [ae.feedforward_layers() for ae in self.layers]
        return list(itertools.chain.from_iterable(feedforward_layers))

    def ae_models(self):
        for i, ae in enumerate(self.layers):
            yield StackedAutoencoderLayer(ae, self.layers[:i])
