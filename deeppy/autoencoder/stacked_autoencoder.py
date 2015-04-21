import itertools
from ..feed_forward.layers import ParamMixin
from ..feed_forward.loss import Loss
from .autoencoder import AutoencoderBase


class InputWrap(object):
    def __init__(self, input, x_shape):
        self.input = input
        self.x_shape = x_shape

    def __getattr__(self, attr):
        # Wrap Input methods
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ae, attr)


class StackedAutoencoderLayer(AutoencoderBase):
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
        for layer in self.prev_layers:
            x = layer.encode(x)
        return self.ae._update(x)

    def __getattr__(self, attr):
        # Wrap Autoencoder methods
        # abll: this is a hack. I thin it would be better to do plain
        # inheritance from Autoencoder
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ae, attr)


class StackedAutoencoder(AutoencoderBase, ParamMixin):
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
        all_params = [layer._params for layer in self.layers
                      if isinstance(layer, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    def _update(self, x):
        y = x
        for ae in self.layers:
            y = ae.encode(y)
        z = y
        for ae in reversed(self.layers):
            z = ae.decode(z)
        y_grad = self.loss.grad(x, z)
        for ae in self.layers:
            y_grad = ae.decode_bprop(y_grad)
        x_grad = y_grad
        for ae in reversed(self.layers):
            x_grad = ae.encode_bprop(x_grad)
        return self.loss.loss(x, z)

    def _output_shape(self, input_shape):
        for layer in self.layers:
            input_shape = layer.output_shape(input_shape)
        return input_shape

    def ae_models(self):
        for i, ae in enumerate(self.layers):
            yield StackedAutoencoderLayer(ae, self.layers[:i])

    def nn_layers(self):
        nn_layers = [ae.nn_layers() for ae in self.layers]
        return list(itertools.chain.from_iterable(nn_layers))
