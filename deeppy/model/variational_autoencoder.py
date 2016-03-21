import numpy as np
import cudarray as ca
import deeppy.expr as expr
from ..base import Model, CollectionMixin
from ..filler import AutoFiller
from ..input import Input


class NormalSampler(expr.Op, CollectionMixin):
    def __init__(self, n_hidden):
        self.weight_filler = AutoFiller()
        self.bias_filler = 0.0
        self.z_mu = self._affine(n_hidden)
        self.z_log_sigma = self._affine(n_hidden)
        self.n_hidden = n_hidden
        self.collection = [self.z_mu, self.z_log_sigma]
        self.batch_size = None

    def _affine(self, n_out):
        return expr.nnet.Affine(
            n_out=n_out,
            weights=self.weight_filler,
            bias=self.bias_filler,
        )

    def __call__(self, h_enc):
        z_mu = self.z_mu(h_enc)
        z_log_sigma = self.z_log_sigma(h_enc)
        eps = expr.random.normal(size=(self.batch_size, self.n_hidden))
        z = z_mu + expr.exp(0.5 * z_log_sigma) * eps
        return z, z_mu, z_log_sigma


class KLDivergence(expr.Op):
    def __call__(self, mu, log_sigma):
        self.mu = mu
        self.log_sigma = log_sigma
        self.inputs = [mu, log_sigma]
        return self

    def setup(self):
        self.shape = (1,)
        self.array = ca.empty(self.shape)
        self.grad_array = ca.empty(self.shape)

    def fprop(self):
        tmp1 = self.mu.array**2
        ca.negative(tmp1, tmp1)
        tmp1 += self.log_sigma.array
        tmp1 += 1
        tmp1 -= ca.exp(self.log_sigma.array)
        self.array = ca.sum(tmp1)
        self.array *= -0.5

    def bprop(self):
        ca.multiply(self.mu.array, self.grad_array, self.mu.grad_array)
        ca.exp(self.log_sigma.array, out=self.log_sigma.grad_array)
        self.log_sigma.grad_array -= 1
        self.log_sigma.grad_array *= 0.5
        self.log_sigma.grad_array *= self.grad_array


class VariationalAutoencoder(Model, CollectionMixin):
    def __init__(self, encoder, decoder, n_hidden, reconstruct_error=None):
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = NormalSampler(n_hidden)
        if reconstruct_error is None:
            reconstruct_error = expr.nnet.BinaryCrossEntropy()
        self.reconstruct_error = reconstruct_error
        self.collection = [encoder, self.sampler, decoder]

    def _embed_expr(self, x):
        h_enc = self.encoder(x)
        z, z_mu, z_log_sigma = self.sampler(h_enc)
        return z_mu

    def _reconstruct_expr(self, z):
        return self.decoder(z)

    def setup(self, x_shape):
        self.sampler.batch_size = x_shape[0]
        self.x_src = expr.Source(x_shape)
        h_enc = self.encoder(self.x_src)
        z, z_mu, z_log_sigma = self.sampler(h_enc)
        kld = KLDivergence()(z_mu, z_log_sigma)
        x_tilde = self.decoder(z)
        logpxz = self.reconstruct_error(x_tilde, self.x_src)
        self.lowerbound = kld + expr.sum(logpxz)
        self._graph = expr.graph.ExprGraph(self.lowerbound)
        self._graph.setup()
        self.lowerbound.grad_array = ca.array(1.0)

    def update(self, x):
        self.x_src.array = x
        self._graph.fprop()
        self._graph.bprop()
        return self.lowerbound.array

    def _batchwise(self, input, expr_fun):
        self.phase = 'test'
        input = Input.from_any(input)
        src = expr.Source(input.x_shape)
        sink = expr_fun(src)
        graph = expr.graph.ExprGraph(sink)
        graph.setup()
        z = []
        for x_batch in input.batches():
            src.array = x_batch['x']
            graph.fprop()
            z.append(np.array(sink.array))
        z = np.concatenate(z)[:input.n_samples]
        return z

    def embed(self, input):
        """ Input to hidden. """
        return self._batchwise(input, self._embed_expr)

    def reconstruct(self, input):
        """ Hidden to input. """
        return self._batchwise(input, self._reconstruct_expr)
