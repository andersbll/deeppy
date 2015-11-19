import numpy as np
import cudarray as ca
import deeppy.expr as expr
from ..base import Model, CollectionMixin
from ..filler import AutoFiller
from ..input import Input


class NormalSampler(expr.Expr, CollectionMixin):
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


class KLDivergence(expr.Expr):
    def __call__(self, mu, log_sigma):
        self.mu = mu
        self.log_sigma = log_sigma
        self.inputs = [mu, log_sigma]
        return self

    def setup(self):
        self.out_shape = (1,)
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        tmp1 = self.mu.out**2
        ca.negative(tmp1, tmp1)
        tmp1 += self.log_sigma.out
        tmp1 += 1
        tmp1 -= ca.exp(self.log_sigma.out)
        self.out = ca.sum(tmp1)
        self.out *= -0.5

    def bprop(self):
        ca.multiply(self.mu.out, self.out_grad, self.mu.out_grad)
        ca.exp(self.log_sigma.out, out=self.log_sigma.out_grad)
        self.log_sigma.out_grad -= 1
        self.log_sigma.out_grad *= 0.5
        self.log_sigma.out_grad *= self.out_grad


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
        lowerbound = kld + expr.sum(logpxz)
        self._lowerbound_graph = expr.ExprGraph(lowerbound)
        self._lowerbound_graph.out_grad = ca.array(1.0)
        self._lowerbound_graph.setup()

    def update(self, x):
        self.x_src.out = x
        self._lowerbound_graph.fprop()
        self._lowerbound_graph.bprop()
        return self._lowerbound_graph.out

    def _batchwise(self, input, expr_fun):
        self.phase = 'test'
        input = Input.from_any(input)
        src = expr.Source(input.x_shape)
        graph = expr.ExprGraph(expr_fun(src))
        graph.setup()
        z = []
        for x_batch in input.batches():
            src.out = x_batch['x']
            graph.fprop()
            z.append(np.array(graph.out))
        z = np.concatenate(z)[:input.n_samples]
        return z

    def embed(self, input):
        """ Input to hidden. """
        return self._batchwise(input, self._embed_expr)

    def reconstruct(self, input):
        """ Hidden to input. """
        return self._batchwise(input, self._reconstruct_expr)
