"""
Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift, http://arxiv.org/abs/1502.03167

This implementation is heavily inspired by code from http://github.com/torch/nn
"""
import cudarray as ca
from ...base import ParamMixin, PhaseMixin
from ...parameter import Parameter
from ..base import UnaryElementWise


class BatchNormalization(UnaryElementWise, ParamMixin, PhaseMixin):
    def __init__(self, momentum=0.9, eps=1e-5, noise_std=0.0, affine=True):
        self.momentum = momentum
        self.eps = eps
        self.phase = 'train'
        self.affine = affine
        if self.affine:
            self.gamma = Parameter(1.0)
            self.beta = Parameter(0.0)
        self.running_mean = None
        self.noise_std = noise_std

    def setup(self):
        super(BatchNormalization, self).setup()
        reduced_shape = (1,) + self.shape[1:]
        if self.running_mean is None:
            self.running_mean = ca.zeros(reduced_shape)
            self.running_std = ca.ones(reduced_shape)
            if self.affine:
                self.gamma.setup(reduced_shape)
                self.beta.setup(reduced_shape)
        else:
            if self.running_mean.shape != reduced_shape:
                raise ValueError('New input shape is not compatible')
        self._tmp_batch_inv_std = ca.zeros(reduced_shape)
        self._tmp_batch_centered = ca.zeros(self.shape)

    @property
    def params(self):
        params = []
        if self.affine:
            params = [self.gamma, self.beta]
        return params

    @params.setter
    def params(self, params):
        if self.affine:
            self.gamma, self.beta = params

    def fprop(self):
        if self.phase == 'train':
            # Calculate batch mean
            tmp = ca.mean(self.x.array, axis=0, keepdims=True)
            # Center input
            ca.subtract(self.x.array, tmp, self._tmp_batch_centered)
            # Update running mean
            tmp *= 1 - self.momentum
            self.running_mean *= self.momentum
            self.running_mean += tmp
            # Calculate batch variance
            ca.power(self._tmp_batch_centered, 2, self.array)
            ca.mean(self.array, axis=0, keepdims=True,
                    out=self._tmp_batch_inv_std)
            # Calculate 1 / E([x - E(x)]^2)
            self._tmp_batch_inv_std += self.eps
            ca.sqrt(self._tmp_batch_inv_std, self._tmp_batch_inv_std)
            ca.power(self._tmp_batch_inv_std, -1, self._tmp_batch_inv_std)
            # Normalize input
            ca.multiply(self._tmp_batch_centered, self._tmp_batch_inv_std,
                        self.array)
            # Update running std
            self.running_std *= self.momentum
            ca.multiply(self._tmp_batch_inv_std, 1-self.momentum, tmp)
            self.running_std += tmp

            if self.noise_std > 0.0:
                noise = ca.random.normal(scale=self.noise_std,
                                         size=self.shape)
                ca.add(self.array, noise, self.array)
        elif self.phase == 'test':
            ca.subtract(self.x.array, self.running_mean, self.array)
            self.array *= self.running_std
        else:
            raise ValueError('Invalid phase: %s' % self.phase)
        if self.affine:
            self.array *= self.gamma.array
            self.array += self.beta.array

    def bprop(self):
        ca.multiply(self._tmp_batch_centered, self.grad_array,
                    self.x.grad_array)
        tmp = ca.mean(self.x.grad_array, axis=0, keepdims=True)
        ca.multiply(self._tmp_batch_centered, tmp, self.x.grad_array)
        self.x.grad_array *= -1
        self.x.grad_array *= self._tmp_batch_inv_std
        self.x.grad_array *= self._tmp_batch_inv_std

        ca.mean(self.grad_array, axis=0, keepdims=True, out=tmp)
        self.x.grad_array += self.grad_array
        self.x.grad_array -= tmp
        self.x.grad_array *= self._tmp_batch_inv_std

        if self.affine:
            self.x.grad_array *= self.gamma.array
            # Normalized input
            self._tmp_batch_centered *= self._tmp_batch_inv_std
            self._tmp_batch_centered *= self.grad_array
            ca.sum(self._tmp_batch_centered, axis=0, keepdims=True,
                   out=self.gamma.grad_array)
            ca.sum(self.grad_array, axis=0, keepdims=True,
                   out=self.beta.grad_array)


class SpatialBatchNormalization(UnaryElementWise, ParamMixin, PhaseMixin):
    def __init__(self, momentum=0.9, eps=1e-5, noise_std=0.0, affine=True):
        self.momentum = momentum
        self.eps = eps
        self.phase = 'train'
        self.affine = affine
        if self.affine:
            self.gamma = Parameter(1.0)
            self.beta = Parameter(0.0)
        self.running_mean = None
        self.noise_std = noise_std

    def setup(self):
        super(SpatialBatchNormalization, self).setup()
        if len(self.shape) != 4:
            raise ValueError('Only 4D data supported')
        reduced_shape = 1, self.shape[1], 1, 1
        if self.running_mean is None:
            self.running_mean = ca.zeros(reduced_shape)
            self.running_std = ca.ones(reduced_shape)
            if self.affine:
                self.gamma.setup(reduced_shape)
                self.beta.setup(reduced_shape)
        else:
            if self.running_mean.shape != reduced_shape:
                raise ValueError('New input shape is not compatible')
        self._tmp_batch_inv_std = ca.zeros(reduced_shape)
        self._tmp_batch_centered = ca.zeros(self.shape)

    @property
    def params(self):
        params = []
        if self.affine:
            params = [self.gamma, self.beta]
        return params

    @params.setter
    def params(self, params):
        if self.affine:
            self.gamma, self.beta = params

    def fprop(self):
        if self.phase == 'train':
            # Calculate batch mean
            tmp = ca.mean(ca.mean(self.x.array, axis=0, keepdims=True),
                          axis=(2, 3), keepdims=True)
            # Center input
            ca.subtract(self.x.array, tmp, self._tmp_batch_centered)
            # Update running mean
            tmp *= 1 - self.momentum
            self.running_mean *= self.momentum
            self.running_mean += tmp
            # Calculate batch variance
            ca.power(self._tmp_batch_centered, 2, self.array)
            ca.mean(ca.mean(self.array, axis=0, keepdims=True), axis=(2, 3),
                    keepdims=True, out=self._tmp_batch_inv_std)
            # Calculate 1 / E([x - E(x)]^2)
            self._tmp_batch_inv_std += self.eps
            ca.sqrt(self._tmp_batch_inv_std, self._tmp_batch_inv_std)
            ca.power(self._tmp_batch_inv_std, -1, self._tmp_batch_inv_std)
            # Normalize input
            ca.multiply(self._tmp_batch_centered, self._tmp_batch_inv_std,
                        self.array)
            # Update running std
            self.running_std *= self.momentum
            ca.multiply(self._tmp_batch_inv_std, 1-self.momentum, tmp)
            self.running_std += tmp

            if self.noise_std > 0.0:
                noise = ca.random.normal(scale=self.noise_std,
                                         size=self.shape)
                ca.add(self.array, noise, self.array)

        elif self.phase == 'test':
            ca.subtract(self.x.array, self.running_mean, self.array)
            self.array *= self.running_std
        else:
            raise ValueError('Invalid phase: %s' % self.phase)
        if self.affine:
            self.array *= self.gamma.array
            self.array += self.beta.array

    def bprop(self):
        ca.multiply(self._tmp_batch_centered, self.grad_array,
                    self.x.grad_array)
        tmp = ca.mean(ca.mean(self.x.grad_array, axis=0, keepdims=True),
                      axis=(2, 3), keepdims=True)
        ca.multiply(self._tmp_batch_centered, tmp, self.x.grad_array)
        self.x.grad_array *= -1
        self.x.grad_array *= self._tmp_batch_inv_std
        self.x.grad_array *= self._tmp_batch_inv_std

        tmp = ca.mean(ca.mean(self.grad_array, axis=0, keepdims=True),
                      axis=(2, 3), keepdims=True)
        self.x.grad_array += self.grad_array
        self.x.grad_array -= tmp
        self.x.grad_array *= self._tmp_batch_inv_std

        if self.affine:
            self.x.grad_array *= self.gamma.array
            # Normalized input
            self._tmp_batch_centered *= self._tmp_batch_inv_std
            self._tmp_batch_centered *= self.grad_array
            ca.sum(ca.sum(self._tmp_batch_centered, axis=(2, 3),
                          keepdims=True), axis=0, keepdims=True,
                   out=self.gamma.grad_array)
            ca.sum(ca.sum(self.grad_array, axis=(2, 3), keepdims=True), axis=0,
                   keepdims=True, out=self.beta.grad_array)
