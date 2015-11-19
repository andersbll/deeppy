"""
Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift, http://arxiv.org/abs/1502.03167

This implementation is heavily inspired by code from http://github.com/torch/nn
"""

import cudarray as ca
from ...base import ParamMixin, PhaseMixin
from ...parameter import Parameter
from ...filler import UniformFiller
from ..base import UnaryElementWise


class BatchNormalization(UnaryElementWise, ParamMixin, PhaseMixin):
    def __init__(self, momentum=0.75, eps=1e-5, affine=True):
        self.momentum = momentum
        self.eps = eps
        self.phase = 'train'
        self.affine = affine
        if self.affine:
            self.gamma = Parameter(UniformFiller(low=0, high=1))
            self.beta = Parameter(0.0)

    def setup(self):
        super(BatchNormalization, self).setup()
        if len(self.out_shape) != 2:
            raise ValueError('Only 1D data supported')
        reduced_shape = 1, self.out_shape[1]
        self.running_mean = ca.zeros(reduced_shape)
        self.running_std = ca.ones(reduced_shape)
        self._tmp_batch_centered = ca.zeros(self.out_shape)
        self._tmp_batch_inv_std = ca.zeros(reduced_shape)
        if self.affine:
            self.gamma.setup(reduced_shape)
            self.beta.setup(reduced_shape)

    @property
    def params(self):
        return self.gamma, self.beta

    @params.setter
    def params(self, params):
        self.gamma, self.beta = params

    def fprop(self):
        if self.phase == 'train':
            # Calculate batch mean
            tmp = ca.mean(self.x.out, axis=0, keepdims=True)
            # Center input
            ca.subtract(self.x.out, tmp, self._tmp_batch_centered)
            # Update running mean
            tmp *= 1 - self.momentum
            self.running_mean *= self.momentum
            self.running_mean += tmp
            # Calculate batch variance
            ca.power(self._tmp_batch_centered, 2, self.out)
            ca.mean(self.out, axis=0, keepdims=True,
                    out=self._tmp_batch_inv_std)
            # Calculate 1 / E([x - E(x)]^2)
            self._tmp_batch_inv_std += self.eps
            ca.sqrt(self._tmp_batch_inv_std, self._tmp_batch_inv_std)
            ca.power(self._tmp_batch_inv_std, -1, self._tmp_batch_inv_std)
            # Normalize input
            ca.multiply(self._tmp_batch_centered, self._tmp_batch_inv_std,
                        self.out)
            # Update running std
            self.running_std *= self.momentum
            ca.multiply(self._tmp_batch_inv_std, 1-self.momentum, tmp)
            self.running_std += tmp
        elif self.phase == 'test':
            ca.subtract(self.x.out, self.running_mean, self.out)
            self.out *= self.running_std
        else:
            raise ValueError('Invalid phase: %s' % self.phase)
        if self.affine:
            self.out *= self.gamma.array
            self.out += self.beta.array

    def bprop(self):
        ca.multiply(self._tmp_batch_centered, self.out_grad, self.x.out_grad)
        tmp = ca.mean(self.x.out_grad, axis=0, keepdims=True)
        ca.multiply(self._tmp_batch_centered, tmp, self.x.out_grad)
        self.x.out_grad *= -1
        self.x.out_grad *= self._tmp_batch_inv_std
        self.x.out_grad *= self._tmp_batch_inv_std

        ca.mean(self.out_grad, axis=0, keepdims=True, out=tmp)
        self.x.out_grad += self.out_grad
        self.x.out_grad -= tmp
        self.x.out_grad *= self._tmp_batch_inv_std

        if self.affine:
            self.x.out_grad *= self.gamma.array
            # Normalized input
            self._tmp_batch_centered *= self._tmp_batch_inv_std
            self._tmp_batch_centered *= self.out_grad
            ca.sum(self._tmp_batch_centered, axis=0, keepdims=True,
                   out=self.gamma.grad_array)
            ca.sum(self.out_grad, axis=0, keepdims=True,
                   out=self.beta.grad_array)


class SpatialBatchNormalization(UnaryElementWise, ParamMixin, PhaseMixin):
    def __init__(self, momentum=0.75, eps=1e-5, affine=True):
        self.momentum = momentum
        self.eps = eps
        self.phase = 'train'
        self.affine = affine
        if self.affine:
            self.gamma = Parameter(UniformFiller(low=0, high=1))
            self.beta = Parameter(0.0)

    def setup(self):
        super(SpatialBatchNormalization, self).setup()
        if len(self.out_shape) != 4:
            raise ValueError('Only 4D data supported')
        reduced_shape = 1, self.out_shape[1], 1, 1
        self.running_mean = ca.zeros(reduced_shape)
        self.running_std = ca.ones(reduced_shape)
        self._tmp_batch_centered = ca.zeros(self.out_shape)
        self._tmp_batch_inv_std = ca.zeros(reduced_shape)
        if self.affine:
            self.gamma.setup(reduced_shape)
            self.beta.setup(reduced_shape)

    @property
    def params(self):
        return self.gamma, self.beta

    @params.setter
    def params(self, params):
        self.gamma, self.beta = params

    def fprop(self):
        if self.phase == 'train':
            # Calculate batch mean
            tmp = ca.mean(ca.mean(self.x.out, axis=0, keepdims=True),
                          axis=(2, 3), keepdims=True)
            # Center input
            ca.subtract(self.x.out, tmp, self._tmp_batch_centered)
            # Update running mean
            tmp *= 1 - self.momentum
            self.running_mean *= self.momentum
            self.running_mean += tmp
            # Calculate batch variance
            ca.power(self._tmp_batch_centered, 2, self.out)
            ca.mean(ca.mean(self.out, axis=0, keepdims=True), axis=(2, 3),
                    keepdims=True, out=self._tmp_batch_inv_std)
            # Calculate 1 / E([x - E(x)]^2)
            self._tmp_batch_inv_std += self.eps
            ca.sqrt(self._tmp_batch_inv_std, self._tmp_batch_inv_std)
            ca.power(self._tmp_batch_inv_std, -1, self._tmp_batch_inv_std)
            # Normalize input
            ca.multiply(self._tmp_batch_centered, self._tmp_batch_inv_std,
                        self.out)
            # Update running std
            self.running_std *= self.momentum
            ca.multiply(self._tmp_batch_inv_std, 1-self.momentum, tmp)
            self.running_std += tmp
        elif self.phase == 'test':
            ca.subtract(self.x.out, self.running_mean, self.out)
            self.out *= self.running_std
        else:
            raise ValueError('Invalid phase: %s' % self.phase)
        if self.affine:
            self.out *= self.gamma.array
            self.out += self.beta.array

    def bprop(self):
        ca.multiply(self._tmp_batch_centered, self.out_grad, self.x.out_grad)
        tmp = ca.mean(ca.mean(self.x.out_grad, axis=0, keepdims=True),
                      axis=(2, 3), keepdims=True)
        ca.multiply(self._tmp_batch_centered, tmp, self.x.out_grad)
        self.x.out_grad *= -1
        self.x.out_grad *= self._tmp_batch_inv_std
        self.x.out_grad *= self._tmp_batch_inv_std

        tmp = ca.mean(ca.mean(self.out_grad, axis=0, keepdims=True),
                      axis=(2, 3), keepdims=True)
        self.x.out_grad += self.out_grad
        self.x.out_grad -= tmp
        self.x.out_grad *= self._tmp_batch_inv_std

        if self.affine:
            self.x.out_grad *= self.gamma.array
            # Normalized input
            self._tmp_batch_centered *= self._tmp_batch_inv_std
            self._tmp_batch_centered *= self.out_grad
            ca.sum(ca.sum(self._tmp_batch_centered, axis=(2, 3),
                          keepdims=True), axis=0, keepdims=True,
                   out=self.gamma.grad_array)
            ca.sum(ca.sum(self.out_grad, axis=(2, 3), keepdims=True), axis=0,
                   keepdims=True, out=self.beta.grad_array)
