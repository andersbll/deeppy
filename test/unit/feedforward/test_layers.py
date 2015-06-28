import itertools
import numpy as np
import scipy.optimize
import cudarray as ca
import deeppy as dp
from copy import copy
from deeppy.base import ParamMixin


batch_sizes = [1, 4, 5, 10, 24]
n_ins = [1, 2, 8, 7, 25]
n_outs = [1, 2, 8, 7, 25]


def allclose(a, b, rtol=None, atol=None):
    if ca.float_ == np.float32:
        # Higher tolerances when CUDArray uses a float32 backend.
        rtol = 1e-04 if rtol is None else rtol
        atol = 1e-06 if atol is None else atol
    else:
        rtol = 1e-05 if rtol is None else rtol
        atol = 1e-08 if atol is None else atol
    return np.allclose(a, b, rtol, atol)


def grad_close(func, grad, x0, eps, *args):
    if ca.float_ == np.float64:
        rtol, atol = 1e-04, 1e-05
    else:
        rtol, atol = 1e-03, 1e-03
    grad_true = grad(x0, *args)
    grad_approx = scipy.optimize.approx_fprime(x0, func, eps, *args)
    return allclose(grad_true, grad_approx, rtol=rtol, atol=atol)


def check_grad(layer, x0, eps=None, seed=1):
    if eps is None:
        eps = np.sqrt(np.finfo(ca.float_).eps)

    # Check input gradient
    def func(x):
        ca.random.seed(seed)
        x = np.reshape(x, x0.shape)
        y = np.array(layer.fprop(ca.array(x), 'train')).astype(np.float_)
        return np.sum(y)

    def grad(x):
        ca.random.seed(seed)
        x = np.reshape(x, x0.shape)
        y = layer.fprop(ca.array(x), 'train')
        y_grad = ca.ones_like(y, dtype=ca.float_)
        x_grad = np.array(layer.bprop(y_grad))
        return np.ravel(x_grad)

    assert grad_close(func, grad, np.ravel(x0), eps)

    # Check parameter gradients
    if isinstance(layer, ParamMixin):
        def func(x, *args):
            ca.random.seed(seed)
            p_idx = args[0]
            param_vals = layer._params[p_idx].array
            param_vals *= 0
            param_vals += ca.array(np.reshape(x, param_vals.shape))
            y = np.array(layer.fprop(ca.array(x0), 'train')).astype(np.float_)
            return np.sum(y)

        def grad(x, *args):
            ca.random.seed(seed)
            p_idx = args[0]
            param_vals = layer._params[p_idx].array
            param_vals *= 0
            param_vals += ca.array(np.reshape(x, param_vals.shape))
            out = layer.fprop(ca.array(x0), 'train')
            y_grad = ca.ones_like(out, dtype=ca.float_)
            layer.bprop(y_grad)
            param_grad = layer._params[p_idx].grad()
            return np.ravel(np.array(param_grad))

        for p_idx, p in enumerate(layer._params):
            args = p_idx
            x = np.array(layer._params[p_idx].array)
            assert grad_close(func, grad, np.ravel(x), eps, args)


def check_params(layer):
    assert isinstance(layer, ParamMixin)
    old_arrays = [np.copy(p.array) for p in layer._params]
    params = copy(layer._params)
    for p in params:
        a = p.array
        a *= 2
    layer._params = params
    for p, a in zip(layer._params, old_arrays):
        assert np.allclose(p.array / 2, a)


def test_fully_connected():
    layer_confs = itertools.product(batch_sizes, n_ins, n_outs)
    for batch_size, n_in, n_out in layer_confs:
        print('FullyConnected: batch_size=%i, n_in=%i, n_out=%i'
              % (batch_size, n_in, n_out))
        x_shape = (batch_size, n_in)
        x = np.random.normal(size=x_shape).astype(ca.float_)
        w = np.random.normal(size=(n_in, n_out)).astype(ca.float_)*1e-4
        b = np.random.normal(size=n_out).astype(ca.float_)*1e-4
        layer = dp.FullyConnected(n_out, weights=w, bias=b)
        layer._setup(x_shape)
        assert layer.y_shape(x_shape) == (batch_size, n_out)

        phase = 'train'
        y = np.array(layer.fprop(ca.array(x), phase))
        assert allclose(np.dot(x, w) + b, y)

        y_grad = y
        x_grad = np.array(layer.bprop(ca.array(y_grad)))
        assert allclose(np.dot(y, w.T), x_grad)

        check_grad(layer, x)
        check_params(layer)


def test_activation():
    activations = ['sigmoid', 'tanh', 'relu']
    layer_confs = itertools.product(batch_sizes, n_ins, activations)
    for batch_size, n_in, activation in layer_confs:
        print('Activation: batch_size=%i, n_in=%i, fun=%s'
              % (batch_size, n_in, activation))
        x_shape = (batch_size, n_in)
        x = np.random.normal(size=x_shape)
        if activation == 'relu':
            # Change x values that are too close to 0. The numeric
            # differentiation may make such values change sign resulting in
            # faulty gradient approximation.
            eps = 1e3
            x[np.logical_and(-eps < x, x < eps)] = 0.1
        layer = dp.Activation(activation)
        layer._setup(x_shape)

        assert layer.y_shape(x_shape) == x_shape

        check_grad(layer, x)
