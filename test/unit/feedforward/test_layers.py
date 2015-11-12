import itertools
import numpy as np
import cudarray as ca
import deeppy as dp
from copy import copy
from deeppy.base import ParamMixin


batch_sizes = [1, 4, 5, 10, 24]
n_ins = [1, 2, 8, 7, 25]
n_outs = [1, 2, 8, 7, 25]


def allclose(a, b, rtol=None, atol=None):
    if ca.float_ == np.float32:
        rtol = 1e-04 if rtol is None else rtol
        atol = 1e-06 if atol is None else atol
    else:
        rtol = 1e-05 if rtol is None else rtol
        atol = 1e-08 if atol is None else atol
    return np.allclose(a, b, rtol, atol)


def gradclose(a, b, rtol=None, atol=None):
    if ca.float_ == np.float32:
        rtol = 1e-05 if rtol is None else rtol
        atol = 1e-07 if atol is None else atol
    else:
        rtol = 1e-05 if rtol is None else rtol
        atol = 1e-08 if atol is None else atol
    diff = abs(a - b) - atol - rtol * (abs(a) + abs(b))
    is_close = np.all(diff < 0)
    if not is_close:
        denom = abs(a) + abs(b)
        mask = denom == 0
        rel_error = abs(a - b) / (denom + mask)
        rel_error[mask] = 0
        rel_error = np.max(rel_error)
        abs_error = np.max(abs(a - b))
        print('rel_error=%.4e, abs_error=%.4e, rtol=%.2e, atol=%.2e'
              % (rel_error, abs_error, rtol, atol))
    return is_close


def approx_fprime(x, f, eps=None, *args):
    '''
    Central difference approximation of the gradient of a scalar function.
    '''
    if eps is None:
        eps = np.sqrt(np.finfo(ca.float_).eps)
    grad = np.zeros_like(x)
    step = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        step[idx] = eps * max(abs(x[idx]), 1.0)
        grad[idx] = (f(*((x+step,) + args)) -
                     f(*((x-step,) + args))) / (2*step[idx])
        step[idx] = 0.0
    return grad


def check_grad(layer, x0, seed=1, eps=None, rtol=None, atol=None):
    '''
    Numerical gradient checking of layer bprop.
    '''
    # Check input gradient
    def fun(x):
        ca.random.seed(seed)
        y = np.array(layer.fprop(ca.array(x))).astype(np.float_)
        return np.sum(y)

    def fun_grad(x):
        y = layer.fprop(ca.array(x))
        y_grad = ca.ones_like(y, dtype=ca.float_)
        x_grad = np.array(layer.bprop(y_grad))
        return x_grad

    g_approx = approx_fprime(x0, fun, eps)
    g_true = fun_grad(x0)
    assert gradclose(g_true, g_approx, rtol, atol)

    # Check parameter gradients
    if isinstance(layer, ParamMixin):
        def fun(x, p_idx):
            ca.random.seed(seed)
            param_array = layer.params[p_idx].array
            param_array *= 0
            param_array += ca.array(x)
            y = np.array(layer.fprop(ca.array(x0))).astype(np.float_)
            return np.sum(y)

        def fun_grad(x, p_idx):
            param_array = layer.params[p_idx].array
            param_array *= 0
            param_array += ca.array(x)
            out = layer.fprop(ca.array(x0))
            y_grad = ca.ones_like(out, dtype=ca.float_)
            layer.bprop(y_grad)
            param_grad = np.array(layer.params[p_idx].grad())
            return param_grad.astype(np.float_)

        for p_idx, p in enumerate(layer.params):
            x = np.array(layer.params[p_idx].array)
            g_true = fun_grad(x, p_idx)
            g_approx = approx_fprime(x, fun, eps, p_idx)
            assert gradclose(g_true, g_approx, rtol, atol)


def check_params(layer):
    assert isinstance(layer, ParamMixin)
    old_arrays = [np.copy(p.array) for p in layer.params]
    params = copy(layer.params)
    for p in params:
        a = p.array
        a *= 2
    layer.params = params
    for p, a in zip(layer.params, old_arrays):
        assert np.allclose(p.array / 2, a)


def test_fully_connected():
    confs = itertools.product(batch_sizes, n_ins, n_outs)
    for batch_size, n_in, n_out in confs:
        print('FullyConnected: batch_size=%i, n_in=%i, n_out=%i'
              % (batch_size, n_in, n_out))
        x_shape = (batch_size, n_in)
        x = np.random.normal(size=x_shape).astype(ca.float_)
        w = np.random.normal(size=(n_in, n_out)).astype(ca.float_)
        b = np.random.normal(size=n_out).astype(ca.float_)
        layer = dp.FullyConnected(n_out, weights=w, bias=b)
        layer.setup(x_shape)
        assert layer.y_shape(x_shape) == (batch_size, n_out)

        y = np.array(layer.fprop(ca.array(x)))
        assert allclose(np.dot(x, w) + b, y)

        y_grad = y
        x_grad = np.array(layer.bprop(ca.array(y_grad)))
        assert allclose(np.dot(y, w.T), x_grad)

        check_grad(layer, x)
        check_params(layer)
