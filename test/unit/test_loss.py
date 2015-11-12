import itertools
import numpy as np
import cudarray as ca
import deeppy as dp
from feedforward.test_layers import approx_fprime, gradclose


batch_sizes = [1, 5, 10]
n_ins = [1, 2, 8, 7]


def check_grad(loss, x0, y0, seed=1, eps=None, rtol=None, atol=None):
    '''
    Numerical gradient checking of loss functions.
    '''
    def fun(x):
        ca.random.seed(seed)
        y = np.array(loss.loss(ca.array(x), ca.array(y0))).astype(np.float_)
        return np.sum(y)

    def fun_grad(x):
        x_grad = np.array(loss.grad(ca.array(x), ca.array(y0)))
        return x_grad

    g_approx = approx_fprime(x0, fun, eps)
    g_true = fun_grad(x0)
    assert gradclose(g_true, g_approx, rtol, atol)


def test_softmaxcrossentropy():
    confs = itertools.product(batch_sizes, n_ins)
    for batch_size, n_in in confs:
        print('SoftmaxCrossEntropy: batch_size=%i, n_in=%i'
              % (batch_size, n_in))
        x_shape = (batch_size, n_in)
        x = np.random.normal(size=x_shape)
        y = np.random.randint(low=0, high=n_in, size=batch_size)
        loss = dp.SoftmaxCrossEntropy()
        loss.setup(x_shape)
        assert loss.loss(ca.array(x), ca.array(y)).shape == x_shape[:1]
        check_grad(loss, x, y)


def test_binarycrossentropy():
    confs = itertools.product(batch_sizes, n_ins)
    for batch_size, n_in in confs:
        print('BinaryCrossEntropy: batch_size=%i, n_in=%i'
              % (batch_size, n_in))
        x_shape = (batch_size, n_in)
        x = np.random.uniform(size=x_shape)
        y = np.random.uniform(size=x_shape)
        loss = dp.BinaryCrossEntropy()
        loss.setup(x_shape)
        assert loss.loss(ca.array(x), ca.array(y)).shape == x_shape[:1]
        check_grad(loss, x, y)


def test_meansquarederror():
    confs = itertools.product(batch_sizes, n_ins)
    for batch_size, n_in in confs:
        print('MeanSquaredError: batch_size=%i, n_in=%i' % (batch_size, n_in))
        x_shape = (batch_size, n_in)
        x = np.random.normal(size=x_shape)
        y = np.random.normal(size=x_shape)
        loss = dp.MeanSquaredError()
        loss.setup(x_shape)
        assert loss.loss(ca.array(x), ca.array(y)).shape == x_shape[:1]
        check_grad(loss, x, y)
