import itertools
import numpy as np
import deeppy as dp
import deeppy.expr as ex
from deeppy.misc.test import (eps, BPropableSource, approx_fprime, gradclose,
                              graph_funs)


shapes = [(1, 4), (1, 7), (2, 6), (3, 3), (3, 4), (5, 1)]


def test_activations():
    ops = [
        ex.nnet.LeakyReLU, ex.nnet.ReLU, ex.nnet.Sigmoid, ex.nnet.Softmax,
        ex.nnet.Softplus,
    ]
    confs = itertools.product(shapes, ops)
    for shape, op in confs:
        print('%s shape=%s' % (op.__name__.ljust(20), shape))
        src_array = np.random.normal(size=shape).astype(dp.float_)

        if op is ex.nnet.LeakyReLU:
            src_array[np.fabs(src_array) < eps] = eps
        if op is ex.nnet.Softmax:
            src_array *= 100

        src = BPropableSource(src_array)
        sink = op()(src)
        f, f_grad = graph_funs(src, sink)
        g_approx = approx_fprime(src_array, f)
        g_true = f_grad(src_array)
        assert gradclose(g_true, g_approx)
