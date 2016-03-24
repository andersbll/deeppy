import itertools
import numpy as np
import deeppy as dp
import deeppy.expr as ex
from deeppy.misc.test import (eps, BPropableSource, approx_fprime, gradclose,
                              graph_funs)


shapes = [(1, 4), (1, 7), (2, 6), (3, 3), (3, 4), (5, 1)]


def test_unary():
    ops = [
        ex.Absolute, ex.Negative, ex.Log, ex.Exp, ex.Tanh,
    ]
    confs = itertools.product(shapes, ops)
    for shape, op in confs:
        print('%s shape=%s' % (op.__name__.ljust(20), shape))
        src_array = np.random.normal(size=shape).astype(dp.float_)

        if op is ex.Log:
            src_array = np.fabs(src_array) + eps

        src = BPropableSource(src_array)
        sink = op()(src)
        f, f_grad = graph_funs(src, sink)
        g_approx = approx_fprime(src_array, f)
        g_true = f_grad(src_array)
        assert gradclose(g_true, g_approx)


def test_binary():
    ops = [
        ex.Add, ex.Subtract, ex.Multiply, ex.Divide, ex.Power, ex.Maximum,
        ex.Minimum,
    ]
    confs = itertools.product(shapes, ops)
    for shape, op in confs:
        print('%s shape=%s' % (op.__name__.ljust(20), shape))
        lhs_array = np.random.normal(size=shape).astype(dp.float_)
        rhs_array = np.random.normal(size=shape).astype(dp.float_)
        if op is ex.Power:
            lhs_array = np.fabs(lhs_array)
        if op is ex.Divide:
            rhs_array[np.fabs(rhs_array) < eps] = eps
            rhs_array[np.fabs(rhs_array) < eps] = eps
        lhs_src = BPropableSource(lhs_array)
        rhs_src = BPropableSource(rhs_array)
        sink = op()(lhs_src, rhs_src)

        for src, x0 in [(lhs_src, lhs_array), (rhs_src, rhs_array)]:
            f, f_grad = graph_funs(src, sink)
            g_approx = approx_fprime(x0, f)
            g_true = f_grad(x0)
            assert gradclose(g_true, g_approx)


def test_expressions():
    exprs = [
        lambda a, b, c: a + b + c,
        lambda a, b, c: a - b - c,
        lambda a, b, c: a * b * c,
        lambda a, b, c: a / b / c,
        lambda a, b, c: a + b - c,
        lambda a, b, c: a - b + c,
        lambda a, b, c: a + b + (-c),
        lambda a, b, c: a + b*a + c,
        lambda a, b, c: a + b**2 + c,
        lambda a, b, c: a + b*b + a*b + c,
        lambda a, b, c: a + 2*(b + a) + c,
        lambda a, b, c: c / (a + ((b/1)*(c+1)) + a*b*c),
    ]
    confs = itertools.product(shapes, exprs)
    for shape, op in confs:
        a_array = np.random.normal(size=shape).astype(dp.float_)
        b_array = np.random.normal(size=shape).astype(dp.float_)
        c_array = np.random.normal(size=shape).astype(dp.float_)
        a_src = BPropableSource(a_array)
        b_src = BPropableSource(b_array)
        c_src = BPropableSource(c_array)
        sink = op(a_src, b_src, c_src)

        for src, x0 in [(a_src, a_array), (b_src, b_array), (c_src, c_array)]:
            f, f_grad = graph_funs(src, sink)
            g_approx = approx_fprime(x0, f)
            g_true = f_grad(x0)
            assert gradclose(g_true, g_approx)


def test_broadcast():
    shapes = [
        [(3, 4), (3, 1), (1, 4)],
        [(3, 2, 4), (3, 1, 1), (1, 1, 4)],
        [(1, 2, 1), (1, 1, 1), (3, 1, 1)],
    ]
    exprs = [
        lambda a, b, c: a + b + c,
        lambda a, b, c: a - b - c,
        lambda a, b, c: a * b * c,
        lambda a, b, c: a / b / c,
        lambda a, b, c: a + b - c,
        lambda a, b, c: a - b + c,
        lambda a, b, c: a + b + (-c),
        lambda a, b, c: a + b*a + c,
        lambda a, b, c: a + b**2 + c,
        lambda a, b, c: a + b*b + a*b + c,
        lambda a, b, c: a + 2*(b + a) + c,
        lambda a, b, c: c / (a + ((b/1)*(c+1)) + a*b*c),
    ]
    confs = itertools.product(shapes, exprs)
    for (a_shape, b_shape, c_shape), op in confs:
        a_array = np.random.normal(size=a_shape).astype(dp.float_)
        b_array = np.random.normal(size=b_shape).astype(dp.float_)
        c_array = np.random.normal(size=c_shape).astype(dp.float_)
        a_src = BPropableSource(a_array)
        b_src = BPropableSource(b_array)
        c_src = BPropableSource(c_array)
        sink = op(a_src, b_src, c_src)

        for src, x0 in [(a_src, a_array), (b_src, b_array), (c_src, c_array)]:
            f, f_grad = graph_funs(src, sink)
            g_approx = approx_fprime(x0, f)
            g_true = f_grad(x0)
            assert gradclose(g_true, g_approx)
