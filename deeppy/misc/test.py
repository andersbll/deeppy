import numpy as np
import cudarray as ca
from .. import expr as ex


if ca.float_ == np.float32:
    eps = 1e-04
else:
    eps = 1e-06


def allclose(a, b, rtol=None, atol=None):
    if ca.float_ == np.float32:
        rtol = 1e-03 if rtol is None else rtol
        atol = 1e-04 if atol is None else atol
    else:
        rtol = 1e-05 if rtol is None else rtol
        atol = 1e-08 if atol is None else atol
    return np.allclose(a, b, rtol, atol)


def gradclose(a, b, rtol=None, atol=None):
    if ca.float_ == np.float32:
        rtol = 1e-03 if rtol is None else rtol
        atol = 1e-04 if atol is None else atol
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


class BPropableSource(ex.base.Op, ex.base.NoBPropMixin, ex.base.NoFPropMixin):
    bpropable = True

    def __init__(self, array):
        if isinstance(array, np.ndarray):
            array = ca.array(array)
        self.shape = array.shape
        self.array = array
        self.grad_array = ca.zeros(self.shape)


def graph_funs(src, sink, seed=1):
    graph = ex.graph.ExprGraph(sink)
    graph.setup()

    def fun(x):
        ca.random.seed(seed)
        src.array = ca.array(x)
        graph.fprop()
        y = np.array(sink.array).astype(np.float_)
        return np.sum(y)

    def fun_grad(x):
        ca.random.seed(seed)
        src.array = ca.array(x)
        graph.fprop()
        sink.grad_array = ca.ones(sink.shape, dtype=ca.float_)
        graph.bprop()
        x_grad = np.array(src.grad_array)
        return x_grad

    return fun, fun_grad
