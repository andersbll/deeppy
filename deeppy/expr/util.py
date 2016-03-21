import numpy as np
from .base import Op, _require_op


_measures = {
    'mean': ('%.2e', np.mean),
    'std': ('%.2e', np.std),
    'shape': ('%s', lambda x: str(x.shape)),
    'absnorm': ('%.2e', lambda x: np.sum(np.fabs(x))),
}


class Print(Op):
    def __init__(self, rate=1, label=None, fprop=True, bprop=False,
                 measures={}):
        self.i = 0
        self.rate = rate
        self.label = label
        self.print_fprop = fprop
        self.print_bprop = bprop
        self.measures = measures

    def __call__(self, x):
        x = _require_op(x)
        self.x = x
        self.inputs = [x]
        self.bpropable = x.bpropable or self.print_bprop
        return self

    def setup(self):
        self.shape = self.x.shape
        self.array = self.x.array
        if self.bpropable:
            self.grad_array = self.x.grad_array
        if self.label is None:
            self.label = self.x.__class__.__name__

    def _message(self, val):
        msg = self.label + ' '
        for name, (s, fun) in dict(_measures, **self.measures).items():
            msg += ' ' + name + ':' + (s % fun(val))
        return msg

    def fprop(self):
        self.array = self.x.array
        self.i += 1
        if self.print_fprop and (self.i-1) % self.rate == 0:
            print(self._message(np.array(self.array)))

    def bprop(self):
        if self.print_bprop and (self.i-1) % self.rate == 0:
            print(self._message(np.array(self.grad_array)))
        if self.x.bpropable:
            self.x.grad_array = self.grad_array
