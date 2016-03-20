from ..base import CollectionMixin
from .base import Op


class Sequential(Op, CollectionMixin):
    def __init__(self, ops):
        self.collection = ops

    def __call__(self, x):
        for expr in self.collection:
            x = expr(x)
        return x
