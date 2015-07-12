import cudarray as ca
from ..loss import Loss


class ContrastiveLoss(Loss):
    def __init__(self, margin=1.0):
        self.name = 'contrastive'
        self.margin = margin
        self._tmp_x1 = None
        self._tmp_x2 = None
        self._tmp_dists = None

    def fprop(self, x1, x2):
        if self._tmp_x1 is not x1 or self._tmp_x2 is not x2:
            self._tmp_dists = ca.sum((x1-x2)**2, axis=1, keepdims=True)
            self._tmp_x1 = x1
            self._tmp_x2 = x2
        return self._tmp_dists

    def loss(self, target, x1, x2):
        dists = self.fprop(x1, x2)
        return target*dists + (1-target)*ca.maximum(self.margin-dists, 0)

    def grad(self, target, x1, x2):
        dists = self.fprop(x1, x2)
        target = ca.reshape(target, target.shape+(1,))

        grad_dists1 = 2*(x1-x2)
        genuine = target*grad_dists1
        imposter = (1-target)*(-grad_dists1)
        non_saturated_imposters = self.margin-dists > 0.0
        imposter *= non_saturated_imposters
        grad_x1 = genuine + imposter
        return grad_x1, -grad_x1

    def y_shape(self, x_shape):
        return (x_shape[0],)
