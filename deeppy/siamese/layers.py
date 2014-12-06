import cudarray as ca
from ..feed_forward.layers import Layer, LossMixin


class ContrastiveLoss(Layer, LossMixin):
    def __init__(self, margin=1.0):
        self.name = 'contrastive'
        self.margin = margin

    def _setup(self, input_shape):
        self.n_classes = input_shape[1]

    def fprop(self, x1, x2, phase):
        self.last_x1 = x1
        self.last_x2 = x2
        dists = ca.sum((x1-x2)**2, axis=1, keepdims=True)
        return dists

    def input_grad(self, y, dists):
        x1 = self.last_x1
        x2 = self.last_x2
        y = ca.reshape(y, y.shape+(1,))

        grad_dists1 = 2*(x1-x2)
        genuine = y*grad_dists1
        imposter = (1-y)*(-grad_dists1)
        non_saturated_imposters = self.margin-dists > 0.0
        imposter *= non_saturated_imposters
        grad_x1 = genuine + imposter
        return grad_x1, -grad_x1

    def loss(self, y, dists):
        return y*dists + (1-y)*ca.maximum(self.margin-dists, 0)

    def output_shape(self, input_shape):
        return (input_shape[0],)
