import cudarray as ca
from .layers import Layer, FullyConnected


class Dropout(Layer):
    def __init__(self, dropout=0.5):
        self.name = 'dropout'
        self.dropout = dropout

    def fprop(self, X, phase):
        if self.dropout > 0.0:
            if phase == 'train':
                self.mask = self.dropout < ca.random.uniform(size=X.shape)
                Y = X * self.mask
            elif phase == 'test':
                Y = X * (1.0 - self.dropout)
        return Y

    def bprop(self, Y_grad):
        if self.dropout > 0.0:
            return Y_grad * self.mask
        else:
            return Y_grad

    def output_shape(self, input_shape):
        return input_shape


class DropoutFullyConnected(FullyConnected):
    def __init__(self, n_output, weights, bias=0.0, weight_decay=0.0,
                 dropout=0.5):
        super(DropoutFullyConnected, self).__init__(
            n_output=n_output, weights=weights, bias=bias,
            weight_decay=weight_decay
        )
        self.name = 'fc_drop'
        self.dropout = dropout

    def fprop(self, X, phase):
        Y = super(DropoutFullyConnected, self).fprop(X, phase)
        if self.dropout > 0.0:
            if phase == 'train':
                self.mask = self.dropout < ca.random.uniform(size=Y.shape)
                Y *= self.mask
            elif phase == 'test':
                Y *= (1.0 - self.dropout)
        return Y

    def bprop(self, y_grad, to_x=True):
        if self.dropout > 0.0:
            y_grad *= self.mask
        return super(DropoutFullyConnected, self).bprop(y_grad, to_x)
