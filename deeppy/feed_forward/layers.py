import numpy as np
import cudarray as ca
from ..fillers import filler, ConstantFiller

class Layer(object):
    def _setup(self, input_shape):
        """ Setup layer with parameters that are unknown at __init__(). """
        pass

    def fprop(self, X):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, Y_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

    def output_shape(self, input_shape):
        """ Calculate shape of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Calculate layer output for given input (prediction). """
        raise NotImplementedError()


class LossMixin(object):
    def loss(self, Y_true, Y_pred):
        """ Calculate mean loss given output and predicted output. """
        raise NotImplementedError()

    def input_grad(self, Y_true, Y_pred):
        """ Calculate input gradient given output and predicted output. """
        raise NotImplementedError()


class ParamMixin(object):
    def params(self):
        """ Layer parameters. """
        raise NotImplementedError()

    def param_grads(self):
        """ Get layer parameter gradients as calculated from bprop(). """
        raise NotImplementedError()

    def param_incs(self):
        """ Get layer parameter steps as calculated from bprop(). """
        raise NotImplementedError()


class FullyConnected(Layer, ParamMixin):

    def __init__(self, n_output, weights, bias=0.0, weight_decay=0.0, momentum = 0, droprate = 0):
        self.n_output = n_output
        self.weight_filler = filler(weights)
        self.bias_filler = filler(bias)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.droprate_filler = ConstantFiller(droprate)

    def _setup(self, input_shape):
        n_input = input_shape[1]
        W_shape = (n_input, self.n_output)
        b_shape = self.n_output
        self.W = ca.array(self.weight_filler.array(W_shape))
        self.b = ca.array(self.bias_filler.array(b_shape))
        self.droprate = ca.array(self.droprate_filler.array(n_input))
        self.dW = 0

    def fprop(self, X):
        #get random mask from droprate
        self.mask = self.droprate < ca.random.random(self.droprate.shape)
        self.last_X = ca.multiply(X, self.mask)
        return ca.dot(self.last_X, self.W) + self.b

    def predict(self, X):
        expectation = (ca.ones(self.droprate.shape) - self.droprate)
        expectated_W = ca.transpose(ca.multiply(expectation, ca.transpose(self.W)))
        return ca.dot(X, expectated_W) + self.b

    def bprop(self, Y_grad):
        n = Y_grad.shape[0]
        self.dW = ca.dot(self.last_X.T, Y_grad)/n - self.weight_decay*self.W + ca.multiply(self.dW, self.momentum)
        self.db = ca.mean(Y_grad, axis=0)
        return ca.multiply(ca.dot(Y_grad, self.W.T), self.mask)

    def params(self):
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def param_grads(self):
        # XXX: stupid: undo weight decay to get gradient
        gW = self.dW+self.weight_decay*self.W
        return gW, self.db

    def output_shape(self, input_shape):
        return (input_shape[0], self.n_output)


class Activation(Layer):
    def __init__(self, type):
        if type == 'sigmoid':
            self.fun = ca.nnet.sigmoid
            self.fun_d = ca.nnet.sigmoid_d
        elif type == 'relu':
            self.fun = ca.nnet.relu
            self.fun_d = ca.nnet.relu_d
        elif type == 'tanh':
            self.fun = ca.nnet.tanh
            self.fun_d = ca.nnet.tanh_d
        else:
            raise ValueError('Invalid activation function.')

    def fprop(self, X):
        self.last_X = X
        return self.fun(X)

    def predict(self, X):
        return self.fun(X)

    def bprop(self, Y_grad):
        return Y_grad*self.fun_d(self.last_X)

    def output_shape(self, input_shape):
        return input_shape


class MultinomialLogReg(Layer, LossMixin):
    """ Multinomial logistic regression with a cross-entropy loss function. """
    def fprop(self, X):
        return ca.nnet.softmax(X)

    def predict(self, X):
        return ca.nnet.softmax(X)

    def bprop(self, Y_grad):
        raise NotImplementedError(
            'MultinomialLogReg does not support back-propagation of gradients.'
            + ' It should occur only as the last layer of a NeuralNetwork.'
        )

    def input_grad(self, y, y_pred):
        # Assumes one-hot encoding.
        return -(y - y_pred)

    def loss(self, y, y_pred):
        # Assumes one-hot encoding.
        return ca.nnet.categorical_cross_entropy(y, y_pred)

    def output_shape(self, input_shape):
        return input_shape
