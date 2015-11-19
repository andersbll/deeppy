import cudarray as ca
from ..base import UnaryElementWise


class ReLU(UnaryElementWise):
    def fprop(self):
        ca.nnet.relu(self.x.out, self.out)

    def bprop(self):
        ca.nnet.relu_d(self.x.out, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


class Sigmoid(UnaryElementWise):
    def fprop(self):
        ca.nnet.sigmoid(self.x.out, self.out)

    def bprop(self):
        ca.nnet.sigmoid_d(self.x.out, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


class Softmax(UnaryElementWise):
    def fprop(self):
        self.out = ca.nnet.softmax(self.x.out)

    def bprop(self, y_grad):
        raise NotImplementedError()


class Softplus(UnaryElementWise):
    def fprop(self):
        ca.exp(self.x.out, self.out)
        # TODO: use log1p()
        self.out += 1
        ca.log(self.out, self.out)

    def bprop(self):
        ca.negative(self.x.out, self.x.out_grad)
        ca.exp(self.x.out_grad, self.x.out_grad)
        self.x.out_grad += 1
        ca.divide(1.0, self.x.out_grad, out=self.x.out_grad)
        self.x.out_grad *= self.out_grad


def relu(x):
    return ReLU()(x)


def sigmoid(x):
    return Sigmoid()(x)


def softmax(x):
    return Softmax()(x)


def softplus(x):
    return Softplus()(x)
