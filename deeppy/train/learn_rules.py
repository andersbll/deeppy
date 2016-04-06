import numpy as np
import cudarray as ca


class LearnRule(object):
    learn_rate = None

    def init_state(self, param):
        raise NotImplementedError()

    def step(self, param, state):
        raise NotImplementedError()


class Momentum(LearnRule):
    def __init__(self, learn_rate=0.01, momentum=0.9):
        self.learn_rate = learn_rate
        self.momentum = momentum

    def init_state(self, param):
        last_step = ca.zeros_like(param.grad_array)
        return last_step

    def step(self, param, last_step):
        last_step *= self.momentum
        step = param.grad()
        step *= -self.learn_rate
        last_step += step
        param.step(last_step)


class RMSProp(LearnRule):
    def __init__(self, learn_rate=0.001, decay=0.9, eps=1e-8):
        self.learn_rate = learn_rate
        self.decay = decay
        self.eps = eps

    def init_state(self, param):
        mean_square = ca.zeros_like(param.grad_array)
        return mean_square

    def step(self, param, mean_square):
        grad = param.grad()
        # mean_square = decay*mean_square + (1 - decay)*grad
        mean_square *= self.decay
        tmp = grad**2
        tmp *= (1 - self.decay)
        mean_square += tmp
        # step = -learn_rate*grad/(sqrt(mean_square) + eps)
        ca.sqrt(mean_square, tmp)
        tmp += self.eps
        ca.divide(grad, tmp, tmp)
        tmp *= -self.learn_rate
        param.step(tmp)


class Adam(LearnRule):
    def __init__(self, learn_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learn_rate = learn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_state(self, param):
        m = ca.zeros_like(param.grad_array)
        v = ca.zeros_like(param.grad_array)
        t = np.zeros(1, dtype=int)
        return m, v, t

    def step(self, param, state):
        m, v, t = state
        grad = param.grad()
        t += 1
        t = int(t)
        # m = beta1*m + (1 - beta1)*grad
        m *= self.beta1
        tmp = (1 - self.beta1)*grad
        m += tmp
        # v = beta2*v + (1 - beta2)*grad**2
        v *= self.beta2
        ca.power(grad, 2, tmp)
        tmp *= (1 - self.beta2)
        v += tmp
        # alpha = learn_rate*sqrt(1 - beta2**t)/(1 - beta1**t)
        # step = -alpha_t*m/(sqrt(v) + eps)
        alpha = self.learn_rate*np.sqrt(1 - self.beta2**t)/(1 - self.beta1**t)
        ca.sqrt(v, tmp)
        tmp += self.eps
        ca.divide(m, tmp, tmp)
        tmp *= -alpha
        param.step(tmp)
