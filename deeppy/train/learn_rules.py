import numpy as np
import cudarray as ca


class LearnRule(object):
    learn_rate = None

    def init_state(self, param):
        raise NotImplementedError()

    def step(self, param, state):
        raise NotImplementedError()


class Momentum(LearnRule):
    def __init__(self, learn_rate, momentum=0.9):
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
    def __init__(self, learn_rate, decay=0.9, eps=1e-8):
        self.learn_rate = learn_rate
        self.decay = decay
        self.eps = eps

    def init_state(self, param):
        last_step = ca.zeros_like(param.grad_array)
        return last_step

    def step(self, param, last_step):
        last_step *= self.decay
        step = param.grad()
        last_step += (1.0 - self.decay) * step**2
        scaling = ca.sqrt(last_step) + self.eps
        step *= -self.learn_rate
        step /= scaling
        param.step(step)


class Adam(LearnRule):
    def __init__(self, learn_rate, beta1=0.9, beta2=0.999, lambd=1-1e-8,
                 eps=1e-8):
        self.learn_rate = learn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambd = lambd
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
        beta1_t = self.beta1 * self.lambd**(t - 1)
        m *= beta1_t
        m += (1 - beta1_t) * grad
        v *= self.beta2
        v += (1 - self.beta2) * grad**2
        learn_rate = (self.learn_rate * (1 - self.beta2**t)**0.5 /
                      (1 - self.beta1**t))
        step = m / (ca.sqrt(v) + self.eps)
        step *= -learn_rate
        param.step(step)
