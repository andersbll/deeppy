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
        penalty = param.penalty()
        if penalty is not None:
            step -= penalty
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
        penalty = param.penalty()
        if penalty is not None:
            step -= penalty
        last_step += (1.0 - self.decay) * step**2
        scaling = ca.sqrt(last_step) + self.eps
        step *= -self.learn_rate
        step /= scaling
        param.step(step)
