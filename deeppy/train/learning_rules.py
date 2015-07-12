import cudarray as ca


class LearningRule(object):
    def _setup(self, params, batch_size):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()


class Momentum(LearningRule):
    def __init__(self, learn_rate, momentum):
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.batch_size = None
        self.params = None
        self.steps = None

    def _setup(self, params, batch_size):
        self.batch_size = batch_size
        self.params = params
        self.steps = [ca.zeros_like(param.grad_array) for param in params]

    def step(self):
        for param, last_step in zip(self.params, self.steps):
            last_step *= self.momentum
            step = param.grad()
            penalty = param.penalty()
            if penalty is not None:
                step -= penalty
            step_rate = self.learn_rate * param.learn_rate / self.batch_size
            step *= -step_rate
            last_step += step
            param.step(last_step)


class RMSProp(LearningRule):
    def __init__(self, learn_rate, decay=0.9, max_scaling=1e3):
        self.learn_rate = learn_rate
        self.decay = decay
        self.max_scaling_inv = 1./max_scaling
        self.batch_size = None
        self.params = None
        self.steps = None

    def _setup(self, params, batch_size):
        self.batch_size = batch_size
        self.params = params
        self.steps = [ca.zeros_like(param.grad_array) for param in params]

    def step(self):
        for param, rms_grad in zip(self.params, self.steps):
            rms_grad *= self.decay
            step = param.grad()
            penalty = param.penalty()
            if penalty is not None:
                step -= penalty
            rms_grad += (1.0 - self.decay) * step**2
            scaling = ca.maximum(ca.sqrt(rms_grad), self.max_scaling_inv)
            step_rate = self.learn_rate * param.learn_rate / self.batch_size
            param.step(step / scaling * (-step_rate))
