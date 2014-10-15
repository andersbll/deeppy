import numpy as np
import cudarray as ca

import logging
logger = logging.getLogger(__name__)


class LearningRule(object):
    def _setup(self, params, batch_size):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def monitor(self):
        raise NotImplementedError()


class Momentum(LearningRule):
    def __init__(self, learn_rate, momentum):
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.batch_size = 1

    def _setup(self, params, batch_size):
        self.params = params
        self.batch_size = batch_size
        self.steps = [ca.zeros_like(p.values) for p in params]

    def step(self):
        for param, last_step in zip(self.params, self.steps):
            last_step *= self.momentum
            step = param.grad
            if param.penalty is not None:
                step -= param.penalty()
            step *= self.learn_rate*param.learn_rate/self.batch_size
            last_step += step
            p_values = param.values
            p_values -= last_step

    def monitor(self):
        for param, step in zip(self.params, self.steps):
            if param.monitor:
                val_mean_abs = np.array(ca.mean(ca.abs(param.values)))
                step_mean_abs = np.array(ca.mean(ca.abs(step)))
                logger.info('%s:\t%.1e  [%.1e]'
                            % (param.name, val_mean_abs, step_mean_abs))
