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
            step_rate = self.learn_rate * param.learn_rate / self.batch_size
            step *= step_rate
            last_step += step
            p_values = param.values
            p_values -= last_step

    def monitor(self):
        for param, step in zip(self.params, self.steps):
            if param.monitor:
                val_mean_abs = np.array(ca.mean(ca.fabs(param.values)))
                step_mean_abs = np.array(ca.mean(ca.fabs(step)))
                logger.info('%s:\t%.1e  [%.1e]'
                            % (param.name, val_mean_abs, step_mean_abs))


class RMSProp(LearningRule):
    def __init__(self, learn_rate, decay=0.9, max_scaling=1e3):
        self.learn_rate = learn_rate
        self.decay = decay
        self.batch_size = 1
        self.max_scaling_inv = 1./max_scaling

    def _setup(self, params, batch_size):
        self.params = params
        self.batch_size = batch_size
        self.steps = [ca.zeros_like(p.values) for p in params]

    def step(self):
        for param, rms_grad in zip(self.params, self.steps):
            rms_grad *= self.decay
            step = param.grad
            if param.penalty is not None:
                step -= param.penalty()
            rms_grad += (1.0 - self.decay) * step**2
            scaling = ca.maximum(ca.sqrt(rms_grad), self.max_scaling_inv)
            step_rate = self.learn_rate * param.learn_rate / self.batch_size
            p_values = param.values
            p_values -= step / scaling * step_rate

    def monitor(self):
        for param, step in zip(self.params, self.steps):
            if param.monitor:
                val_mean_abs = np.array(ca.mean(ca.fabs(param.values)))
                step_mean_abs = np.array(ca.mean(ca.fabs(step)))
                logger.info('%s:\t%.1e  [%.1e]'
                            % (param.name, val_mean_abs, step_mean_abs))
