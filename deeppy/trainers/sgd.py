import time
import numpy as np
import cudarray as ca
from ..helpers import one_hot_encode, one_hot_decode

import logging
logger = logging.getLogger(__name__)


def print_params(param, step):
    if param.monitor:
        val_mean_abs = np.mean(np.abs(param.values))
        step_mean_abs = np.mean(np.abs(step))
        logger.info('%s:\t%.1e  [%.1e]'
                    % (param.name, val_mean_abs, step_mean_abs))


class StochasticGradientDescent:
    def __init__(self, batch_size, learn_rate, learn_momentum=0.95,
                 min_epochs=5, max_epochs=1000, improvement_thresh=0.995,
                 patience_incr=1.5):
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.learn_momentum = learn_momentum
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience_incr = patience_incr
        self.improvement_thresh = improvement_thresh
        self.validation = False

    def train(self, model, X, Y, X_valid=None, Y_valid=None):
        validation = X_valid is not None

        n_samples = Y.shape[0]
        n_batches = n_samples // self.batch_size

        # TODO
        Y_one_hot = one_hot_encode(Y)
        model._setup(X, Y_one_hot)
        params = model._params()
        param_steps = [ca.zeros_like(p.values) for p in params]

        n_params = np.sum([p.values.size for p in params])
        logger.info('SGD: Model contains %i parameters.' % n_params)
        logger.info('SGD: %d mini-batch gradient updates per epoch.'
                    % n_batches)

        epoch = 0
        converged = False
        patience = self.min_epochs
        best_score = np.inf
        start_time = time.clock()
        while epoch < self.max_epochs and not converged:
            epoch += 1
            batch_costs = []
            for b in range(n_batches):
                batch_begin = b * self.batch_size
                batch_end = batch_begin + self.batch_size
                X_batch = ca.array(X[batch_begin:batch_end])
                Y_batch = ca.array(Y_one_hot[batch_begin:batch_end])

                cost = np.array(model._bprop(X_batch, Y_batch))
                batch_costs.append(cost)

                # Gradient updates
                for param, last_step in zip(params, param_steps):
                    last_step *= self.learn_momentum
                    last_step -= self.learn_rate * param.gradient
                    if param.penalty_fun is not None:
                        last_step += param.penalty_fun()
                    p_values = param.values
                    p_values += last_step

            epoch_cost = np.mean(batch_costs)
            if validation:
                val_error = model.error(X_valid, Y_valid)
                if val_error < best_score:
                    improvement = val_error / best_score
                    if improvement < self.improvement_thresh:
                        # increase patience on significant improvement
                        patience = max(patience, epoch*self.patience_incr)
                    best_score = val_error
                logger.info('epoch %.2f/%.2f' % (epoch, patience)
                            + ', cost %f' % epoch_cost
                            + ', val_error %.4f' % val_error)
                for param, step in zip(params, param_steps):
                    print_params(param, step)
                if patience <= epoch:
                    logger.info('SGD: Converged on validation set.')
                    converged = True
            else:
                if epoch_cost < best_score:
                    improvement = epoch_cost / best_score
                    if improvement < self.improvement_thresh:
                        # increase patience on significant improvement
                        patience = max(patience, epoch*self.patience_incr)
                    best_score = epoch_cost
                logger.info('epoch %i/%i' % (epoch, patience)
                            + ', cost %f' % epoch_cost)
                if patience <= epoch:
                    logger.info('SGD: Converged on training set.')
                    converged = True

        end_time = time.clock()
        if not converged:
            logger.info('SGD: Stopped by max_epochs.')
        duration = float(end_time - start_time)
        logger.info('SGD: Optimization ran for %.2f minutes ' % (duration/60)
                    + '(%d epochs, %.1f s/epoch)' % (epoch, duration/epoch))
