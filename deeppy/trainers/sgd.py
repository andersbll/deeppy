import time
import numpy as np
import cudarray as ca

import logging
logger = logging.getLogger(__name__)


class StochasticGradientDescent:
    def __init__(self, batch_size, learn_rule, learn_momentum=0.95,
                 min_epochs=5, max_epochs=1000, improvement_thresh=0.995,
                 patience_incr=1.5):
        self.batch_size = batch_size
        self.learn_rule = learn_rule
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience_incr = patience_incr
        self.improvement_thresh = improvement_thresh

    def train(self, model, X, Y, valid_error_fun=None):
        n_samples = Y.shape[0]
        n_batches = n_samples // self.batch_size

        model._setup(X, Y)
        params = model._params()
        self.learn_rule._setup(params, self.batch_size)
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
                Y_batch = ca.array(Y[batch_begin:batch_end])
                cost = np.array(model._bprop(X_batch, Y_batch))
                batch_costs.append(cost)
                # Gradient updates
                self.learn_rule.step()

            epoch_cost = np.mean(batch_costs)
            if valid_error_fun is not None:
                val_error = valid_error_fun()
                model._setup(X[:self.batch_size], Y[:self.batch_size])
                if val_error < best_score:
                    improvement = val_error / best_score
                    if improvement < self.improvement_thresh:
                        # increase patience on significant improvement
                        patience = max(patience, epoch*self.patience_incr)
                    best_score = val_error
                logger.info('epoch %d/%d' % (epoch, patience)
                            + ', cost %f' % epoch_cost
                            + ', val_error %.4f' % val_error)
                self.learn_rule.monitor()
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
                logger.info('epoch %d/%d' % (epoch, patience)
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
