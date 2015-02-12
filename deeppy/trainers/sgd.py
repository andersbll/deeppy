import time
import numpy as np
import cudarray as ca

import logging
logger = logging.getLogger(__name__)

from ..input import to_input


class StochasticGradientDescent:
    def __init__(self, learn_rule, min_epochs=5, max_epochs=1000,
                 improvement_thresh=0.995, patience_incr=1.5):
        self.learn_rule = learn_rule
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience_incr = patience_incr
        self.improvement_thresh = improvement_thresh

    def train(self, model, input, valid_error_fun=None):
        input = to_input(input)
        model._setup(input)
        params = model._params()
        #self.learn_rule._setup(params, input.batch_size)
        #Hack because each pixels is a instance, thus batch size is
        #the number of pixels
        self.learn_rule._setup(params, input.y_shape[0])
        n_params = np.sum([p.array.size for p in params])
        logger.info('SGD: Model contains %i parameters.' % n_params)
        logger.info('SGD: %d mini-batch gradient updates per epoch.'
                    % input.n_batches)
        epoch = 0
        converged = False
        patience = self.min_epochs
        best_score = np.inf
        start_time = time.clock()
        while epoch < self.max_epochs and not converged:
            epoch += 1

            batch_costs = []
            for batch in input.supervised_batches():
                cost = np.array(ca.mean(model._update(batch)))
                batch_costs.append(cost)
                # Update gradient
                self.learn_rule.step()

            epoch_cost = np.mean(batch_costs)
            if valid_error_fun is not None:
                val_error = valid_error_fun(best_score, epoch)

                if val_error < best_score:
                    improvement = val_error / best_score
                    if improvement < self.improvement_thresh:
                        # increase patience on significant improvement
                        patience = max(patience, epoch*self.patience_incr)
                    best_score = val_error
                logger.info('epoch %d/%d' % (epoch, patience)
                            + ', cost %f' % epoch_cost
                            + ', val_error %.4f' % val_error)
                for p in params:
                    p.monitor()
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
                for p in params:
                    p.monitor()
                if patience <= epoch:
                    logger.info('SGD: Converged on training set.')
                    converged = True

        end_time = time.clock()
        if not converged:
            logger.info('SGD: Stopped by max_epochs.')
        duration = float(end_time - start_time)
        logger.info('SGD: Optimization ran for %.2f minutes ' % (duration/60)
                    + '(%d epochs, %.1f s/epoch)' % (epoch, duration/epoch))
