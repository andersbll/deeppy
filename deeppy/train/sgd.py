import time
import numpy as np
import cudarray as ca
from ..input import Input
from ..parameter import SharedParameter

import logging
log = logging.getLogger(__name__)


class StochasticGradientDescent(object):
    def __init__(self, learn_rule, min_epochs=5, max_epochs=1000,
                 improvement_thresh=0.995, patience_incr=1.5):
        self.learn_rule = learn_rule
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience_incr = patience_incr
        self.improvement_thresh = improvement_thresh

    def train(self, model, input, error_fun=None):
        input = Input.from_any(input)
        model.setup(**input.shapes)
        params = [p for p in model.params
                  if not isinstance(p, SharedParameter)]
        self.learn_rule.learn_rate /= input.batch_size
        learn_rule_states = [self.learn_rule.init_state(p) for p in params]
        n_params = np.sum([p.array.size for p in params])
        log.info('SGD: Model contains %i parameters.', n_params)
        log.info('SGD: %d gradient updates per epoch.', input.n_batches)

        epoch = 0
        converged = False
        patience = self.min_epochs
        best_score = np.inf
        start_time = time.clock()
        while epoch < self.max_epochs and not converged:
            epoch += 1

            batch_losses = []
            for batch in input.batches():
                loss = np.array(ca.mean(model.update(**batch)))
                batch_losses.append(loss)
                # Update gradient
                for param, state in zip(params, learn_rule_states):
                    self.learn_rule.step(param, state)

            epoch_loss = np.mean(batch_losses)
            if error_fun is not None:
                error = error_fun()
                if error < best_score:
                    improvement = error / best_score
                    if improvement < self.improvement_thresh:
                        # increase patience on significant improvement
                        patience = max(patience, epoch*self.patience_incr)
                    best_score = error
                log.info('epoch %d/%d, loss %f, error %.4f', epoch,
                         patience, epoch_loss, error)
                for param in params:
                    param.monitor()
                if patience <= epoch:
                    log.info('SGD: Converged on validation set.')
                    converged = True
            else:
                if epoch_loss < best_score:
                    improvement = epoch_loss / best_score
                    if improvement < self.improvement_thresh:
                        # increase patience on significant improvement
                        patience = max(patience, epoch*self.patience_incr)
                    best_score = epoch_loss
                log.info('epoch %d/%d, loss %f', epoch, patience, epoch_loss)
                for param in params:
                    param.monitor()
                if patience <= epoch:
                    log.info('SGD: Converged on training set.')
                    converged = True

        end_time = time.clock()
        if not converged:
            log.info('SGD: Stopped by max_epochs.')
        duration = float(end_time - start_time)
        log.info('SGD: Optimization ran for %.2f minutes (%d epochs, '
                 '%.1f s/epoch)', duration/60, epoch, duration/epoch)
