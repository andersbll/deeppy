import time
import numpy as np
import cudarray as ca
from ..feed import Feed
from ..parameter import SharedParameter

import logging
log = logging.getLogger(__name__)


class GradientDescent(object):
    def __init__(self, model, feed, learn_rule):
        self.feed = Feed.from_any(feed)
        self.learn_rule = learn_rule
        self.model = model
        self.params = None
        self.learn_rule_states = None
        self.reset()

    def reset(self):
        self.feed.reset()
        self.model.setup(**self.feed.shapes)
        self.params = [p for p in self.model.params
                       if not isinstance(p, SharedParameter)]
        self.learn_rule_states = [self.learn_rule.init_state(p)
                                  for p in self.params]
        n_params = np.sum([p.array.size for p in self.params])
        log.info('SGD: Model contains %i parameters.', n_params)
        log.info('SGD: %d gradient updates per epoch.', self.feed.epoch_size)

    def train_epoch(self):
        batch_losses = []
        for batch in self.feed.batches():
            loss = np.array(ca.mean(self.model.update(**batch)))
            for param, state in zip(self.params, self.learn_rule_states):
                self.learn_rule.step(param, state)
            batch_losses.append(loss)
        epoch_loss = np.mean(batch_losses)
        return epoch_loss

    def train_epochs(self, n_epochs, annealer=None, error_fun=None):
        self.train_patience(annealer, error_fun, min_epochs=n_epochs,
                            max_epochs=n_epochs)

    def train_patience(self, annealer=None, error_fun=None, min_epochs=5,
                       max_epochs=1000, improvement_thresh=0.995,
                       patience_incr=1.5):
        epoch = 0
        converged = False
        patience = min_epochs
        best_score = np.inf
        start_time = time.clock()
        while epoch < max_epochs and not converged:
            epoch += 1
            epoch_loss = self.train_epoch()
            if error_fun is None:
                epoch_error = epoch_loss
            else:
                epoch_error = error_fun()
            if epoch_error < best_score:
                improvement = epoch_error / best_score
                if improvement < improvement_thresh:
                    # increase patience on significant improvement
                    patience = max(patience, epoch*patience_incr)
                best_score = epoch_error
            if error_fun is None:
                log.info('epoch %d/%d, loss %f', epoch, patience, epoch_loss)
            else:
                log.info('epoch %d/%d, loss %f, error %.4f', epoch, patience,
                         epoch_loss, epoch_error)
            for param in self.params:
                param.monitor()
            if patience < epoch:
                log.info('SGD: Converged.')
                converged = True
            if annealer is not None:
                self.learn_rule.learn_rate = annealer.value(epoch)
        end_time = time.clock()
        if not converged:
            log.info('SGD: Stopped by max_epochs.')
        duration = float(end_time - start_time)
        log.info('SGD: Optimization ran for %.2f minutes (%d epochs, '
                 '%.1f s/epoch)', duration/60, epoch, duration/epoch)
