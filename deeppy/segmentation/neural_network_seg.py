import warnings
import numpy as np
import cudarray as ca
import itertools
from .layers_seg import ParamMixin_seg

import logging
logger = logging.getLogger(__name__)


class NeuralNetwork_seg:
    def __init__(self, layers):
        self._initialized = False
        self.layers = layers
        self.bprop_until = next(idx for idx, layer in enumerate(layers)
                                if isinstance(layer, ParamMixin_seg))

    def _setup(self, X, Y):
        # Setup layers sequentially
        if self._initialized:
            return

        print "Setting UP"
        next_shape = X.shape[1:]
        input_index = None

        if input_index == None:
            img_h, img_w = X.shape[-2:]
            input_index = np.arange(img_h*img_w, dtype=np.int_)
            input_index = input_index.reshape((1,)+X.shape[-2:])

        for layer in self.layers:
            layer._setup(next_shape)
            next_shape = layer.output_shape(next_shape)
            input_index = layer.output_index(input_index)

        if next_shape != Y.shape[1:]:
            warnings.warn('Output shape %s does not match Y %s. Y will be masked'
                             % (next_shape, Y.shape))

        self._initialized = True

    def _params(self):
        all_params = [layer.params() for layer in self.layers
                      if isinstance(layer, ParamMixin_seg)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    def _bprop(self, X, Y):
        X = np.reshape(X, X.shape[1:])
        Y = np.reshape(Y, Y.shape[1:])
        # Forward propagation
        #print "Train"
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next, 'train')

        #print layer.name
        #print X_next[0:10,:]
        Y_pred = X_next
        ###print Y_pred
        #Y_pre_decoded = ca.nnet.one_hot_decode(Y_pred)
        #print "Train"
        #print ("predict: class1: %d, class2:%d" % (np.sum(Y_pre_decoded), abs(Y_pre_decoded.size - np.sum(Y_pre_decoded))))
        #print ("True: class1: %d, class2:%d" % (np.sum(Y), abs(Y.size - np.sum(Y))))
        # Back propagation of partial derivatives
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        layers = self.layers[self.bprop_until:-1]
        for layer in reversed(layers):
            next_grad = layer.bprop(next_grad)

        return self.layers[-1].loss(Y, Y_pred)

    def _loss(self, X, Y):
        X = np.reshape(X, X.shape[1:])
        Y = np.reshape(Y, Y.shape[1:])
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next, 'test')
        Y_pred = X_next
        return self.layers[-1].loss(Y, Y_pred)

    def predict(self, X, Y_shape, batch_size=1):
        """ Calculate an output Y for the given input X. """
        if batch_size == 0:
            batch_size = X.shape[0]
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        Y_pred = np.empty(Y_shape)
        for b in range(n_batches):
            X_next = ca.array(X[b])
            ##print "--------------"
            ##print "start"
            for layer in self.layers[:-1]:
                ##print X_next
                X_next = layer.fprop(X_next, 'test')
                ##print "--------------"
                ##print layer.name

            ##print X_next
            Y_pred_batch = self.layers[-1].predict(X_next)
            Y_pred[b] = (Y_pred_batch)
        return Y_pred

    def error(self, X, Y, batch_size=1):
        #X = np.reshape(X, X.shape[1:])
        #Y = np.reshape(Y, Y.shape[1:])
        """ Calculate error on the given data. """
        ##print "Test"
        Y_pred = self.predict(X, Y.shape, batch_size)
        logger.info('Test')
        logger.info('Predict: class1: %d, class2:%d' % (np.sum(Y_pred), abs(Y_pred.size - np.sum(Y_pred))))
        logger.info('True: class1: %d, class2:%d' % (np.sum(Y), abs(Y.size - np.sum(Y))))
        error = Y_pred != Y

        return np.mean(error)
