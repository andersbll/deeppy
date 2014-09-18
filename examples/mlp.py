#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sklearn.datasets

import os
os.environ['deeppy_BACKEND'] = 'numpy'
import deeppy


def run():
    # Fetch data
    digits = sklearn.datasets.load_digits()
    X_train = digits.data
    X_train /= np.max(X_train)
    y_train = digits.target
    n_classes = np.unique(y_train).size

    # Setup multi-layer perceptron
    nn = deeppy.NeuralNetwork(
        layers=[
            deeppy.FullyConnected(
                n_output=50,
                weights=deeppy.NormalFiller(sigma=0.1),
                weight_decay=0.002,
            ),
            deeppy.Activation('relu'),
            deeppy.FullyConnected(
                n_output=n_classes,
                weights=deeppy.NormalFiller(sigma=0.1),
                weight_decay=0.002,
            ),
            deeppy.MultinomialLogReg(),
        ],
    )

#    # Verify network for correct back-propagation of parameter gradients
#    print('Checking gradients')
#    nn.check_gradients(X_train[:100], y_train[:100])

    # Train neural network
    print('Training neural network')
    nn.fit(X_train, y_train, learning_rate=0.1, max_iter=25, batch_size=32)

    # Evaluate on training data
    error = nn.error(X_train, y_train)
    print('Training error rate: %.4f' % error)


if __name__ == '__main__':
    run()
