#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sklearn.datasets
import deeppy as dp


def run():
    # Fetch data
    digits = sklearn.datasets.load_digits()
    X_train = digits.data
    X_train /= np.max(X_train)
    y_train = digits.target
    n_classes = np.unique(y_train).size

    # Setup multi-layer perceptron
    nn = dp.NeuralNetwork(
        layers=[
            dp.FullyConnected(
                n_output=50,
                weights=dp.NormalFiller(sigma=0.1),
                weight_decay=0.000001,
            ),
            dp.Activation('sigmoid'),
            dp.FullyConnected(
                n_output=n_classes,
                weights=dp.NormalFiller(sigma=0.1),
                weight_decay=0.000001,
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    print('Training neural network')
    trainer = dp.StochasticGradientDescent(
        batch_size=32, learn_rate=0.05, learn_momentum=0.95, max_epochs=25
    )

    trainer.train(nn, X_train, y_train)

    # Evaluate on training data
    error = nn.error(X_train, y_train)
    print('Training error rate: %.4f' % error)


if __name__ == '__main__':
    run()
