#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import sklearn.datasets
import deeppy as dp


def run():
    # Fetch data
    digits = sklearn.datasets.load_digits()
    X_train = digits.data.astype(dp.float_)
    X_train /= np.max(X_train)
    y_train = digits.target.astype(dp.int_)
    n_classes = np.unique(y_train).size

    # Setup multi-layer perceptron
    nn = dp.NeuralNetwork(
        layers=[
            dp.FullyConnected(
                n_output=50,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.000001)),
            ),
            dp.Activation('sigmoid'),
            dp.FullyConnected(
                n_output=n_classes,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.000001)),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    trainer = dp.StochasticGradientDescent(
        batch_size=32,
        max_epochs=25,
        learn_rule=dp.Momentum(learn_rate=0.1, momentum=0.95),
    )
    trainer.train(nn, X_train, y_train)

    # Evaluate on training data
    error = nn.error(X_train, y_train)
    print('Training error rate: %.4f' % error)


if __name__ == '__main__':
    run()
