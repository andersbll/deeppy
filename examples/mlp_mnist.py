#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import sklearn.datasets
import deeppy as dp


def run():
    # Fetch data
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')

    X = mnist.data.astype(dp.float_)/255.0
    y = mnist.target.astype(dp.int_)
    n = y.size
    shuffle_idxs = np.random.random_integers(0, n-1, n)
    X = X[shuffle_idxs, ...]
    y = y[shuffle_idxs, ...]

    n_test = 10000
    n_valid = 10000
    n_train = n - n_test - n_valid
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_valid = X[n_train:n_train+n_valid]
    y_valid = y[n_train:n_train+n_valid]
    X_test = X[n_train+n_valid:]
    y_test = y[n_train+n_valid:]

    n_classes = np.unique(y_train).size

    # Setup neural network
    nn = dp.NeuralNetwork(
        layers=[
            dp.FullyConnected(
                n_output=800,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.0001)),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_output=800,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.0001)),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_output=n_classes,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.0001)),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    trainer = dp.StochasticGradientDescent(
        batch_size=128, learn_rate=0.1, learn_momentum=0.9, max_epochs=25
    )
    trainer.train(nn, X_train, y_train, X_valid, y_valid)

    # Visualize weights from first layer
    W = next(np.array(layer.params()[0].values) for layer in nn.layers
             if isinstance(layer, dp.FullyConnected))
    W = np.reshape(W.T, (-1, 28, 28))
    dp.misc.img_save(dp.misc.img_tile(dp.misc.img_stretch(W)),
                     os.path.join('mnist', 'mlp_weights.png'))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
