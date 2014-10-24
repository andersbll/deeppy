#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp


def run():
    # Fetch data
<<<<<<< HEAD
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')

    X = mnist.data.astype(dp.float_)/255.0-0.5
    X = np.reshape(X, (-1, 1, 28, 28))
    y = mnist.target.astype(dp.int_)
    n = y.size
    shuffle_idxs = np.random.random_integers(0, n-1, n)
    X = X[shuffle_idxs, ...]
    y = y[shuffle_idxs, ...]

    n_test = 100
    n_valid = 100
    n_train = 100
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_valid = X[n_train:n_train+n_valid]
    y_valid = y[n_train:n_train+n_valid]
    X_test = X[n_train+n_valid:]
    y_test = y[n_train+n_valid:]

    n_classes = np.unique(y_train).size
     # Setup neural network
    pool_kwargs = {
        'win_shape': (2, 2),
        'strides': (2, 2),
        'border_mode': 'same',
        'method': 'max',
    }
=======
    dataset = dp.data.MNIST()
    x, y = dataset.data()
    x = x[:, np.newaxis, :, :].astype(dp.float_)/255.0-0.5
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    # Setup neural network
>>>>>>> master
    nn = dp.NeuralNetwork(
        layers=[
            dp.Convolutional(
                n_filters=20,
                filter_shape=(5, 5),
                weights=dp.NormalFiller(sigma=0.01),
                weight_decay=0.00001,
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Flatten(),
            dp.FullyConnected(
<<<<<<< HEAD
                n_output=n_classes,
=======
                n_output=500,
                weights=dp.NormalFiller(sigma=0.01),
            ),
            dp.FullyConnected(
                n_output=dataset.n_classes,
>>>>>>> master
                weights=dp.NormalFiller(sigma=0.01),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    def valid_error():
        return nn.error(x_test, y_test)
    trainer = dp.StochasticGradientDescent(
<<<<<<< HEAD
        batch_size=20, learn_rate=0.1, learn_momentum=0.9, max_epochs=15
    )
    print("train")
    trainer.train(nn, X_train, y_train, valid_error_fun)
    print("train end")
=======
        batch_size=128,
        max_epochs=15,
        learn_rule=dp.Momentum(learn_rate=0.1, momentum=0.9),
    )
    trainer.train(nn, x_train, y_train, valid_error)

>>>>>>> master
    # Visualize convolutional filters to disk
    for layer_idx, layer in enumerate(nn.layers):
        if not isinstance(layer, dp.Convolutional):
            continue
        W = np.array(layer.params()[0].values)
        dp.misc.img_save(dp.misc.conv_filter_tile(W),
                         os.path.join('mnist',
                                      'convnet_layer_%i.png' % layer_idx))

    # Evaluate on test data
    error = nn.error(x_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
