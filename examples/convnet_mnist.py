#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp


def run():
    # Prepare data
    dataset = dp.datasets.MNIST()
    x, y = dataset.data()
    x = x[:, np.newaxis, :, :].astype(dp.float_)/255.0-0.5
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    train_data = dp.SupervisedData(x_train, y_train, batch_size=128)
    test_data = dp.SupervisedData(x_test, y_test)

    # Setup neural network
    nn = dp.NeuralNetwork(
        layers=[
            dp.Convolutional(
                n_filters=20,
                filter_shape=(5, 5),
                weights=dp.NormalFiller(sigma=0.01),
                weight_decay=0.00001,
            ),
            dp.Activation('relu'),
            dp.Pool(
                win_shape=(2, 2),
                strides=(2, 2),
                method='max',
            ),
            dp.Convolutional(
                n_filters=50,
                filter_shape=(5, 5),
                weights=dp.NormalFiller(sigma=0.01),
                weight_decay=0.00001,
            ),
            dp.Activation('relu'),
            dp.Pool(
                win_shape=(2, 2),
                strides=(2, 2),
                method='max',
            ),
            dp.Flatten(),
            dp.FullyConnected(
                n_output=500,
                weights=dp.NormalFiller(sigma=0.01),
            ),
            dp.FullyConnected(
                n_output=dataset.n_classes,
                weights=dp.NormalFiller(sigma=0.01),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    def valid_error():
        return nn.error(test_data)
    trainer = dp.StochasticGradientDescent(
        max_epochs=15,
        learn_rule=dp.Momentum(learn_rate=0.1, momentum=0.9),
    )
    trainer.train(nn, train_data, valid_error)

    # Visualize convolutional filters to disk
    for layer_idx, layer in enumerate(nn.layers):
        if not isinstance(layer, dp.Convolutional):
            continue
        W = np.array(layer.params()[0].values)
        dp.misc.img_save(dp.misc.conv_filter_tile(W),
                         os.path.join('mnist',
                                      'convnet_layer_%i.png' % layer_idx))

    # Evaluate on test data
    error = nn.error(test_data)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
