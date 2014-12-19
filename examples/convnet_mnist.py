#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp


def run():
    # Prepare data
    dataset = dp.datasets.MNIST()
    x, y = dataset.data()
    x = x[:, np.newaxis, :, :].astype(dp.float_)/255.0
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    train_input = dp.SupervisedInput(x_train, y_train, batch_size=128)
    test_input = dp.SupervisedInput(x_test, y_test)

    # Setup neural network
    nn = dp.NeuralNetwork(
        layers=[
            dp.Convolutional(
                n_filters=32,
                filter_shape=(5, 5),
                weights=dp.Parameter(dp.AutoFiller(), weight_decay=0.0001),
            ),
            dp.Activation('relu'),
            dp.Pool(
                win_shape=(3, 3),
                strides=(2, 2),
                method='max',
            ),
            dp.Convolutional(
                n_filters=64,
                filter_shape=(5, 5),
                weights=dp.Parameter(dp.AutoFiller(), weight_decay=0.0001),
            ),
            dp.Activation('relu'),
            dp.Pool(
                win_shape=(3, 3),
                strides=(2, 2),
                method='max',
            ),
            dp.Flatten(),
            dp.FullyConnected(
                n_output=128,
                weights=dp.Parameter(dp.AutoFiller()),
            ),
            dp.FullyConnected(
                n_output=dataset.n_classes,
                weights=dp.Parameter(dp.AutoFiller()),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    def valid_error():
        return nn.error(test_input)
    trainer = dp.StochasticGradientDescent(
        max_epochs=15,
        learn_rule=dp.Momentum(learn_rate=0.01, momentum=0.9),
    )
    trainer.train(nn, train_input, valid_error)

    # Visualize convolutional filters to disk
    for layer_idx, layer in enumerate(nn.layers):
        if not isinstance(layer, dp.Convolutional):
            continue
        W = np.array(layer.params()[0].values)
        dp.misc.img_save(dp.misc.conv_filter_tile(W),
                         os.path.join('mnist',
                                      'convnet_layer_%i.png' % layer_idx))

    # Evaluate on test data
    error = nn.error(test_input)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
