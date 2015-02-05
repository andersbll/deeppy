#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp


def run():
    # Prepare data
    dataset = dp.datasets.MNIST()
    x, y = dataset.data()
    x = x.astype(dp.float_)[:, np.newaxis, :, :]
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = x[train_idx[:10]]
    y_train = y[train_idx[:10]]
    x_test = x[test_idx[:10]]
    y_test = y[test_idx[:10]]

    scaler = dp.UniformScaler(high=255.)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    batch_size = 128
    train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
    test_input = dp.SupervisedInput(x_test, y_test)

    # Setup neural network
    net = dp.NeuralNetwork(
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
    def val_error():
        return net.error(test_input)
    trainer = dp.StochasticGradientDescent(
        max_epochs=15,
        learn_rule=dp.Momentum(learn_rate=0.01, momentum=0.9),
    )
    trainer.train(net, train_input, val_error)

    # Visualize convolutional filters to disk
    for l, layer in enumerate(net.layers):
        if not isinstance(layer, dp.Convolutional):
            continue
        W = np.array(layer.params()[0].array)
        filepath = os.path.join('mnist', 'conv_layer_%i.png' % l)
        dp.misc.img_save(dp.misc.conv_filter_tile(W), filepath)

    # Evaluate on test data
    error = net.error(test_input)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
