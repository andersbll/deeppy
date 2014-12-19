#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp


def run():
    # Prepare data
    dataset = dp.datasets.CIFAR10()
    x, y = dataset.data()
    x = x.astype(dp.float_)
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    scaler = dp.UniformScaler(feature_wise=True)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    batch_size = 128
    train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
    test_input = dp.SupervisedInput(x_test, y_test, batch_size=batch_size)

    # Setup neural network
    pool_kwargs = {
        'win_shape': (3, 3),
        'strides': (2, 2),
        'border_mode': 'same',
        'method': 'max',
    }
    net = dp.NeuralNetwork(
        layers=[
            dp.Convolutional(
                n_filters=32,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.Parameter(dp.NormalFiller(sigma=0.0001),
                                     weight_decay=0.004, monitor=True),
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Convolutional(
                n_filters=32,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                     weight_decay=0.004, monitor=True),
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Convolutional(
                n_filters=64,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                     weight_decay=0.004, monitor=True),
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Flatten(),
            dp.FullyConnected(
                n_output=64,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     weight_decay=0.004, monitor=True),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_output=dataset.n_classes,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     weight_decay=0.004, monitor=True),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    def val_error():
        return net.error(test_input)
    n_epochs = [8, 8]
    learn_rate = 0.001
    for i, max_epochs in enumerate(n_epochs):
        lr = learn_rate/10**i
        trainer = dp.StochasticGradientDescent(
            max_epochs=max_epochs,
            learn_rule=dp.Momentum(learn_rate=lr, momentum=0.9),
        )
        trainer.train(net, train_input, val_error)

    # Visualize convolutional filters to disk
    for l, layer in enumerate(net.layers):
        if not isinstance(layer, dp.Convolutional):
            continue
        W = np.array(layer.params()[0].values)
        filepath = os.path.join('cifar10', 'conv_layer_%i.png' % l)
        dp.misc.img_save(dp.misc.conv_filter_tile(W), filepath)

    # Evaluate on test data
    error = net.error(test_input)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
