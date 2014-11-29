#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp


def preprocess_imgs(imgs):
    imgs = imgs.astype(dp.float_)
    imgs -= np.mean(imgs, axis=0, keepdims=True)
    return imgs


def run():
    # Prepare data
    batch_size = 128
    dataset = dp.datasets.CIFAR10()
    x, y = dataset.data()
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = preprocess_imgs(x[train_idx])
    y_train = y[train_idx]
    x_test = preprocess_imgs(x[test_idx])
    y_test = y[test_idx]
    train_data = dp.SupervisedData(x_train, y_train, batch_size=batch_size)
    test_data = dp.SupervisedData(x_test, y_test, batch_size=batch_size)

    # Setup neural network
    pool_kwargs = {
        'win_shape': (3, 3),
        'strides': (2, 2),
        'border_mode': 'same',
        'method': 'max',
    }
    nn = dp.NeuralNetwork(
        layers=[
            dp.Convolutional(
                n_filters=32,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.Parameter(dp.NormalFiller(sigma=0.0001),
                                     penalty=('l2', 0.004), monitor=True),
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Convolutional(
                n_filters=32,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                     penalty=('l2', 0.004), monitor=True),
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Convolutional(
                n_filters=64,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                     penalty=('l2', 0.004), monitor=True),
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Flatten(),
            dp.FullyConnected(
                n_output=64,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.03)),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_output=dataset.n_classes,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.03)),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    n_epochs = [8, 8]
    learn_rate = 0.001

    def valid_error():
        return nn.error(test_data)
    for i, max_epochs in enumerate(n_epochs):
        lr = learn_rate/10**i
        trainer = dp.StochasticGradientDescent(
            max_epochs=max_epochs,
            learn_rule=dp.Momentum(learn_rate=lr, momentum=0.9),
        )
        trainer.train(nn, train_data, valid_error)

    # Visualize convolutional filters to disk
    for l, layer in enumerate(nn.layers):
        if not isinstance(layer, dp.Convolutional):
            continue
        W = np.array(layer.params()[0].values)
        dp.misc.img_save(dp.misc.conv_filter_tile(W),
                         os.path.join('cifar10', 'convnet_layer_%i.png' % l))

    # Evaluate on test data
    error = nn.error(test_data)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
