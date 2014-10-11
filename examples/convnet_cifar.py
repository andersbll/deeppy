#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import sklearn.datasets
import deeppy as dp
import skdata.cifar10


def preprocess_imgs(imgs):
    imgs = imgs.astype(float)
    imgs -= np.mean(imgs, axis=0, keepdims=True)
    # Convert images to BC01 format
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs


def run():
    # Fetch data
    dataset = skdata.cifar10.view.OfficialImageClassificationTask()
    X_train = preprocess_imgs(dataset.train.x)
    y_train = dataset.train.y
    X_test = preprocess_imgs(dataset.test.x)
    y_test = dataset.test.y
    X_valid = X_test
    y_valid = y_test
    n_classes = np.unique(y_test).size

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
                weights=dp.NormalFiller(sigma=0.0001),
                weight_decay=0.004,
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Convolutional(
                n_filters=32,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.NormalFiller(sigma=0.01),
                weight_decay=0.004,
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Convolutional(
                n_filters=64,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.NormalFiller(sigma=0.01),
                weight_decay=0.004,
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Flatten(),
            dp.FullyConnected(
                n_output=64,
                weights=dp.NormalFiller(sigma=0.1),
                weight_decay=0.03,
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_output=n_classes,
                weights=dp.NormalFiller(sigma=0.1),
                weight_decay=0.03,
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    n_epochs = [8, 8]
    learn_rate = 0.001
    for i, max_epochs in enumerate(n_epochs):
        lr = learn_rate/10**i
        trainer = dp.StochasticGradientDescent(
            batch_size=128, learn_rate=lr, learn_momentum=0.9,
            max_epochs=max_epochs
        )
        trainer.train(nn, X_train, y_train, X_valid, y_valid)

    # Visualize convolutional filters to disk
    for l, layer in enumerate(nn.layers):
        if not isinstance(layer, dp.Convolutional):
            continue
        W = np.array(layer.params()[0].values)
        dp.misc.img_save(dp.misc.conv_filter_tile(W),
                         os.path.join('cifar10', 'convnet_layer_%i.png' % l))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
