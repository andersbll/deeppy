#!/usr/bin/env python
# coding: utf-8

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
            dp.Pool(
                win_shape=(3, 3),
                strides=(2, 2),
                border_mode='same',
                method='max',
            ),
            dp.Convolutional(
                n_filters=32,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.NormalFiller(sigma=0.01),
                weight_decay=0.004,
            ),
            dp.Activation('relu'),
            dp.Pool(
                win_shape=(3, 3),
                strides=(2, 2),
                border_mode='same',
                method='max',
            ),
            dp.Convolutional(
                n_filters=64,
                filter_shape=(5, 5),
                border_mode='same',
                weights=dp.NormalFiller(sigma=0.01),
                weight_decay=0.004,
            ),
            dp.Activation('relu'),
            dp.Pool(
                win_shape=(3, 3),
                strides=(2, 2),
                border_mode='same',
                method='max',
            ),
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
    trainer = dp.StochasticGradientDescent(
        batch_size=128, learn_rate=0.0001, learn_momentum=0.9, max_epochs=15
    )
    trainer.train(nn, X_train, y_train, X_valid, y_valid)

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
