#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp
import skdata.cifar10


def preprocess_imgs(imgs):
    imgs = imgs.astype(dp.float_)
    imgs -= np.mean(imgs, axis=0, keepdims=True)
    # Convert images to BC01 format
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs


def run():
    # Fetch data
    dataset = skdata.cifar10.view.OfficialImageClassificationTask()
    X_train = preprocess_imgs(dataset.train.x)
    y_train = dataset.train.y.astype(dp.int_)
    n_classes = np.unique(y_train).size

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
                n_output=n_classes,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.03)),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    batch_size = 128
    X_profile = X_train[:batch_size, ...]
    y_profile = y_train[:batch_size, ...]
    dp.misc.profile(nn, X_profile, y_profile)


if __name__ == '__main__':
    run()
