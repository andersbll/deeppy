#!/usr/bin/env python
# coding: utf-8

import numpy as np
import deeppy as dp


def run():
    np.random.seed(3)
    layers = [
        dp.Activation('relu'),
        dp.Activation('sigmoid'),
        dp.Activation('tanh'),
        dp.FullyConnected(
            n_out=3,
            weights=dp.AutoFiller(),
        ),
        dp.Dropout(0.2),
        dp.DropoutFullyConnected(
            n_out=10,
            weights=dp.AutoFiller(),
            dropout=0.5,
        ),
    ]

    input_shape = (1, 5)
    x = np.random.normal(size=input_shape).astype(dp.float_)
    for layer in layers:
        dp.misc.check_bprop(layer, x)

    conv_layers = [
        dp.Convolution(
            n_filters=32,
            filter_shape=(3, 3),
            border_mode='same',
            weights=dp.AutoFiller(),
        ),
        dp.Convolution(
            n_filters=32,
            filter_shape=(5, 5),
            border_mode='valid',
            weights=dp.AutoFiller(),
        ),
        dp.Pool(
            win_shape=(3, 3),
            strides=(2, 2),
            method='max',
        )
    ]
    input_shape = (5, 3, 8, 8)
    x = np.random.normal(size=input_shape).astype(dp.float_)
    for layer in conv_layers:
        dp.misc.check_bprop(layer, x)


if __name__ == '__main__':
    run()
