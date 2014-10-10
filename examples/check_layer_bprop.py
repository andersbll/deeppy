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
            n_output=3,
            weights=dp.NormalFiller(sigma=0.01),
        ),
        dp.Dropout(0.2),
        dp.DropoutFullyConnected(
            n_output=10,
            weights=dp.NormalFiller(sigma=0.01),
            dropout=0.5,
        ),
    ]

    input_shape = (1, 5)
    x = np.random.normal(size=input_shape)
    for layer in layers:
        print(layer)
        dp.misc.check_bprop(layer, x)


if __name__ == '__main__':
    run()
