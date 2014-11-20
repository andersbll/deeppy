#!/usr/bin/env python
# coding: utf-8

import numpy as np
import deeppy as dp


def run():
    np.random.seed(3)
    conv_layers = [
        dp.Convolutional_seg(
            n_filters=48,
            filter_shape=(5, 5),
            weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                 penalty=('l2', 0.04)),
        ),
        dp.Activation_seg('sigmoid'),
        dp.Flatten_seg()
    ]

    input_shape = (1, 1, 8, 8)
    x = np.random.normal(size=input_shape).astype(dp.float_)
    for layer in conv_layers:
        dp.misc.check_bprop(layer, x)

    layers = [
        dp.FullyConnected_seg(
            n_output=30,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                 penalty=('l2', 0.03), monitor=True),
        ),
        dp.Activation_seg('sigmoid'),
        dp.FullyConnected_seg(
            n_output=2,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1), monitor=True),
        ),
    ]

    input_shape = (5, 64)
    x = np.random.normal(size=input_shape).astype(dp.float_)
    for layer in layers:
        dp.misc.check_bprop(layer, x)

if __name__ == '__main__':
    run()
