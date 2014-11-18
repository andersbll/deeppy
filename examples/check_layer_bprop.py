#!/usr/bin/env python
# coding: utf-8

import numpy as np
import deeppy as dp


def run():
    np.random.seed(3)
    conv_layers = [
        dp.Convolutional(
            n_filters=1,
            filter_shape=(4, 4),
            strides=(1,1),
            weights=dp.NormalFiller(sigma=0.01)
        )
    ]
    input_shape = (1, 1, 8, 8)
    x = np.random.normal(size=input_shape).astype(dp.float_)
    for layer in conv_layers:
        dp.misc.check_bprop(layer, x)


if __name__ == '__main__':
    run()
