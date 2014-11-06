#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp
from skimage.io import imshow
from skimage import io


def run():
    # Fetch data
     # Setup neural network
    pool_kwargs = {
        'win_shape': (2, 2),
    }

    y = io.imread('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-label.png', plugin='pil');
    train = io.imread('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-volum.png', plugin='pil');

    y = y == 0
    y = y.astype(dp.int_)
    train = train.astype(dp.float_) /255.0-0.5
    train = np.resize(train, (1,1,1,512,512))

    x_train = train[:,:,:,10:100,10:100]
    y_train = y[10:100,10:100]
    y_train = np.resize(y_train, (1, 8100))
    x_test = train[:,:,:,110:200,110:200]
    y_test = y[110:200,110:200]
    y_test = np.resize(y_test, (1, 8100))

    # Setup neural network
    nn = dp.NeuralNetwork(
        layers=[
            dp.Convolutional(
                n_filters=2,
                filter_shape=(5, 5),
                weights=dp.NormalFiller(sigma=0.01),
                weight_decay=0.00001,
            ),
            dp.Activation('relu'),
            dp.Pool(**pool_kwargs),
            dp.Flatten(),
            dp.FullyConnected(
                n_output=5,
                weights=dp.NormalFiller(sigma=0.01),
            ),
            dp.FullyConnected(
                n_output=2,
                weights=dp.NormalFiller(sigma=0.01),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    def valid_error():
        return nn.error(x_test, y_test)
        
    trainer = dp.StochasticGradientDescent(
        batch_size=1,
        max_epochs=4,
        learn_rule=dp.Momentum(learn_rate=0.1, momentum=0.9),
    )
    trainer.train(nn, x_train, y_train, valid_error)

    # Evaluate on test data
    error = nn.error(x_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()