#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp


def run():
    # Fetch data
    dataset = dp.data.MNIST()
    x, y = dataset.data(flat=True)
    x = x.astype(dp.float_)/255.0
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    # Setup neural network
    nn = dp.NeuralNetwork(
        layers=[
            dp.Dropout(0.2),
            dp.DropoutFullyConnected(
                n_output=800,
                dropout=0.5,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                     penalty=('l2', 0.00001), monitor=True),
            ),
            dp.Activation('relu'),
            dp.DropoutFullyConnected(
                n_output=800,
                dropout=0.5,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                     penalty=('l2', 0.00001), monitor=True),
            ),
            dp.Activation('relu'),
            dp.DropoutFullyConnected(
                n_output=dataset.n_classes,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                     penalty=('l2', 0.00001), monitor=True),
            ),
            dp.MultinomialLogReg(),
        ],
    )

    # Train neural network
    def valid_error():
        return nn.error(x_test, y_test)
    trainer = dp.StochasticGradientDescent(
        batch_size=128,
        max_epochs=50,
        learn_rule=dp.Momentum(learn_rate=0.1, momentum=0.9),
    )
    trainer.train(nn, x_train, y_train, valid_error)

    # Visualize weights from first layer
    W = next(np.array(layer.params()[0].values) for layer in nn.layers
             if isinstance(layer, dp.FullyConnected))
    W = np.reshape(W.T, (-1, 28, 28))
    dp.misc.img_save(dp.misc.img_tile(dp.misc.img_stretch(W)),
                     os.path.join('mnist', 'mlp_dropout_weights.png'))

    # Evaluate on test data
    error = nn.error(x_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
