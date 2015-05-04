#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import deeppy as dp


def run():
    # Prepare data
    dataset = dp.dataset.MNIST()
    x, y = dataset.data(flat=True)
    x = x.astype(dp.float_)
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    scaler = dp.UniformScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    batch_size = 128
    train_input = dp.Input(x_train, batch_size=batch_size)

    # Setup autoencoders
    sae = dp.StackedAutoencoder(
        layers=[
            dp.DenoisingAutoencoder(
                n_out=1000,
                weights=dp.Parameter(dp.AutoFiller()),
                activation='sigmoid',
                corruption=0.25,
            ),
            dp.DenoisingAutoencoder(
                n_out=1000,
                weights=dp.Parameter(dp.AutoFiller()),
                activation='sigmoid',
                corruption=0.25,
            ),
            dp.DenoisingAutoencoder(
                n_out=1000,
                weights=dp.Parameter(dp.AutoFiller()),
                activation='sigmoid',
                corruption=0.25,
            ),
        ],
    )

    # Train autoencoders layer-wise
    trainer = dp.StochasticGradientDescent(
        min_epochs=25, learn_rule=dp.Momentum(learn_rate=0.1, momentum=0.9),
    )
    for ae in sae.ae_models():
        trainer.train(ae, train_input)

    # Train stacked autoencoders
    trainer.train(sae, train_input)

    # Visualize weights from first layer
    W = next(np.array(layer.W.array) for layer in sae.layers
             if isinstance(layer, dp.Autoencoder))
    W = np.reshape(W.T, (-1, 28, 28))
    filepath = os.path.join('mnist', 'ae_weights.png')
    dp.misc.img_save(dp.misc.img_tile(dp.misc.img_stretch(W)), filepath)

    # Setup neural network using the stacked autoencoder layers
    net = dp.NeuralNetwork(
        layers=sae.feedforward_layers() + [
            dp.FullyConnected(
                n_out=dataset.n_classes,
                weights=dp.Parameter(dp.AutoFiller()),
            ),        
        ],
        loss=dp.MultinomialLogReg(),
    )

    # Fine-tune neural network
    train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
    test_input = dp.SupervisedInput(x_test, y_test)
    def val_error():
        return net.error(test_input)
    trainer = dp.StochasticGradientDescent(
        max_epochs=50, learn_rule=dp.Momentum(learn_rate=0.1, momentum=0.9),
    )
    trainer.train(net, train_input, val_error)

    # Evaluate on test data
    error = net.error(test_input)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
