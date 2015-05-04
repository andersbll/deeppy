#!/usr/bin/env python
# coding: utf-8

import os
import random
import matplotlib
# Use non-GUI rendering backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import deeppy as dp


def run():
    # Prepare MNIST data
    dataset = dp.dataset.MNIST()
    x, y = dataset.data(flat=True)
    x = x.astype(dp.float_)
    y = y.astype(dp.int_)
    train_idx, test_idx = dataset.split()
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    scaler = dp.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Generate image pairs
    n_pairs = 100000
    x1 = np.empty((n_pairs, 28*28), dtype=dp.float_)
    x2 = np.empty_like(x1, dtype=dp.float_)
    y = np.empty(n_pairs, dtype=dp.int_)
    n_imgs = x_train.shape[0]
    n = 0
    while n < n_pairs:
        i = random.randint(0, n_imgs-1)
        j = random.randint(0, n_imgs-1)
        if i == j:
            continue
        x1[n, ...] = x_train[i]
        x2[n, ...] = x_train[j]
        if y_train[i] == y_train[j]:
            y[n] = 1
        else:
            y[n] = 0
        n += 1

    # Input to network
    train_input = dp.SupervisedSiameseInput(x1, x2, y, batch_size=128)
    test_input = dp.SupervisedInput(x_test, y_test)

    # Setup network
    net = dp.SiameseNetwork(
        siamese_layers=[
            dp.Dropout(),
            dp.FullyConnected(
                n_out=800,
                weights=dp.Parameter(dp.AutoFiller(), weight_decay=0.00001),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_out=800,
                weights=dp.Parameter(dp.AutoFiller(), weight_decay=0.00001),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_out=2,
                weights=dp.Parameter(dp.AutoFiller(), weight_decay=0.00001),
            ),
        ],
        loss=dp.ContrastiveLoss(margin=0.5),
    )

    # Train network
    trainer = dp.StochasticGradientDescent(
        max_epochs=10,
        learn_rule=dp.RMSProp(learn_rate=0.001),
    )
    trainer.train(net, train_input)

    # Visualize feature space
    feat = net.features(test_input)
    colors = ['tomato', 'lawngreen', 'royalblue', 'gold', 'saddlebrown',
              'violet', 'turquoise', 'mediumpurple', 'darkorange', 'darkgray']
    plt.figure()
    for i in range(10):
        plt.scatter(feat[y_test == i, 0], feat[y_test == i, 1], s=3,
                    c=colors[i], linewidths=0)
    plt.legend([str(i) for i in range(10)], scatterpoints=1, markerscale=4)
    if not os.path.exists('mnist'):
        os.mkdirs('mnist')
    plt.savefig(os.path.join('mnist', 'siamese_dists.png'), dpi=200)


if __name__ == '__main__':
    run()
