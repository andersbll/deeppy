#!/usr/bin/env python

"""
Convnets for image classification (1)
=====================================

"""

import numpy as np
import deeppy as dp
import matplotlib
import matplotlib.pyplot as plt


# Fetch MNIST data
dataset = dp.dataset.MNIST()
x_train, y_train, x_test, y_test = dataset.data(dp_dtypes=True)

# Bring images to BCHW format
x_train = x_train[:, np.newaxis, :, :]
x_test = x_test[:, np.newaxis, :, :]

# Normalize pixel intensities
scaler = dp.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Prepare network inputs
batch_size = 128
train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
test_input = dp.SupervisedInput(x_test, y_test)

# Setup network
def pool_layer():
    return dp.Pool(
        win_shape=(2, 2),
        strides=(2, 2),
        border_mode='valid',
        method='max',
    )

def conv_layer(n_filters):
    return dp.Convolution(
        n_filters=n_filters,
        filter_shape=(5, 5),
        border_mode='valid',
        weights=dp.Parameter(dp.AutoFiller(gain=1.39),
                             weight_decay=0.0005),
    )

weight_gain_fc = 1.84
weight_decay_fc = 0.002
net = dp.NeuralNetwork(
    layers=[
        conv_layer(32),
        dp.Activation('relu'),
        pool_layer(),
        conv_layer(64),
        dp.Activation('relu'),
        pool_layer(),
        dp.Flatten(),
        dp.DropoutFullyConnected(
            n_out=512,
            dropout=0.5,
            weights=dp.Parameter(dp.AutoFiller(weight_gain_fc),
                                 weight_decay=weight_decay_fc),
        ),
        dp.Activation('relu'),
        dp.FullyConnected(
            n_out=dataset.n_classes,
            weights=dp.Parameter(dp.AutoFiller(weight_gain_fc)),
        ),
    ],
    loss=dp.SoftmaxCrossEntropy(),
)

# Train network
n_epochs = [50, 15, 15]
learn_rate = 0.05
momentum = 0.88
for i, epochs in enumerate(n_epochs):
    trainer = dp.StochasticGradientDescent(
        max_epochs=epochs, learn_rule=dp.Momentum(learn_rate=learn_rate/10**i,
                                                  momentum=momentum),
    )
    trainer.train(net, train_input)


# Plot misclassified images.
def plot_img(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

errors = net.predict(x_test) != y_test
n_errors = np.sum(errors)
x_errors = np.squeeze(x_test[errors])

plot_img(dp.misc.img_tile(dp.misc.img_stretch(x_errors), aspect_ratio=0.6),
         'All %i misclassified digits' % n_errors)

# Plot convolutional filters.
filters = [l.weights.array for l in net.layers
           if isinstance(l, dp.Convolution)]

fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 3])
for i, f in enumerate(filters):
    ax = plt.subplot(gs[i])
    ax.imshow(dp.misc.conv_filter_tile(f), cmap='gray',
              interpolation='nearest')
    ax.set_title('Conv layer %i' % i)
    ax.axis('off')
plt.tight_layout()
