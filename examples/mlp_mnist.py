#!/usr/bin/env python

"""
Digit classification
====================

"""

import numpy as np
import matplotlib.pyplot as plt
import deeppy as dp


# Fetch MNIST data
dataset = dp.dataset.MNIST()
x_train, y_train, x_test, y_test = dataset.arrays(flat=True, dp_dtypes=True)

# Normalize pixel intensities
scaler = dp.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Prepare network inputs
batch_size = 128
train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
test_input = dp.Input(x_test)

# Setup network
weight_gain = 2.0
weight_decay = 0.0005
net = dp.NeuralNetwork(
    layers=[
        dp.Affine(
            n_out=1024,
            weights=dp.Parameter(dp.AutoFiller(weight_gain),
                                 weight_decay=weight_decay),
        ),
        dp.ReLU(),
        dp.Affine(
            n_out=1024,
            weights=dp.Parameter(dp.AutoFiller(weight_gain),
                                 weight_decay=weight_decay),
        ),
        dp.ReLU(),
        dp.Affine(
            n_out=dataset.n_classes,
            weights=dp.Parameter(dp.AutoFiller()),
        ),
    ],
    loss=dp.SoftmaxCrossEntropy(),
)

# Train network
n_epochs = [50, 15]
learn_rate = 0.05
for i, epochs in enumerate(n_epochs):
    trainer = dp.GradientDescent(
        max_epochs=epochs,
        learn_rule=dp.Momentum(learn_rate=learn_rate/10**i, momentum=0.94),
    )
    trainer.train(net, train_input)

# Evaluate on test data
error = np.mean(net.predict(test_input) != y_test)
print('Test error rate: %.4f' % error)


# Plot dataset examples
def plot_img(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()

imgs = np.reshape(x_train[:63, ...], (-1, 28, 28))
plot_img(dp.misc.img_tile(dp.misc.img_stretch(imgs)),
         'Dataset examples')

# Plot learned features in first layer
w = np.array(net.layers[0].weights.array)
w = np.reshape(w.T, (-1,) + dataset.img_shape)
w = w[np.argsort(np.std(w, axis=(1, 2)))[-64:]]
plot_img(dp.misc.img_tile(dp.misc.img_stretch(w)),
         'Examples of features learned')
