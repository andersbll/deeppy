#!/usr/bin/env python

"""
Autoencoder pretraining of neural networks
==========================================

"""

import numpy as np
import matplotlib.pyplot as plt
import deeppy as dp


# Fetch MNIST data
dataset = dp.dataset.MNIST()
x_train, y_train, x_test, y_test = dataset.arrays(flat=True, dp_dtypes=True)

# Normalize pixel intensities
scaler = dp.UniformScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Prepare autoencoder feed
batch_size = 128
train_feed = dp.Feed(x_train, batch_size=batch_size)

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
n_epochs = 25
learn_rate = 0.025/batch_size
learn_rule = dp.Momentum(learn_rate, momentum=0.9)
for ae in sae.ae_models():
    trainer = dp.GradientDescent(ae, train_feed, learn_rule)
    trainer.train_epochs(n_epochs)

# Train stacked autoencoders
trainer = dp.GradientDescent(sae, train_feed, learn_rule)
trainer.train_epochs(n_epochs)

# Setup neural network using the stacked autoencoder layers
net = dp.NeuralNetwork(
    layers=sae.feedforward_layers() + [
        dp.Affine(
            n_out=dataset.n_classes,
            weights=dp.Parameter(dp.AutoFiller()),
        ),
    ],
    loss=dp.SoftmaxCrossEntropy(),
)

# Fine-tune neural network
train_feed = dp.SupervisedFeed(x_train, y_train, batch_size=batch_size)
test_feed = dp.Feed(x_test)
trainer = dp.GradientDescent(net, train_feed, learn_rule)
trainer.train_epochs(n_epochs)

# Evaluate on test data
error = np.mean(net.predict(test_feed) != y_test)
print('Test error rate: %.4f' % error)


# Plot learned features
def plot_img(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()

w = np.array(sae.layers[0].weights.array)
w = np.reshape(w.T, (-1,) + dataset.img_shape)
sortidx = np.argsort(np.std(w, axis=(1, 2)))[-64:]
plot_img(dp.misc.img_tile(dp.misc.img_stretch(w[sortidx])),
         'Autoencoder features')

# Plot learned features in first layer
w = np.array(net.layers[0].weights.array)
w = np.reshape(w.T, (-1,) + dataset.img_shape)
plot_img(dp.misc.img_tile(dp.misc.img_stretch(w[sortidx])),
         'Fine-tuned features')
