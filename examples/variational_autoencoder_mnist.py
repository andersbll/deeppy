#!/usr/bin/env python

"""
Variational autoencoders
========================

"""
import numpy as np
import matplotlib.pyplot as plt
import deeppy as dp
import deeppy.expr as expr


# Fetch MNIST data
dataset = dp.dataset.MNIST()
x_train, y_train, x_test, y_test = dataset.arrays(flat=True, dp_dtypes=True)

# Normalize pixel intensities
scaler = dp.UniformScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Prepare network inputs
batch_size = 128
train_input = dp.Input(x_train, batch_size=batch_size)

# Setup network
def affine(n_out, gain):
    return expr.nnet.Affine(n_out=n_out, weights=dp.AutoFiller(gain))

gain = 1.0
n_in = x_train.shape[1]
n_encoder = 512
n_decoder = 512
n_hidden = 32
sigma = 0.01

encoder = expr.Sequential([
    affine(n_encoder, gain),
    expr.nnet.ReLU(),
    affine(n_encoder, gain),
    expr.nnet.ReLU(),
])
decoder = expr.Sequential([
    affine(n_decoder, gain),
    expr.nnet.ReLU(),
    affine(n_decoder, gain),
    expr.nnet.ReLU(),
    affine(n_in, gain),
    expr.nnet.Sigmoid(),
])
net = dp.model.VariationalAutoencoder(
    encoder=encoder,
    decoder=decoder,
    n_hidden=n_hidden,
)

# Train network
learn_rate = 0.25/batch_size
learn_rule = dp.RMSProp(learn_rate)
trainer = dp.GradientDescent(net, train_input, learn_rule)
trainer.train_epochs(n_epochs=25)


def plot_tile(imgs, title):
    imgs = np.reshape(imgs, (-1, 28, 28))
    tile_img = dp.misc.img_tile(dp.misc.img_stretch(imgs))
    plt.figure()
    plt.imshow(tile_img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()

n_examples = 100
embedding = net.embed(x_test[:n_examples])
plot_tile(x_test[:n_examples], 'Dataset examples')

reconstructed = net.reconstruct(embedding)
plot_tile(reconstructed, 'Example reconstructions')

samples_hidden = np.random.normal(size=(n_examples, n_hidden))
samples_reconstructed = net.reconstruct(samples_hidden.astype(dp.float_))
plot_tile(samples_reconstructed, 'Samples from latent space')

walk_from = embedding[0, :][np.newaxis, :]
walk_to = embedding[4, :][np.newaxis, :]
weights = np.linspace(0, 1, num=n_examples)[:, np.newaxis]
walk = (1 - weights)*walk_from + weights*walk_to
walk_reconstructed = net.reconstruct(walk.astype(dp.float_))
plot_tile(walk_reconstructed, 'Walk between two samples in latent space')
