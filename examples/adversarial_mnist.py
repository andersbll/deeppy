#!/usr/bin/env python

"""
Digit generation using generative adversarial nets
==================================================

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import deeppy as dp
import deeppy.expr as expr


# Fetch dataset
dataset = dp.dataset.MNIST()
x_train, y_train, x_test, y_test = dataset.arrays(dp_dtypes=True, flat=True)
n_classes = dataset.n_classes
img_shape = dataset.img_shape

# Normalize pixel intensities
scaler = dp.UniformScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Normalize pixel intensities
scaler = dp.UniformScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

# Setup network
def affine(n_out):
    return expr.nnet.Affine(n_out=n_out, weights=dp.AutoFiller(gain=1.25))

n_in = x_train.shape[1]
n_discriminator = 1024
n_hidden = 64
n_generator = 1024
generator = expr.Sequential([
    affine(n_generator),
    expr.nnet.BatchNormalization(),
    expr.nnet.ReLU(),
    affine(n_generator),
    expr.nnet.BatchNormalization(),
    expr.nnet.ReLU(),
    affine(n_in),
    expr.nnet.Sigmoid(),
])
discriminator = expr.Sequential([
    expr.nnet.Dropout(0.5),
    affine(n_discriminator),
    expr.nnet.ReLU(),
    expr.nnet.Dropout(0.5),
    affine(n_discriminator),
    expr.nnet.ReLU(),
    affine(1),
    expr.nnet.Sigmoid(),

])
model = dp.model.AdversarialNet(generator, discriminator, n_hidden=512)

# Prepare network inputs
batch_size = 64
train_input = dp.Input(x_train, batch_size=batch_size)

# Samples to be plotted during training
n_examples = 100
samples = np.random.normal(size=(n_examples, model.n_hidden)).astype(dp.float_)
plot_epochs = [0, 4, 14]
plot_imgs = [(x_train[:n_examples], 'Dataset examples')]

# Train network
n_epochs = 15
margin = 0.25
equilibrium = 0.6931
learn_rate = 0.075
learn_rule_g = dp.RMSProp(learn_rate=learn_rate)
learn_rule_d = dp.RMSProp(learn_rate=learn_rate)
model.setup(**train_input.shapes)
g_params, d_params = model.params
learn_rule_g.learn_rate /= batch_size
learn_rule_d.learn_rate /= batch_size*2
g_states = [learn_rule_g.init_state(p) for p in g_params]
d_states = [learn_rule_d.init_state(p) for p in d_params]
for epoch in range(n_epochs):
    batch_costs = []
    for batch in train_input.batches():
        real_cost, gen_cost = model.update(**batch)
        batch_costs.append((real_cost, gen_cost))
        update_g = True
        update_d = True
        if real_cost < equilibrium - margin or gen_cost < equilibrium - margin:
            update_d = False
        if real_cost > equilibrium + margin or gen_cost > equilibrium + margin:
            update_g = False
        if not (update_g or update_d):
            update_g = True
            update_d = True
        if update_g:
            for param, state in zip(g_params, g_states):
                learn_rule_g.step(param, state)
        if update_d:
            for param, state in zip(d_params, d_states):
                learn_rule_d.step(param, state)
    real_cost = np.mean([cost[0] for cost in batch_costs])
    gen_cost = np.mean([cost[1] for cost in batch_costs])
    print('epoch %d real_cost:%.4f  gen_cost:%.4f' % (epoch, real_cost,
                                                      gen_cost))
    if epoch in plot_epochs:
        samples_img = model.generate(samples)
        plot_imgs.append((samples_img, 'Samples after epoch %i' % (epoch + 1)))
        model.setup(train_input.x_shape)
        model.phase = 'train'

# Plot
fig = plt.figure()
fig_gs = matplotlib.gridspec.GridSpec(2, 2)
for i, (imgs, title) in enumerate(plot_imgs):
    imgs = np.reshape(imgs, (-1,) + img_shape)
    imgs = dp.misc.to_b01c(imgs)
    img_tile = dp.misc.img_tile(dp.misc.img_stretch(imgs))
    ax = plt.subplot(fig_gs[i // 2, i % 2])
    ax.imshow(img_tile, interpolation='nearest', cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
