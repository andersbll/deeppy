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
    Pool_seg_kwargs = {
        'win_shape': (2, 2),
    }

    Y = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-labels.tif');
    X = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-volume.tif');

    n_train = 10
    n_test = 1

    imageSize = 128

    x_train = np.empty((n_train,1,1,imageSize,imageSize))
    y_train = np.zeros((n_train,imageSize*imageSize), dtype=int)

    x_test = np.empty((n_test,1,1,imageSize,imageSize))
    y_test = np.zeros((n_test,imageSize*imageSize), dtype=int)

    for im_nr in range(n_train):
        x = X[im_nr].astype(dp.float_) /255.0-0.5
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_train[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize]

        y = Y[im_nr] == 0
        y = y.astype(dp.int_)
        y_train[im_nr,:] = np.resize(y[0:imageSize,0:imageSize], (imageSize*imageSize))

    for im_nr in range(n_test):
        x = X[im_nr].astype(dp.float_) /255.0-0.5
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_test[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize]

        y = Y[im_nr] == 0
        y = y.astype(dp.int_)
        y_test[im_nr,:] = np.resize(y[0:imageSize,0:imageSize], (imageSize*imageSize))

    # Setup neural network
    nn = dp.NeuralNetwork_seg(
        layers=[
            dp.Convolutional_seg(
                n_filters=4,
                filter_shape=(3, 3),
                weights=dp.Parameter(dp.NormalFiller(sigma=0.0001),
                                     penalty=('l2', 0.004), monitor=True),
            ),
            dp.Activation_seg('relu'),
            dp.Pool_seg(win_shape=(3, 3), strides=(3,3)),
            dp.Convolutional_seg(
                n_filters=4,
                filter_shape=(5, 5),
                weights=dp.Parameter(dp.NormalFiller(sigma=0.0001),
                                     penalty=('l2', 0.004), monitor=True),
            ),
            dp.Activation_seg('relu'),
            dp.Pool_seg(**Pool_seg_kwargs),
            dp.Flatten_seg(),
            dp.FullyConnected_seg(
                n_output=50,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.03)),
            ),
            dp.Activation_seg('relu'),
            dp.FullyConnected_seg(
                n_output=2,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                     penalty=('l2', 0.03), monitor=True),
            ),
            dp.MultinomialLogReg_seg(),
        ],
    )

    # Train neural network
    def valid_error():
        return nn.error(x_test, y_test)
        
    trainer = dp.StochasticGradientDescent(
        batch_size=1,
        max_epochs=5,
        learn_rule=dp.Momentum(learn_rate=0.1, momentum=0.9),
    )
    trainer.train(nn, x_train, y_train, valid_error)

    # Evaluate on test data
    #error = nn.error(x_test, y_test)
    #rint('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()