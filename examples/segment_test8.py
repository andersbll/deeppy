#!/usr/bin/env python
# coding: utf-8
import sys
import os
import time
import numpy as np
from skimage.io import imshow
from skimage import io
import deeppy as dp
import cPickle as pickle
import matplotlib.pyplot as plt
from tifffile import imsave

import logging
logger = logging.getLogger(__name__)

def run():
    # Fetch data
     # Setup neural network
    #/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-labels.tif
    Y = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-labels3.tif');
    X = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-volume3.tif');
    #Y = pickle.load(open( '../img/Y.pic', "rb" ))
    #X = pickle.load(open( '../img/X.pic', "rb" ))

    n_train = 2
    n_test = 1

    imageSize = 64

    logger.info("-------- %s -----------" % imageSize)

    x_train = np.empty((n_train,1,1,imageSize,imageSize))
    y_train = np.zeros((n_train,imageSize*imageSize), dtype=int)

    x_test = np.empty((n_test,1,1,imageSize,imageSize))
    y_test = np.zeros((n_test,imageSize*imageSize), dtype=int)

    for im_nr in range(n_train):
        x = X[im_nr].astype(dp.float_)
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_train[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize] / 255 - 0.5

        y = Y[im_nr] == 255
        y = y.astype(dp.int_)
        y_train[im_nr,:] = np.resize(y[0:imageSize,0:imageSize], (imageSize*imageSize))

    for im_nr in range(n_test):
        x = X[im_nr+n_train].astype(dp.float_)
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_test[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize] / 255 - 0.5

        y = Y[im_nr+n_train] == 255
        y = y.astype(dp.int_)
        y_test[im_nr,:] = np.resize(y[0:imageSize,0:imageSize], (imageSize*imageSize))

    # Setup neural network
    nn = dp.NeuralNetwork_seg(
        layers=[
            dp.Convolutional_seg(
                n_filters=5,
                filter_shape=(4, 4),
                weights=dp.Parameter(dp.NormalFiller(sigma=0.08251029372962436),
                                     weight_decay=0.07819657596507375, monitor=True),
            ),
            dp.Activation_seg('relu'),
            dp.Pool_seg(),
            dp.Convolutional_seg(
                n_filters=5,
                filter_shape=(5, 5),
                weights=dp.Parameter(dp.NormalFiller(sigma=0.09959728753713044),
                                      weight_decay=0.058906177839306996),
            ),
            dp.Activation_seg('relu'),
            dp.Pool_seg(),
            dp.Convolutional_seg(
                n_filters=5,
                filter_shape=(4, 4),
                weights=dp.Parameter(dp.NormalFiller(sigma=0.019752205110245127),
                                      weight_decay= 0.08771924454205336),
            ),
            dp.Activation_seg('relu'),
            dp.Pool_seg(),
            dp.Flatten_seg(win_shape=(3,3)),
            dp.FullyConnected(
                n_output=20,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.029050819595450705),
                                      weight_decay=0.08597905698096026),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_output=2,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.9374317610734719),
                                      weight_decay=0.020255044287754976),
            ),
            dp.MultinomialLogReg_seg(),
        ],
    )


    batch_size = 1
    train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
    test_input = dp.SupervisedInput(x_test, y_test, batch_size=batch_size)

    def val_error():
        return nn.error(test_input)
    n_epochs = [2, 2]
    learn_rate = 0.0035
    for i, max_epochs in enumerate(n_epochs):
        lr = learn_rate/10**i
        trainer = dp.StochasticGradientDescent(
            max_epochs=max_epochs,
            learn_rule=dp.Momentum(learn_rate=lr, momentum=0.9),
        )
        trainer.train(nn, train_input, val_error)

    #Predict competition data
    X = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/test-volume.tif');

    x_test = np.empty((len(X),1,1,imageSize,imageSize))

    for im_nr in range(len(X)):
        x = X[im_nr].astype(dp.float_)
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_test[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize] / 255 - 0.5

    test_input = dp.Input(x_test, batch_size=batch_size)
    print "predict image"
    predictimage = nn.predict(test_input)
    predictimage = np.reshape(predictimage, (len(X),imageSize,imageSize))

    #predictimage = (predictimage == 0)
    predictimage = predictimage.astype('uint8')
    predictimage *= 255

    imsave('predictTest-labels_P2.tif', predictimage)

if __name__ == '__main__':
    run()
