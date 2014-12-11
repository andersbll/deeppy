#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from skimage.io import imshow
from skimage import io
import deeppy as dp
import cPickle as pickle
import matplotlib.pyplot as plt


def run():
    # Fetch data
     # Setup neural network
    #/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-labels.tif 
    #Y = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-labels.tif');
    #X = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-volume.tif');
    Y = pickle.load(open( './img/Y_unbalanced.pic', "rb" ))
    X = pickle.load(open( './img/X_unbalanced.pic', "rb" ))

    n_train = 200
    n_test = 50

    imageSize = 128

    x_train = np.empty((n_train,1,1,imageSize,imageSize))
    y_train = np.zeros((n_train,imageSize*imageSize), dtype=int)

    x_test = np.empty((n_test,1,1,imageSize,imageSize))
    y_test = np.zeros((n_test,imageSize*imageSize), dtype=int)

    for im_nr in range(n_train):
        x = X[im_nr].astype(dp.float_)
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_train[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize] - 0.5

        y = Y[im_nr]
        y = y.astype(dp.int_)
        y_train[im_nr,:] = np.resize(y[0:imageSize,0:imageSize], (imageSize*imageSize))

    for im_nr in range(n_test):
        x = X[im_nr+n_train].astype(dp.float_)
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_test[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize] - 0.5

        y = Y[im_nr+n_train]
        y = y.astype(dp.int_)
        y_test[im_nr,:] = np.resize(y[0:imageSize,0:imageSize], (imageSize*imageSize))

    # Setup neural network
    nn = dp.NeuralNetwork_seg(
        layers=[
            dp.Convolutional_seg(
                n_filters=10,
                filter_shape=(12, 12),
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                      monitor=False),
            ),
            dp.Activation_seg('relu'),
            dp.Pool_seg(win_shape=(4, 4), strides=(2,2)),
            dp.Convolutional_seg(
                n_filters=10,
                filter_shape=(5, 5),
                weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                      monitor=True),
            ),
            dp.Activation_seg('relu'),
            dp.Pool_seg(),
            dp.Flatten_seg(win_shape=(3,3)),
            dp.FullyConnected_seg(
                n_output=100,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                      penalty=('l2', 0.008), monitor=True),
            ),
            dp.Activation_seg('relu'),
            dp.FullyConnected_seg(
                n_output=2,
                weights=dp.Parameter(dp.NormalFiller(sigma=1),
                                      penalty=('l2', 0.008), monitor=False),
            ),
            dp.MultinomialLogReg_seg(),
        ],
    )

    #0.2592
    # Train neural network
    n_epochs = [20]
    learn_rate = 0.001
    batch_size = 1

    print "n_taning : %d , n_test: %d" % (n_train, n_test)
    print "n_epochs %s" % (n_epochs,)

    print "-------------------"
    for l in nn.layers:
        l.print_info()
        print "-------------------"



    def valid_error():
        return nn.error(x_test, y_test)

    for i, max_epochs in enumerate(n_epochs):
        lr = learn_rate /10.0**i
        print ("learn_rate: %f" % lr)
        trainer = dp.StochasticGradientDescent(
            batch_size=batch_size,
            max_epochs=max_epochs,
            learn_rule=dp.Momentum(learn_rate=lr, momentum=0.9),
        )
        trainer.train(nn, x_train, y_train, valid_error)
    # Evaluate on test data
    #error = nn.error(x_test, y_test)
    #rint('Test error rate: %.4f' % error)
    #Must beat
    ##0.1990
    print "predict image"
    predictimage = nn.predict( X=x_test[0:1,:,:,:,:], Y_shape=y_test[0:1,:].shape)
    predictimage = np.reshape(predictimage, (128,128))

    print "error"
    print nn.error(x_test[0:1,:,:,:,:], y_test[0:1,:])

    io.imsave('./testImages/testedimgcorduroy_pebbles8-33_unbalanced.png',np.reshape(x_test[0], (128,128)))
    yHey = np.reshape(y_test[0], (128,128))
    yHey *= 255
    io.imsave('./testImages/testedimgcorduroy_True_pebbles8-33_unbalanced.png', yHey)
    io.imsave('./testImages/testedimg_predictioncorduroy_pebbles8-33_unbalanced.png', predictimage)



if __name__ == '__main__':
    run()