#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from skimage.io import imshow
from skimage import io
import deeppy as dp
import cPickle as pickle
import matplotlib.pyplot as plt


def preprocess_imgs(imgs):
    imgs = imgs.astype(dp.float_)
    imgs -= np.mean(imgs, axis=0, keepdims=True)
    # Convert images to BC01 format
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs


def run():
    # Fetch data
     # Setup neural network
    #/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-labels.tif 
    Y = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-labels.tif');
    X = io.MultiImage('/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/train-volume.tif');
    #Y = pickle.load(open( './img/Y.pic', "rb" ))
    #X = pickle.load(open( './img/X.pic', "rb" ))

    n_train = 1
    n_test = 1

    imageSize = 512

    x_train = np.empty((n_train,1,1,imageSize,imageSize))
    y_train = np.zeros((n_train,imageSize*imageSize), dtype=int)

    x_test = np.empty((n_test,1,1,imageSize,imageSize))
    y_test = np.zeros((n_test,imageSize*imageSize), dtype=int)

    for im_nr in range(n_train):
        x = X[im_nr].astype(dp.float_)
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_train[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize] / 255 - 0.5

        y = Y[im_nr] - 255
        y = y.astype(dp.int_)
        y_train[im_nr,:] = np.resize(y[0:imageSize,0:imageSize], (imageSize*imageSize))

    for im_nr in range(n_test):
        x = X[im_nr+n_train].astype(dp.float_)
        #x = np.arange(imageSize*imageSize, dtype=np.float)
        #x  = x.reshape((imageSize, imageSize))
        x_test[im_nr,0,0,:,:] = x[0:imageSize,0:imageSize] / 255 - 0.5

        y = Y[im_nr+n_train] - 255
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
            dp.Flatten_seg(),
            dp.FullyConnected_seg(
                n_output=50,
                weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                      monitor=True),
            ),
            dp.Activation_seg('relu'),
            dp.FullyConnected_seg(
                n_output=2,
                weights=dp.Parameter(dp.NormalFiller(sigma=1),
                                      monitor=False),
            ),
            dp.MultinomialLogReg_seg(),
        ],
    )

    #0.2592
    # Train neural network
    n_epochs = [8, 16, 24]
    learn_rate = 0.001
    batch_size = 1

    X_profile = x_train
    y_profile = y_train
    dp.misc.profile(nn, X_profile, y_profile)


if __name__ == '__main__':
    run()
