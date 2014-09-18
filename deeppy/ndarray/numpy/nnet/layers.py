import numpy as np


def softmax(X):
    e = np.exp(X - np.amax(X, axis=1, keepdims=True))
    return e/np.sum(e, axis=1, keepdims=True)


def categorical_cross_entropy(y_pred, y_true):
    # Assumes one-hot encoding.
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # XXX: do we need to normalize?
    y_pred /= y_pred.sum(axis=1, keepdims=True)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return loss
