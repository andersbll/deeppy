import numpy as np


def one_hot_encode(labels):
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def one_hot_decode(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)
