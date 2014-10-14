## Deep learning in Python



### Features
 - Pythonic user interface based on NumPy's ndarray.
 - Based on [cudarray] that runs on the CPU or on Nvidia GPUs when available.
 - Feed-forward nets
   - Regularization: L2 weight decay.
   - Dropout layers.
   - Convnets: Convolution, pooling, local response normalization.
   - Stochastic gradient descent with momentum.


### Installation
First, install [cudarray]. Then install deeppy with the standard

    python setup.py install


### TODO
 - Program design that allows for interchangeable learning algorithms (momentum, rprop, etc.).
 - Support for regression problems in feed forward neural network.
 - Other network types (autoencoders, stochastic neural networks, etc.).
 - Dataset module (we don't want to require scikit-learn or skdata)
 - Interactive training method with visuals


### Influences
Thanks to the following projects for showing the way:
 - scikit-learn
 - Caffe
 - Theano

[cudarray]: http://github.com/andersbll/cudarray
