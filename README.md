## Deep learning in Python

[![Join the chat at https://gitter.im/andersbll/deeppy](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/andersbll/deeppy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

DeepPy tries to combine state-of-the-art deep learning models with a Pythonic interface in an extensible framework.


### Features
 - Pythonic programming interface based on NumPy's ndarray.
 - Runs on CPU or Nvidia GPUs when available (thanks to [CUDArray][cudarray]).
 - Feedforward networks
   - Dropout layers.
   - Convnets layers: Convolution, pooling, local response normalization.
 - Siamese Networks
 - Training module
   - Stochastic gradient descent.
   - Interchangeable learning rules: Momentum, RMSProp.
   - Regularization: L2 weight decay.
 - Dataset module
   - MNIST, CIFAR10


### Installation
First, install [CUDArray][cudarray]. Then install DeepPy with the standard

    python setup.py install


### TODO
 - Dropout normalization of weights.
 - Documentation!
 - Support for regression problems in feed forward neural network.
 - Other network types (autoencoders, stochastic neural networks, etc.).
 - Interactive training method with visualization.


### Influences
Thanks to the following projects for showing the way.
 - [scikit-learn](http://scikit-learn.org/)
 - [Caffe](http://caffe.berkeleyvision.org/)
 - [Pylearn2](http://deeplearning.net/software/pylearn2/)


[cudarray]: http://github.com/andersbll/cudarray
