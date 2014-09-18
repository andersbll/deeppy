# Pythonic neural networks.

Name proposals:
 - deeppy ('deep-pi')
 - pynets
 - pynnets
 - deeplearn
 - ?

## Features (aka. TODO list)
 - Pythonic user interface based on NumPy's ndarray.
 - Runs on either CUDA or NumPy/Cython allowing you to run on the CPU or on Nvidia GPUs if available.
 - Competitive speed compared to Torch, Caffe, cuda-convnet and Theano.

### Feed-forward nets
 - Regularization: L2 weight decay.
 - Dropout layers.
 - Convnets: Convolution, pooling, local response normalization.
 

### Autoencoders
 - Variants: Sparse, Denoising autoencoder
 - Stacked autoencoders

### Stochastic neural networks
 - Restricted Boltzmann machines.
 - Deep Boltzmann machines.

### Recurrent neural networks
 - Nothing

### Training algorithms
 - Stochastic gradient descent with momentum.


## Influences
Thanks to the following projects for showing the way:
 - scikit-learn
 - Caffe
 - Theano
