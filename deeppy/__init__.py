__version__ = '0.1.dev'

import os
import logging

debug_mode = os.getenv('DEEPPY_DEBUG', '')
debug_mode = None if debug_mode == '' else debug_mode.lower()

from . import dataset
from . import expr
from . import misc
from . import model
from .autoencoder.autoencoder import Autoencoder, DenoisingAutoencoder
from .autoencoder.stacked_autoencoder import StackedAutoencoder
from .base import bool_, int_, float_
from .feedforward.activation_layers import (
    Activation, LeakyReLU, ParametricReLU, ReLU, Sigmoid, Softmax, Softplus,
    Tanh
)
from .feedforward.neural_network import NeuralNetwork
from .feedforward.layers import Affine
from .feedforward.dropout_layers import Dropout
from .feedforward.convnet_layers import (
    Convolution, Flatten, Pool, LocalContrastNormalization,
    LocalResponseNormalization
)
from .filler import (
    AutoFiller, CopyFiller, ConstantFiller, NormalFiller, UniformFiller
)
from .input import Input, SupervisedInput
from .loss import SoftmaxCrossEntropy, BinaryCrossEntropy, MeanSquaredError
from .parameter import Parameter
from .preprocess.scalers import StandardScaler, UniformScaler
from .siamese.input import SiameseInput, SupervisedSiameseInput
from .siamese.loss import ContrastiveLoss
from .siamese.siamese_network import SiameseNetwork
from .train.annealers import ZeroAnnealer, DecayAnnealer, GammaAnnealer
from .train.learn_rules import Adam, Momentum, RMSProp
from .train.gradient_descent import GradientDescent


log = logging.getLogger(__name__)
if debug_mode is not None:
    log.info('DeepPy in debug mode: %s' % debug_mode)
