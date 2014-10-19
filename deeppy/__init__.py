import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)

import data
import misc
from .base import (
    bool_,
    int_,
    float_,
    Parameter,
)
from .feed_forward import (
    NeuralNetwork,
    FullyConnected,
    Activation,
    MultinomialLogReg,
    Dropout,
    DropoutFullyConnected,
    Convolutional,
    Flatten,
    Pool,
    LocalResponseNormalization,
)
from .fillers import (
    CopyFiller,
    ConstantFiller,
    NormalFiller,
)
from .trainers.learning_rules import (
    Momentum,
)
from .trainers.sgd import (
    StochasticGradientDescent,
)
