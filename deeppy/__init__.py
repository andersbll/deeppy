import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)

from . import data
from . import misc
from .base import (
    bool_,
    int_,
    float_,
    Parameter,
)
from .segmentation import (
    NeuralNetwork,
    FullyConnected,
    Activation,
    MultinomialLogReg,
    Dropout,
    DropoutFullyConnected,
    Convolutional,
    Flatten,
    Pool,
)
from .fillers import (
    CopyFiller,
    ConstantFiller,
    NormalFiller,
)
from .trainers.learning_rules import (
    Momentum,
    RMSProp,
)
from .trainers.sgd import (
    StochasticGradientDescent,
)
