import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)

from . import datasets
from . import misc
from .base import (
    bool_,
    int_,
    float_,
    Parameter,
)
from .input import (
    Input,
    SupervisedInput,
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
from .siamese import (
    ContrastiveLoss,
    SiameseNetwork,
    SiameseInput,
    SupervisedSiameseInput,
)
from .trainers.learning_rules import (
    Momentum,
    RMSProp,
)
from .trainers.sgd import (
    StochasticGradientDescent,
)
