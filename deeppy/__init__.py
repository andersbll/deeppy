__version__ = '0.1.dev'

from . import datasets
from . import misc
from .autoencoder import (
    Autoencoder,
    DenoisingAutoencoder,
    StackedAutoencoder,
)
from .base import (
    bool_,
    int_,
    float_,
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
    LocalContrastNormalization,
    LocalResponseNormalization,
)
from .fillers import (
    AutoFiller,
    CopyFiller,
    ConstantFiller,
    NormalFiller,
    UniformFiller,
)
from .parameter import (
    Parameter,
)

from .preprocess import (
    StandardScaler,
    UniformScaler,
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
