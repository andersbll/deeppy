import numpy as np
import cudarray as ca
import logging
from .fillers import Filler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)


bool_ = ca.bool_
int_ = ca.int_
float_ = ca.float_
