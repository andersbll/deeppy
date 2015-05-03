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


class ParamMixin(object):
    @property
    def _params(self):
        """ List of Parameter objects. """
        raise NotImplementedError()

    @_params.setter
    def _params(self, params):
        raise NotImplementedError()


class Model(ParamMixin):
    def _setup(self, input):
        pass

    def _update(self, batch):
        raise NotImplementedError()


class PickleMixin(object):
    def __getstate__(self):
        return dict((k, v) for k, v in self.__dict__.items()
                    if not k.startswith('_tmp_'))
