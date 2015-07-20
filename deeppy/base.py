import cudarray as ca
import logging

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


class PhaseMixin(object):
    _phase = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase


class Model(ParamMixin):
    def _setup(self, input):
        pass

    def _update(self, batch):
        raise NotImplementedError()


class PickleMixin(object):
    def __getstate__(self):
        return dict((k, None) if k.startswith('_tmp_') else (k, v)
                    for k, v in self.__dict__.items())
