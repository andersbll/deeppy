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
    def params(self):
        """ List of Parameter objects. """
        raise NotImplementedError()

    @params.setter
    def params(self, params):
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
    def setup(self, **array_shapes):
        pass

    def update(self, **arrays):
        raise NotImplementedError()


class PickleMixin(object):
    def __getstate__(self):
        return dict((k, None) if k.startswith('_tmp_') else (k, v)
                    for k, v in self.__dict__.items())
