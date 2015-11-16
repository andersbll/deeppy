import cudarray as ca
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)


bool_ = ca.bool_
int_ = ca.int_
float_ = ca.float_


class CollectionMixin(object):
    collection = []


class ParamMixin(object):
    @property
    def params(self):
        """ List of Parameter objects. """
        if not isinstance(self, CollectionMixin):
            raise NotImplementedError()
        params = []
        for obj in self.collection:
            if isinstance(obj, ParamMixin):
                params.extend(obj.params)
        return params

    @params.setter
    def params(self, params):
        if not isinstance(self, CollectionMixin):
            raise NotImplementedError()
        idx = 0
        for obj in self.collection:
            if isinstance(obj, ParamMixin):
                n_params = len(obj._params)
                obj._params = params[idx:idx+n_params]
                idx += n_params


class PhaseMixin(object):
    _phase = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        if self._phase == phase:
            return
        self._phase = phase
        if isinstance(self, CollectionMixin):
            for obj in self.collection:
                if isinstance(obj, PhaseMixin):
                    obj.phase = phase


class Model(ParamMixin):
    def setup(self, **array_shapes):
        pass

    def update(self, **arrays):
        raise NotImplementedError()


class PickleMixin(object):
    _pickle_ignore = ['_tmp_']

    def _pickle_discard(self, attr_name):
        for s in self._pickle_ignore:
            if attr_name.startswith(s):
                return True
        return False

    def __getstate__(self):
        return dict((k, None) if self._pickle_discard(k) else (k, v)
                    for k, v in self.__dict__.items())
