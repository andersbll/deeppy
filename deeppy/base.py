from collections import namedtuple


class Parameter(namedtuple('Parameter', ['values', 'gradient', 'name',
                                         'penalty_fun', 'normalize_fun',
                                         'monitor'])):
    def __new__(cls, values, gradient, name='', penalty_fun=None,
                normalize_fun=None, monitor=False):
        return super(Parameter, cls).__new__(
            cls, values=values, gradient=gradient, name=name,
            penalty_fun=penalty_fun, normalize_fun=normalize_fun,
            monitor=monitor
        )
