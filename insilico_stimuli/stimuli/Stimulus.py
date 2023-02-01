from .StimuliSet import StimuliSet

from ..parameters import Parameter
import random

class Stimulus(object):
    """
    Base class for all other Stimuli
    """
    @staticmethod
    def parse_parameter(param):
        if not isinstance(param, Parameter):
            return param

        try:
            return param.sample()
        except NotImplementedError:
            return random.choice(param.values)

    def to_set(self):
        return StimuliSet(self.__class__, self.canvas_size, **self._params_dict)

    def forward(self):
        raise NotImplementedError('forward method is not implemented')

    def get_config(self):
        raise NotImplementedError('get_config method is not implemented')

