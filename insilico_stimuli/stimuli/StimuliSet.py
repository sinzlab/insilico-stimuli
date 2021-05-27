# when adding a stimulus class, always add at least the methods "params" and "stimulus".
from insilico_stimuli.parameters import *

import numpy as np
import torch

class StimuliSet:
    """
    Stimulus set class for managing multiple instances of the same stimulus
    """

    def __init__(self, stimulus, canvas_size, **kwargs):
        self.stimulus = stimulus

        self.canvas_size = canvas_size
        self.arg_dict = kwargs.copy()

        self._parameter_keys = []

        self.register_params()

    def register_params(self):
        for arg_key in self.arg_dict.keys():
            arg_value = self.parse_arg(self.arg_dict[arg_key], arg_key)

            setattr(self, arg_key, arg_value)
            self._parameter_keys.append(arg_key)

    def parse_arg(self, arg_value, arg_key):
        """
        Parse argument based on the Parameter Class.
        Args:
            arg_value: value of the argument to be registered
            arg_key: name of the argument to be registered
        """
        if isinstance(arg_value, list):
            return arg_value

        elif isinstance(arg_value, FiniteSelection):
            sample = arg_value.sample()  # random sample of n values specified in sizes
            if isinstance(sample, list):
                return sample
            else:
                raise TypeError("{}.sample() must be of type list.".format(arg_key))

        elif isinstance(arg_value, FiniteParameter):
            if isinstance(arg_value.values, list):
                return arg_value.values
            else:
                raise TypeError("{}.values must be of type list.".format(arg_key))

        elif isinstance(arg_value, UniformRange):
            sample = arg_value.sample()

            if isinstance(arg_value.range, list):
                setattr(self, arg_key + "_range", arg_value.range)
            else:
                raise TypeError("{}.range must be of type list.".format(arg_key))

            if isinstance(sample, list):
                return sample
            else:
                raise TypeError("{}.sample() must be of type list.".format(arg_key))

    def params(self):
        params = []

        for key in self._parameter_keys:
            params.append((getattr(self, key), key))

        return params

    def num_params(self):
        """
        Returns:
            list: Number of different input parameters for each parameter from the 'params' method.
        """
        return [len(p[0]) for p in self.params()]

    def params_from_idx(self, idx):
        """
        labels the different parameter combinations.

        Args:
            idx (int): The index of the desired parameter combination

        Returns:
            list: parameter combinations of the desired index 'idx'
        """
        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]  # p[0] is parameter content
        return params

    def params_dict_from_idx(self, idx):
        """
        Args:
            idx (int): The index of the desired parameter combination

        Returns:
            dict: dictionary of the parameter combination specified in 'idx'
        """
        params = self.params_from_idx(idx)
        return {p[1]: params[i] for i, p in enumerate(self.params())}

    def stimulus_from_idx(self, idx):
        """
        Args:
            idx (int): The index of the desired parameter combination

        Returns: The image as numpy.ndarray with pixel values belonging to the parameter combinations of index 'idx'
        """
        stimulus = self.stimulus(self.canvas_size, **self.params_dict_from_idx(idx))()

        if isinstance(stimulus, torch.Tensor):
            stimulus = stimulus.detach().cpu().numpy()

        return stimulus

    def image_batches(self, batch_size):
        """
        Generator function dividing the resulting images from all parameter combinations into batches.

        Args:
            batch_size (int): The number of images per batch.

        Yields: The image batches as numpy.ndarray with shape (batch_size, image height, image width) or
                (num_params % batch_size, image height, image width), for the last batch.
        """
        num_stims = np.prod(self.num_params())
        for batch_start in np.arange(0, num_stims, batch_size):
            batch_end = np.minimum(batch_start + batch_size, num_stims)
            images = [self.stimulus_from_idx(i) for i in range(batch_start, batch_end)]
            yield np.array(images)

    def images(self):
        """
        Generates the images for the desired stimuli.

        Returns: The images of all possible parameter combinations as numpy.ndarray with shape
        ('total number of parameter combinations', 'image height', 'image width')
        """
        num_stims = np.prod(self.num_params())
        return np.array([self.stimulus_from_idx(i) for i in range(num_stims)])
