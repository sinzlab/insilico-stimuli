# when adding a stimulus class, always add at least the methods "params" and "stimulus".

import numpy as np
from numpy import pi
from numpy import random as rn
import numpy.linalg as LA

import torch
from functools import partial
from ax.service.managed_loop import optimize

import matplotlib.pyplot as plt

from insilico_stimuli.parameters import *


class StimuliSet:
    """
    Base class for all other stimuli classes.
    """
    def __init__(self):
        pass

    def params(self):
        raise NotImplementedError

    def num_params(self):
        """
        Returns:
            list: Number of different input parameters for each parameter from the 'params' method.
        """
        return [len(p[0]) for p in self.params()]

    def stimulus(self, *args, **kwargs):
        raise NotImplementedError

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
        return self.stimulus(**self.params_dict_from_idx(idx))

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


class BarsSet(StimuliSet):
    """
    A class to generate bar stimuli.
    """
    def __init__(self, canvas_size, locations, lengths, widths, contrasts, orientations, grey_levels,
                 pixel_boundaries=None):
        """Args:
            canvas_size (list of int): The canvas size [width, height].
            locations (list of list): specifies the center position of the bar. Can be either of type list or an
                object from parameters.py module. This module has 3 relevant classes: FiniteParameter, FiniteSelection,
                and UniformRange. FiniteParameter objects will be treated exactely like lists. FiniteSelection objects
                will generate n samples from the given list of values from a probability mass function. UniformRange
                objects will sample from a continuous distribution within the defined parameter ranges. If location is
                of type UniformRange, there cannot be an additional argument for the cumulative density distribution.
            lengths (list of float): determines the bar length. Can be either of type list or an object from
                parameters.py module.
            widths (list of float): determines the bar width. Can be either of type list or an object from parameters.py
                module.
            contrasts (list of float): defines the amplitude of the stimulus in %. Takes values from -1 to 1. E.g., for
                a grey_level=-0.2 and pixel_boundaries=[-1,1], a contrast of 1 (=100%) means the amplitude of the bar
                stimulus is 0.8. Negative contrasts lead to a white bar on less luminated background. Can be either of
                type list or an object from parameters.py module.
            orientations (list of float): determines the orientation of a bar. Its values are given in [rad] and can
                range from [0, pi). Can be either of type list or an object from parameters.py module.
            grey_levels (list of float): determines the mean luminance (pixel value) of the image. Can be either of type
                list or an object from parameters.py module.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
        """
        self.arg_dict = locals().copy()

        # Treat all the 'non-stimulus-oriented' arguments
        # canvas_siz
        if type(canvas_size) is list:
            self.canvas_size = canvas_size
        else:
            raise TypeError('canvas_size must be of type list.')

        # pixel_boundaries
        if pixel_boundaries is None:
            self.pixel_boundaries = [-1, 1]
        elif type(pixel_boundaries) is list:
            self.pixel_boundaries = pixel_boundaries
        else:
            raise TypeError('pixel_boundaries must be of type list.')

        # Treat all the 'stimulus-oriented' arguments
        # locations
        if isinstance(locations, list):
            self.locations = locations
        elif isinstance(locations, FiniteSelection):
            sample = locations.sample()
            if isinstance(sample, list):
                self.locations = sample
            else:
                raise TypeError("locations.sample() must be of type list.")
        elif isinstance(locations, FiniteParameter):
            if type(locations.values) is list:
                if all(isinstance(loc, list) for loc in locations.values):
                    self.locations = locations.values
                else:
                    raise TypeError('all list entries in locations.values have to be lists.')
            else:
                raise TypeError('locations.values has to be a list of lists.')
        elif isinstance(locations, UniformRange):
            sample = locations.sample()
            if isinstance(sample, list):
                self.locations = sample
            else:
                raise TypeError("locations.sample() must be of type list.")
            if isinstance(locations.range, list):
                self.locations_range = locations.range
            else:
                raise TypeError("locations.range must be of type list.")

        # read out the other inputs and store them as attributes
        self._parameter_converter()

        # For this class search methods, we want to get the parameters in an ax-friendly format
        type_check = []
        for arg in self.arg_dict:
            if arg in ["self", "canvas_size", "pixel_boundaries"]:
                pass
            else:
                type_check.append(isinstance(self.arg_dict[arg], (FiniteParameter, FiniteSelection, UniformRange)))
        if all(type_check):
            self.auto_params = self._param_dict_for_search(locations=locations,
                                                           lengths=lengths,
                                                           widths=widths,
                                                           contrasts=contrasts,
                                                           orientations=orientations,
                                                           grey_levels=grey_levels)

    def params(self):
        return [
            (self.locations, 'location'),
            (self.lengths, 'length'),
            (self.widths, 'width'),
            (self.contrasts, 'contrast'),
            (self.orientations, 'orientation'),
            (self.grey_levels, 'grey_level')
        ]

    def _parameter_converter(self):
        """ Reads out the type of all the ordinary input arguments and converts them to attributes. """
        for arg_key in self.arg_dict:
            arg_value = self.arg_dict[arg_key]
            if arg_key in ["canvas_size", "pixel_boundaries", "locations", "self"]:  # exceptions
                pass
            else:  # lengths, widths, contrasts, orientations, grey_levels
                if isinstance(arg_value, list):
                    setattr(self, arg_key, arg_value)
                elif isinstance(arg_value, FiniteSelection):
                    sample = arg_value.sample()  # random sample of n values specified in sizes
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                elif isinstance(arg_value, FiniteParameter):
                    if isinstance(arg_value.values, list):
                        setattr(self, arg_key, arg_value.values)
                    else:
                        raise TypeError("{}.values must be of type list.".format(arg_key))
                elif isinstance(arg_value, UniformRange):
                    sample = arg_value.sample()
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                    if isinstance(arg_value.range, list):
                        setattr(self, arg_key + "_range", arg_value.range)
                    else:
                        raise TypeError("{}.range must be of type list.".format(arg_key))

    def stimulus(self, location, length, width, contrast, orientation, grey_level):
        """
        Args:
            location (list of float): center position of the bar stimulus.
            length (float): length of the bar [#pixels].
            width (float): width of the bar [#pixels].
            contrast (float): contrast of the bar.
            orientation (float): orientation of the bar in radians. Takes values from [0, pi).
            grey_level (float): mean luminance of the image.

        Returns:
            numpy.ndarray: bar stimulus as array with pixel intensities.
        """
        if width > length:
            raise ValueError("width cannot be larger than length.")

        # coordinate grid
        x, y = np.meshgrid(np.arange(self.canvas_size[0]) - location[0],
                           np.arange(self.canvas_size[1]) - location[1])
        coords = np.stack([x.flatten(), y.flatten()])

        # rotation matrix
        R = np.array([[np.cos(np.pi - orientation), -np.sin(np.pi - orientation)],
                      [np.sin(np.pi - orientation), np.cos(np.pi - orientation)]])

        # scaling matrix
        A = np.array([[width / 2, 0],
                      [0, length / 2]])

        # inverse base change
        Minv = LA.inv(R.dot(A)).dot(coords)

        # infinity norm with "radius" 1 will induce desired rectangle
        Minv_norm = np.max(np.abs(Minv), axis=0)
        M_inv_norm_mat = Minv_norm.reshape(x.shape)
        bar_no_contrast = M_inv_norm_mat <= 1

        # add contrast
        amplitude = contrast * min(abs(self.pixel_boundaries[0] - grey_level),
                                   abs(self.pixel_boundaries[1] - grey_level))
        bar = amplitude * bar_no_contrast + grey_level
        return bar

    def _param_dict_for_search(self, locations, lengths, widths, contrasts, orientations, grey_levels):
        """
        Create a dictionary of all bar stimulus parameters in an ax-friendly format.

        Args:
            locations: object from parameters.py module, defining the center of stimulus.
            lengths: object from parameters.py module, defining the bar length.
            widths: object from parameters.py module, defining the bar width.
            contrasts: object from parameters.py module, defining the contrast of the image.
            orientations: object from parameters.py module, defining the orientation of the bar in radians.
            grey_levels: object from parameters.py module, defining the mean pixel intensity of the stimulus.

        Returns:
            dict of dict: dictionary of all parameters and their respective attributes, i.e. 'name, 'type', 'bounds' and
                'log_scale'.
        """
        rn.seed(None)  # truely random samples

        arg_dict = locals().copy()
        del arg_dict['self']

        param_dict = {}
        for arg_key in arg_dict:
            # "finite case" -> 'type' = choice (more than one value) or 'type' = fixed (only one value)
            if isinstance(arg_dict[arg_key], FiniteParameter) or isinstance(arg_dict[arg_key], FiniteSelection):
                # define the type configuration based on the number of list elements
                if type(getattr(self, arg_key)) is list:
                    if len(getattr(self, arg_key)) > 1:
                        name_type = "choice"
                    else:
                        name_type = "fixed"

                if arg_key == 'locations':  # exception handling #1: locations
                    # width
                    if name_type == "choice":
                        name_width = arg_key[:-1] + "_width"
                        param_dict[name_width] = {"name": name_width,
                                                  "type": name_type,
                                                  "values": [float(loc[0]) for loc in getattr(self, arg_key)]}
                        # height
                        name_height = arg_key[:-1] + "_height"
                        param_dict[name_height] = {"name": name_height,
                                                   "type": name_type,
                                                   "values": [float(loc[1]) for loc in getattr(self, arg_key)]}
                    elif name_type == "fixed":
                        name_width = arg_key[:-1] + "_width"
                        param_dict[name_width] = {"name": name_width,
                                                  "type": name_type,
                                                  "value": [float(loc[0]) for loc in getattr(self, arg_key)][0]}
                        # height
                        name_height = arg_key[:-1] + "_height"
                        param_dict[name_height] = {"name": name_height,
                                                   "type": name_type,
                                                   "value": [float(loc[1]) for loc in getattr(self, arg_key)][0]}
                else:
                    name = arg_key[:-1]
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}

            # "infinite case" -> 'type' = range
            elif isinstance(arg_dict[arg_key], UniformRange):
                if arg_key == 'locations':
                    range_name = arg_key + '_range'
                    # width
                    name_width = arg_key[:-1] + "_width"
                    param_dict[name_width] = {"name": name_width,
                                              "type": "range",
                                              "bounds": getattr(self, range_name)[0]}
                    # height
                    name_height = arg_key[:-1] + "_height"
                    param_dict[name_height] = {"name": name_height,
                                               "type": "range",
                                               "bounds": getattr(self, range_name)[1]}
                else:
                    name = arg_key[:-1]
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}
        return param_dict

    def get_image_from_params(self, auto_params):
        """
        Generates the bar stimulus corresponding to the parameters given in auto_params.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'lengths': value3, 'widths' : ...}

        Returns:
            numpy.array: Pixel intensities of the desired bar stimulus.

        """
        auto_params_copy = auto_params.copy()
        auto_params_copy['location'] = [auto_params_copy['location_width'], auto_params_copy['location_height']]
        del auto_params_copy['location_width'], auto_params_copy['location_height']
        return self.stimulus(**auto_params_copy)

    def train_evaluate(self, auto_params, model, data_key, unit_idx):
        """
        Evaluates the activation of a specific neuron in an evaluated (e.g. nnfabrik) model given the bar stimulus
        parameters.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'length': value3, 'width' : ...}
            model (Encoder): evaluated model (e.g. nnfabrik) of interest.
            data_key (str): session ID.
            unit_idx (int): index of the desired model neuron.

        Returns:
            float: The activation of the bar stimulus image of the model neuron specified in unit_idx.
        """
        auto_params_copy = auto_params.copy()
        image = self.get_image_from_params(auto_params_copy)
        image_tensor = torch.tensor(image).expand(1, 1, self.canvas_size[1], self.canvas_size[0]).float()
        activation = model(image_tensor, data_key=data_key).detach().numpy().squeeze()
        return float(activation[unit_idx])

    def find_optimal_stimulus(self, model, data_key, unit_idx, total_trials=30):
        """
        Runs Bayesian parameter optimization to find optimal bar stimulus (refer to https://ax.dev/docs/api.html).

        Args:
            model (Encoder): the underlying model of interest.
            data_key (str): session ID of model.
            unit_idx (int): unit index of desired neuron.
            total_trials (int or None): number of optimization steps (default is 30 trials).

        Returns
            - list of dict: The list entries are dictionaries which store the optimal parameter combinations for the
            corresponding unit. It has the variable name in the key and the optimal value in the values, i.e.
            [{'location_width': value1, 'location_height': value2, 'length': value3, ...}, ...]
            - list of tuple: The unit activations of the found optimal bar stimulus of the form
            [({'activation': mean_unit1}, {'activation': {'activation': sem_unit1}}), ...].
        """
        if not hasattr(self, "auto_params"):
            raise TypeError("find_optimal_stimulus search method only supports input parameters from module "
                            "parameters.py")

        parameters = list(self.auto_params.values())

        # define helper function as input to 'optimize'
        def train_evaluate_helper(auto_params):
            return partial(self.train_evaluate, model=model, data_key=data_key, unit_idx=unit_idx)(auto_params)

        # run Bayesian search
        best_params, values, _, _ = optimize(parameters=parameters.copy(),
                                             evaluation_function=train_evaluate_helper,
                                             objective_name='activation',
                                             total_trials=total_trials)
        return best_params, values

    def find_optimal_stimulus_bruteforce(self, model, data_key, batch_size=100, return_activations=False, unit_idx=None,
                                         plotflag=False):
        """
        Finds optimal parameter combination for all units based on brute force testing method.

        Args:
            model (Encoder): The evaluated model as an encoder class.
            data_key (char): data key or session ID of model.
            batch_size (int or optional): number of images per batch.
            return_activations (bool or None): return maximal activation alongside its parameter combination
            unit_idx (int or None): unit index of the desired model neuron. If not specified, return the best
                parameters for all model neurons (advised, because search is done for all units anyway).
            plotflag (bool or None): if True, plots the evolution of the maximal activation of the number of images
                tested (default: False).

        Returns
            - params (list of dict): The optimal parameter settings for each of the different units
            - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
        """
        if any([isinstance(par, UniformRange) for par in list(self.arg_dict.values())]):
            raise TypeError('This method needs inputs of type FiniteParameter or FiniteSelection.')

        n_images = np.prod(self.num_params())  # number of all parameter combinations
        n_units = model.readout[data_key].outdims  # number of units

        max_act_evo = np.zeros((n_images + 1, n_units))  # init storage of maximal activation evolution
        activations = np.zeros(n_units)  # init activation array for all tested images

        # divide set of images in batches before showing it to the model
        for batch_idx, batch in enumerate(self.image_batches(batch_size)):

            if batch.shape[0] != batch_size:
                batch_size = batch.shape[0]

            # create images and compute activation for current batch
            images_batch = batch.reshape((batch_size,) + tuple(self.canvas_size))
            images_batch = np.expand_dims(images_batch, axis=1)
            images_batch = torch.tensor(images_batch).float()
            activations_batch = model(images_batch, data_key=data_key).detach().numpy().squeeze()

            if plotflag:  # evolution of maximal activation
                for unit in range(0, n_units):
                    for idx, act in enumerate(activations_batch):
                        i = (idx + 1) + batch_idx * batch_size
                        max_act_evo[i, unit] = max(act[unit], max_act_evo[i - 1, unit])

            # max and argmax for current batch
            activations = np.vstack([activations, activations_batch])

        # delete the first row (only zeros) by which we initialized
        activations = np.delete(activations, 0, axis=0)

        # get maximal activations for each unit
        max_activations = np.amax(activations, axis=0)

        # get the image index of the maximal activations
        argmax_activations = np.argmax(activations, axis=0)

        params = [None] * n_units  # init list with parameter dictionaries
        for unit, opt_param_idx in enumerate(argmax_activations):
            params[unit] = self.params_dict_from_idx(opt_param_idx)

        # plot the evolution of the maximal activation for each additional image
        if plotflag:
            fig, ax = plt.subplots()
            for unit in range(0, n_units):
                ax.plot(np.arange(0, n_images + 1), max_act_evo[:, unit])
            plt.xlabel('Number of Images')
            plt.ylabel('Maximal Activation')

        # catch return options
        if unit_idx is not None:
            if return_activations:
                return params[unit_idx], activations[unit_idx], max_activations[unit_idx]
            else:
                return params[unit_idx], activations[unit_idx]
        else:
            if return_activations:
                return params, activations, max_activations
            else:
                return params, activations


class GaborSet(StimuliSet):
    """
    A class to generate Gabor stimuli as sinusoidal gratings modulated by a Gaussian envelope.
    """
    def __init__(self, canvas_size, locations, sizes, spatial_frequencies, contrasts, orientations, phases, grey_levels,
                 eccentricities=None, pixel_boundaries=None, relative_sf=None):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height].
            locations (list of list): specifies the center position of the Gabor. Can be either of type list or an
                object from parameters.py module. This module has 3 relevant classes: FiniteParameter, FiniteSelection,
                and UniformRange. FiniteParameter objects will be treated exactely like lists. FiniteSelection objects
                will generate n samples from the given list of values from a probability mass function. UniformRange
                objects will sample from a continuous distribution within the defined parameter ranges. If location is
                of type UniformRange, there cannot be an additional argument for the cumulative density distribution.
            sizes (list of float): determines the size of the Gabor envelope in direction of the longer axis of the
                ellipse. It is measured in pixels (pixel radius). The size corresponds to 4*SD of the Gaussian envelope
                (+/- 2 SD of envelope). Can be either of type list or an object from parameters.py module.
            spatial_frequencies (list of float): the inverse of the wavelength of the cosine factor entered in
                [cycles / pixel]. Can be either of type list or an object from parameters.py module. By setting the
                parameter 'relative_sf'=True, the spatial frequency depends on size, namely [cycles / envelope]. In
                this case, the value for the spatial frequency reflects how many periods fit into the length of 'size'
                from the center. In order to prevent the occurrence of undesired effects at the image borders, the
                wavelength value should be smaller than one fifth of the input image size. Also, the Nyquist-frequency
                should not be exceeded to avoid undesired sampling artifacts.
            contrasts (list of float): defines the amplitude of the stimulus in %. Takes values from 0 to 1. E.g., for a
                grey_level=-0.2 and pixel_boundaries=[-1,1], a contrast of 1 (=100%) means the amplitude of the Gabor
                stimulus is 0.8. Can be either of type list or an object from parameters.py module.
            orientations (list of float): determines the orientation of the normal to the parallel stripes of a Gabor
                function. Its values are given in [rad] and can range from [0, pi). Can be either of type list or an
                object from parameters.py module.
            phases (list of float): determines the phase offset in the cosine factor of the Gabor function. Its values
                are given in [rad] and can range from [0, 2*pi). Can be either of type list or an object from
                parameters.py module.
            grey_levels (list of float): determines the mean luminance (pixel value) of the image. Can be either of type
                list or an object from parameters.py module.
            eccentricities (list of float): object from parameters.py module, determining the ellipticity of the Gabor.
                Takes values from [0, 1]. Default is 0 (circular Gabor). Can be either of type list or an object from
                parameters.py module.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
            relative_sf (bool or None): Scale 'spatial_frequencies' by size (True) or use absolute units
                (False, default).
        """
        # input dictionary (used for: 'find_optimal_stimulus_bruteforce' and '_parameter_converter')
        self.arg_dict = locals().copy()
        rn.seed(None)  # truely pseudo-random samples

        # Treat all the 'non-stimulus-oriented' arguments
        # canvas_size
        if type(canvas_size) is list:
            self.canvas_size = canvas_size
        else:
            raise TypeError('canvas_size must be of type list.')

        # pixel_boundaries
        if pixel_boundaries is None:
            self.pixel_boundaries = [-1, 1]
        elif type(pixel_boundaries) is list:
            self.pixel_boundaries = pixel_boundaries
        else:
            raise TypeError('pixel_boundaries must be of type list.')

        # relative_sf
        if relative_sf is None:
            self.relative_sf = False
        elif type(relative_sf) is bool:
            self.relative_sf = relative_sf
        else:
            raise TypeError('relative_sf must be of type bool.')

        # Treat all the 'stimulus-oriented' arguments
        # locations
        if isinstance(locations, list):
            self.locations = locations
        elif isinstance(locations, FiniteSelection):
            sample = locations.sample()
            if isinstance(sample, list):
                self.locations = sample
            else:
                raise TypeError("locations.sample() must be of type list.")
        elif isinstance(locations, FiniteParameter):
            if type(locations.values) is list:
                if all(isinstance(loc, list) for loc in locations.values):
                    self.locations = locations.values
                else:
                    raise TypeError('all list entries in locations.values have to be lists.')
            else:
                raise TypeError('locations.values has to be a list of lists.')
        elif isinstance(locations, UniformRange):
            sample = locations.sample()
            if isinstance(sample, list):
                self.locations = sample
            else:
                raise TypeError("locations.sample() must be of type list.")
            if isinstance(locations.range, list):
                self.locations_range = locations.range
            else:
                raise TypeError("locations.range must be of type list.")

        # eccentricities
        if eccentricities is None:
            self.gammas = [1.0]  # default
        elif isinstance(eccentricities, list):
            self.gammas = [1 - e ** 2 for e in eccentricities]
        else:
            if isinstance(eccentricities, FiniteSelection):
                sample = eccentricities.sample()
                if isinstance(sample, list):
                    self.gammas = [1 - e ** 2 for e in sample]
                else:
                    raise TypeError("eccentricities.sample() must be of type list.")
            elif isinstance(eccentricities, FiniteParameter):
                if type(eccentricities.values) is list:
                    self.gammas = [1 - e ** 2 for e in eccentricities.values]
                else:
                    raise TypeError('eccentricities.values must be of type list.')
            elif isinstance(eccentricities, UniformRange):
                sample = eccentricities.sample()
                if isinstance(sample, list):
                    self.gammas = [1 - e ** 2 for e in sample]
                else:
                    raise TypeError("eccentricities.sample() must be of type list.")
                if isinstance(eccentricities.range, list):
                    self.gammas_range = [1 - e ** 2 for e in eccentricities.range[::-1]]
                else:
                    raise TypeError("eccentricities.range must be of type list.")

        # read out the other inputs and store them as attributes
        self._parameter_converter()

        # For this class search methods, we want to get the parameters in an ax-friendly format
        type_check = []
        for arg in self.arg_dict:
            if eccentricities is None:
                if arg in ["self", "canvas_size", "pixel_boundaries", "relative_sf", "eccentricities"]:
                    pass
                else:
                    type_check.append(isinstance(self.arg_dict[arg], (FiniteParameter, FiniteSelection, UniformRange)))
            else:
                if arg in ["self", "canvas_size", "pixel_boundaries", "relative_sf"]:
                    pass
                else:
                    type_check.append(isinstance(self.arg_dict[arg], (FiniteParameter, FiniteSelection, UniformRange)))
        if all(type_check):
            self.auto_params = self._param_dict_for_search(locations=locations,
                                                           sizes=sizes,
                                                           spatial_frequencies=spatial_frequencies,
                                                           contrasts=contrasts,
                                                           orientations=orientations,
                                                           phases=phases,
                                                           gammas=eccentricities,
                                                           grey_levels=grey_levels)

    def params(self):
        """ finite method, aranging the parameters in a list of tuples. """
        return [
            (self.locations, 'location'),
            (self.sizes, 'size'),
            (self.spatial_frequencies, 'spatial_frequency'),
            (self.contrasts, 'contrast'),
            (self.orientations, 'orientation'),
            (self.phases, 'phase'),
            (self.gammas, 'gamma'),
            (self.grey_levels, 'grey_level')
        ]

    def params_from_idx(self, idx):
        """ returns the parameter combination for a desired image index from an enumerable set of images. """
        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]
        # Caution changing the class methods: it is crucial that the index of params matches the correct parameter
        if self.relative_sf:
            params[2] /= params[1]  # params[2] is spatial_frequency and params[1] is size.
        return params

    def _parameter_converter(self):
        """ Reads out the type of all the ordinary input arguments and converts them to attributes. """
        for arg_key in self.arg_dict:
            arg_value = self.arg_dict[arg_key]
            if arg_key in ["canvas_size", "pixel_boundaries", "locations", "relative_sf", "eccentricities", "self"]:
                pass  # exceptions
            else:  # sizes, spatial_frequencies, contrasts, orientations, phases, grey_levels
                if isinstance(arg_value, list):
                    setattr(self, arg_key, arg_value)
                elif isinstance(arg_value, FiniteSelection):
                    sample = arg_value.sample()  # random sample of n values specified in sizes
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                elif isinstance(arg_value, FiniteParameter):
                    if isinstance(arg_value.values, list):
                        setattr(self, arg_key, arg_value.values)
                    else:
                        raise TypeError("{}.values must be of type list.".format(arg_key))
                elif isinstance(arg_value, UniformRange):
                    sample = arg_value.sample()
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                    if isinstance(arg_value.range, list):
                        setattr(self, arg_key + "_range", arg_value.range)
                    else:
                        raise TypeError("{}.range must be of type list.".format(arg_key))

    @staticmethod
    def density(xy, gammas, R, size):
        """
        Computes the Gaussian density value for given scalars x and y.
        Args:
            xy (numpy.array): matrix with the data points of interest.
            gammas (list of float): spatial aspect ratio parameter given as [1, gamma].
            R (numpy.array): rotation matrix.
            size (float): corresponds to the size of the Gaussian.
        Returns (numpy.array): density value for the point [x,y]
        """
        S_inv = np.diag(1 / np.array(gammas))
        return np.exp(-0.5 * np.sum(xy @ R @ S_inv @ R.T * xy, axis=-1) / (size / 4)**2)

    def stimulus(self, location, size, spatial_frequency, contrast, orientation, phase, gamma, grey_level, **kwargs):
        """
        Args:
            location (list of float): The center position of the Gabor.
            size (float): The length of the longer axis of the ellipse of the Gabor envelope.
            spatial_frequency (float): The inverse of the wavelength of the cosine factor.
            contrast (float): Defines the amplitude of the stimulus in %. Takes values from 0 to 1.
            orientation (float): The orientation of the normal to the parallel stripes.
            phase (float): The phase offset of the cosine factor.
            gamma (float): The spatial aspect ratio reflecting the ellipticity of the Gabor.
            grey_level (float): The mean luminance.
            **kwargs: Arbitrary keyword arguments.

        Returns: Image of the desired Gabor stimulus as numpy.ndarray.
        """
        x, y = np.meshgrid(np.arange(self.canvas_size[0]) - location[0],
                           np.arange(self.canvas_size[1]) - location[1])
        coords = np.stack([x.flatten(), y.flatten()])

        # rotation matrix R for envelope
        R_env = np.array([[np.cos(-orientation - np.pi/2), -np.sin(-orientation - np.pi/2)],
                      [np.sin(-orientation - np.pi/2),  np.cos(-orientation - np.pi/2)]])
        envelope = self.density(np.stack((x, y), axis=-1), gammas=[1, gamma], R=R_env, size=size)

        # rotation matrix for grating
        R = np.array([[np.cos(orientation), -np.sin(orientation)],
                      [np.sin(orientation),  np.cos(orientation)]])
        x, y = R.dot(coords).reshape((2, ) + x.shape)
        grating = np.cos(spatial_frequency * (2*pi) * x + phase)

        # add contrast
        gabor_no_contrast = envelope * grating
        amplitude = contrast * min(abs(self.pixel_boundaries[0] - grey_level),
                                   abs(self.pixel_boundaries[1] - grey_level))
        gabor = amplitude * gabor_no_contrast + grey_level

        return gabor

    def _param_dict_for_search(self, locations, sizes, spatial_frequencies, contrasts, orientations, phases, gammas,
                               grey_levels):
        """
        Create a dictionary of all Gabor parameters to an ax-friendly format.

        Args:
            locations: object from parameters.py module, defining the center of stimulus.
            sizes: object from parameters.py module, defining the size of the Gaussian envelope.
            spatial_frequencies: object from parameters.py module, defining the spatial frequency of grating.
            contrasts: object from parameters.py module, defining the contrast of the image.
            orientations: object from parameters.py module, defining the orientation of grating relative to envelope.
            phases: object from parameters.py module, defining the phase offset of the grating.
            gammas: object from parameters.py module, defining the spatial aspect ratio parameter of the envelope.
            grey_levels: object from parameters.py module, defining the mean pixel intensity of the stimulus.

        Returns:
            dict of dict: dictionary of all parameters and their respective attributes, i.e. 'name, 'type', 'bounds' and
                'log_scale'.
        """
        rn.seed(None)  # truely random samples

        arg_dict = locals().copy()
        del arg_dict['self']

        param_dict = {}
        for arg_key in arg_dict:
            # "finite case" -> 'type' = choice (more than one value) or 'type' = fixed (only one value)
            if isinstance(arg_dict[arg_key], FiniteParameter) or isinstance(arg_dict[arg_key], FiniteSelection):
                # define the type configuration based on the number of list elements
                if type(getattr(self, arg_key)) is list:
                    if len(getattr(self, arg_key)) > 1:
                        name_type = "choice"
                    else:
                        name_type = "fixed"

                if arg_key == 'locations':  # exception handling #1: locations
                    # width
                    if name_type == "choice":
                        name_width = arg_key[:-1] + "_width"
                        param_dict[name_width] = {"name": name_width,
                                                  "type": name_type,
                                                  "values": [float(loc[0]) for loc in getattr(self, arg_key)]}
                        # height
                        name_height = arg_key[:-1] + "_height"
                        param_dict[name_height] = {"name": name_height,
                                                   "type": name_type,
                                                   "values": [float(loc[1]) for loc in getattr(self, arg_key)]}
                    elif name_type == "fixed":
                        name_width = arg_key[:-1] + "_width"
                        param_dict[name_width] = {"name": name_width,
                                                  "type": name_type,
                                                  "value": [float(loc[0]) for loc in getattr(self, arg_key)][0]}
                        # height
                        name_height = arg_key[:-1] + "_height"
                        param_dict[name_height] = {"name": name_height,
                                                   "type": name_type,
                                                   "value": [float(loc[1]) for loc in getattr(self, arg_key)][0]}
                elif arg_key == 'spatial_frequencies':  # exception handling #2: spatial_frequencies
                    name = 'spatial_frequency'
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}
                else:
                    name = arg_key[:-1]
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}

            # "infinite case" -> 'type' = range
            elif isinstance(arg_dict[arg_key], UniformRange):
                if arg_key == 'locations':
                    range_name = arg_key + '_range'
                    # width
                    name_width = arg_key[:-1] + "_width"
                    param_dict[name_width] = {"name": name_width,
                                              "type": "range",
                                              "bounds": getattr(self, range_name)[0]}
                    # height
                    name_height = arg_key[:-1] + "_height"
                    param_dict[name_height] = {"name": name_height,
                                               "type": "range",
                                               "bounds": getattr(self, range_name)[1]}
                elif arg_key == 'spatial_frequencies':
                    name = 'spatial_frequency'
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}
                else:
                    name = arg_key[:-1]
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}

            if gammas is None:
                param_dict['gamma'] = {'name': 'gamma',
                                       'type': 'fixed',
                                       'value': 1.0}
        return param_dict

    def get_image_from_params(self, auto_params):
        """
        Generates the Gabor corresponding to the parameters given in auto_params.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'size': value3, 'spatial_frequency' : ...}

        Returns:
            numpy.array: Pixel intensities of the desired Gabor stimulus.

        """
        auto_params_copy = auto_params.copy()
        auto_params_copy['location'] = [auto_params_copy['location_width'], auto_params_copy['location_height']]
        del auto_params_copy['location_width'], auto_params_copy['location_height']
        return self.stimulus(**auto_params_copy)

    def train_evaluate(self, auto_params, model, data_key, unit_idx):
        """
        Evaluates the activation of a specific neuron in an evaluated (e.g. nnfabrik) model given the Gabor parameters.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'size': value3, 'spatial_frequency' : ...}
            model (Encoder): evaluated model (e.g. nnfabrik) of interest.
            data_key (str): session ID.
            unit_idx (int): index of the desired model neuron.

        Returns:
            float: The activation of the Gabor image of the model neuron specified in unit_idx.
        """
        auto_params_copy = auto_params.copy()
        image = self.get_image_from_params(auto_params_copy)
        image_tensor = torch.tensor(image).expand(1, 1, self.canvas_size[1], self.canvas_size[0]).float()
        activation = model(image_tensor, data_key=data_key).detach().numpy().squeeze()
        return float(activation[unit_idx])

    def find_optimal_stimulus(self, model, data_key, unit_idx, total_trials=30):
        """
        Runs Bayesian parameter optimization to find optimal Gabor (refer to https://ax.dev/docs/api.html).

        Args:
            model (Encoder): the underlying model of interest.
            data_key (str): session ID of model.
            unit_idx (int): unit index of desired neuron.
            total_trials (int or None): number of optimization steps (default is 30 trials).

        Returns
            - list of dict: The list entries are dictionaries which store the optimal parameter combinations for the
            corresponding unit. It has the variable name in the key and the optimal value in the values, i.e.
            [{'location_width': value1, 'location_height': value2, 'size': value3, ...}, ...]
            - list of tuple: The unit activations of the found optimal Gabor of the form [({'activation': mean_unit1},
            {'activation': {'activation': sem_unit1}}), ...].
        """
        if not hasattr(self, "auto_params"):
            raise TypeError("find_optimal_stimulus search method only supports input parameters from module "
                            "parameters.py")

        parameters = list(self.auto_params.values())

        # define helper function as input to 'optimize'
        def train_evaluate_helper(auto_params):
            return partial(self.train_evaluate, model=model, data_key=data_key, unit_idx=unit_idx)(auto_params)

       # run Bayesian search
        best_params, values, _, _ = optimize(parameters=parameters.copy(),
                                             evaluation_function=train_evaluate_helper,
                                             objective_name='activation',
                                             total_trials=total_trials)
        return best_params, values

    def find_optimal_stimulus_bruteforce(self, model, data_key, batch_size=100, return_activations=False, unit_idx=None,
                                         plotflag=False):
        """
        Finds optimal parameter combination for all units based on brute force testing method.

        Args:
            model (Encoder): The evaluated model as an encoder class.
            data_key (char): data key or session ID of model.
            batch_size (int or optional): number of images per batch.
            return_activations (bool or None): return maximal activation alongside its parameter combination
            unit_idx (int or None): unit index of the desired model neuron. If not specified, return the best
                parameters for all model neurons (advised, because search is done for all units anyway).
            plotflag (bool or None): if True, plots the evolution of the maximal activation of the number of images
                tested (default: False).

        Returns
            - params (list of dict): The optimal parameter settings for each of the different units
            - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
        """
        if any([isinstance(par, UniformRange) for par in list(self.arg_dict.values())]):
            raise TypeError('This method needs inputs of type FiniteParameter or FiniteSelection.')

        n_images = np.prod(self.num_params())  # number of all parameter combinations
        n_units = model.readout[data_key].outdims  # number of units

        max_act_evo = np.zeros((n_images + 1, n_units))  # init storage of maximal activation evolution
        activations = np.zeros(n_units)  # init activation array for all tested images

        # divide set of images in batches before showing it to the model
        for batch_idx, batch in enumerate(self.image_batches(batch_size)):

            if batch.shape[0] != batch_size:
                batch_size = batch.shape[0]

            # create images and compute activation for current batch
            images_batch = batch.reshape((batch_size,) + tuple(self.canvas_size))
            images_batch = np.expand_dims(images_batch, axis=1)
            images_batch = torch.tensor(images_batch).float()
            activations_batch = model(images_batch, data_key=data_key).detach().numpy().squeeze()

            if plotflag:  # evolution of maximal activation
                for unit in range(0, n_units):
                    for idx, act in enumerate(activations_batch):
                        i = (idx + 1) + batch_idx * batch_size
                        max_act_evo[i, unit] = max(act[unit], max_act_evo[i - 1, unit])

            # max and argmax for current batch
            activations = np.vstack([activations, activations_batch])

        # delete the first row (only zeros) by which we initialized
        activations = np.delete(activations, 0, axis=0)

        # get maximal activations for each unit
        max_activations = np.amax(activations, axis=0)

        # get the image index of the maximal activations
        argmax_activations = np.argmax(activations, axis=0)

        params = [None] * n_units  # init list with parameter dictionaries
        for unit, opt_param_idx in enumerate(argmax_activations):
            params[unit] = self.params_dict_from_idx(opt_param_idx)

        # plot the evolution of the maximal activation for each additional image
        if plotflag:
            fig, ax = plt.subplots()
            for unit in range(0, n_units):
                ax.plot(np.arange(0, n_images + 1), max_act_evo[:, unit])
            plt.xlabel('Number of Images')
            plt.ylabel('Maximal Activation')

        # catch return options
        if unit_idx is not None:
            if return_activations:
                return params[unit_idx], activations[unit_idx], max_activations[unit_idx]
            else:
                return params[unit_idx], activations[unit_idx]
        else:
            if return_activations:
                return params, activations, max_activations
            else:
                return params, activations


class PlaidsGaborSet(GaborSet):
    """
    A class to generate Plaid stimuli by adding two orthogonal Gabors.
    """
    def __init__(self, canvas_size, locations, sizes, spatial_frequencies, orientations, phases, contrasts_preferred,
                 contrasts_overlap, grey_levels, angles=None, pixel_boundaries=None, eccentricities=None,
                 relative_sf=False):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height].
            locations (list of list): Specifies the center position of the Plaids. Can be either of type list or an
                object from parameters.py module. This module has 3 relevant classes: FiniteParameter, FiniteSelection,
                and UniformRange. FiniteParameter objects will be treated exactely like lists. FiniteSelection objects
                will generate n samples from the given list of values from a probability mass function. UniformRange
                objects will sample from a continuous distribution within the defined parameter ranges. If location is
                of type UniformRange, there cannot be an additional argument for the cumulative density distribution.
            sizes (list of float): Overall size of the Plaid. Corresponds to 4*SD (+/- 2 SD) of the Gaussian envelope.
                Can be either a list or an object from parameters.py module.
            spatial_frequencies (list of float): The inverse of the wavelength of the cosine factor entered in
                [cycles / pixel]. Can be either a list or an object from parameters.py module. By setting the parameter
                'relative_sf'=True, the spatial frequency depends on size, namely [cycles / envelope]. In this case, the
                value for the spatial frequency reflects how many periods fit into the length of 'size' from the center.
                Spatial frequency is identical for preferred and overlapping Gabor. In order to prevent the occurrence
                of undesired effects at the image borders, the wavelength value should be smaller than one fifth of the
                input image size. Also, the Nyquist-frequency should not be exceeded to avoid undesired sampling
                artifacts.
            orientations (list of float): The orientation of the preferred Gabor in radians, takes values [0, pi). Can
                be either a list or an object from parameters.py module.
            phases (list of float): The phase offset of the cosine factor of the Plaid in radians, takes values
                [0, 2*pi). Same value is used for both preferred and orthogonal Gabor. Can be either a list or an object
                from parameters.py module.
            contrasts_preferred (list of float): Defines the amplitude of the preferred Gabor in %. Takes values from 0
                to 1. For grey_level=-0.2 and pixel_boundaries=[-1,1], a contrast of 1 (=100%) means the amplitude of
                the Gabor stimulus is 0.8. Can be either a list or an object from parameters.py module.
            contrasts_overlap (list of float): Defines the amplitude of the overlapping Gabor in %. Takes values from
                0 to 1. For grey_level=-0.2 and pixel_boundaries=[-1,1], a contrast of 1 (=100%) means the amplitude
                of the Gabor stimulus is 0.8. Can be either a list or an object from parameters.py module.
            grey_levels (list of float): Mean luminance/pixel value. Can be either a list or an object from
                parameters.py module.
            angles (list of float): The angle between the two overlapping Gabors in radians, default is pi/2
                (orthogonal). Can take values from [0, pi). Can be either a list or an object from parameters.py module.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
            eccentricities (list or None): The ellipticity of the Gabor, default is 0 (circular). Same value for both
                preferred and overlapping Gabor. Takes values from [0,1]. Can be either a list or an object from
                parameters.py module.
            relative_sf (bool or None): Scale 'spatial_frequencies' by size (True) or use absolute units
                (False, by default).
        """
        super().__init__(canvas_size, locations, sizes, spatial_frequencies, contrasts_preferred, orientations,
                         phases, grey_levels, eccentricities, pixel_boundaries, relative_sf)

        # contrasts_preferred
        if isinstance(contrasts_preferred, (FiniteParameter, FiniteSelection, list)):
            self.contrasts_preferred = self.contrasts
            del self.contrasts
        elif isinstance(contrasts_preferred, UniformRange):
            self.contrasts_preferred = self.contrasts
            self.contrasts_preferred_range = self.contrasts_range
            del self.contrasts_range, self.contrasts

        # contrasts_overlap
        if isinstance(contrasts_overlap, list):
            self.contrasts_overlap = contrasts_overlap
        elif isinstance(contrasts_overlap, FiniteSelection):
            sample = contrasts_overlap.sample()
            if isinstance(sample, list):
                self.contrasts_overlap = sample
            else:
                raise TypeError("contrasts_overlap.sample() must be of type list.")
        elif isinstance(contrasts_overlap, FiniteParameter):
            if type(contrasts_overlap.values) is list:
                self.contrasts_overlap = contrasts_overlap.values
            else:
                raise TypeError('contrasts_orthogonal.values must be of type list.')
        elif isinstance(contrasts_overlap, UniformRange):
            sample = contrasts_overlap.sample()
            if isinstance(sample, list):
                self.contrasts_overlap = sample
            else:
                raise TypeError("contrasts_overlap.sample() must be of type list.")
            if isinstance(contrasts_overlap.range, list):
                self.contrasts_overlap_range = contrasts_overlap.range
            else:
                raise TypeError("contrasts_overlap.range must be of type list.")

        # angles
        if angles is None:
            self.angles = [pi / 2]  # orthogonal, 90°
        elif isinstance(angles, list):
            self.angles = angles
        elif isinstance(angles, FiniteSelection):
            sample = angles.sample()
            if isinstance(sample, list):
                self.angles = sample
            else:
                raise TypeError("angles.sample() must be of type list.")
        elif isinstance(angles, FiniteParameter):
            if type(angles.values) is list:
                self.angles = angles.values
            elif type(angles.values) is int:  # linearly spaced angle values from 0° to 180°
                self.angles = np.arange(angles.values) * pi / angles.values
            else:
                raise TypeError('angles.values must be either of type list, float or int.')
        elif isinstance(angles, UniformRange):
            sample = angles.sample()
            if isinstance(sample, list):
                self.angles = sample
            else:
                raise TypeError("angles.sample() must be of type list.")
            if isinstance(angles.range, list):
                self.angles_range = angles.range
            else:
                raise TypeError("angles.range must be of type list.")

    def params(self):
        """ finite method. """
        return [
            (self.locations, 'location'),
            (self.sizes, 'size'),
            (self.spatial_frequencies, 'spatial_frequency'),
            (self.orientations, 'orientation'),
            (self.phases, 'phase'),
            (self.gammas, 'gamma'),
            (self.contrasts_preferred, 'contrast_preferred'),
            (self.contrasts_overlap, 'contrast_overlap'),
            (self.angles, 'angle'),
            (self.grey_levels, 'grey_level')
        ]

    def stimulus(self, location, size, spatial_frequency, orientation, phase, gamma, grey_level,
                 contrast_preferred, contrast_overlap, angle, **kwargs):
        """
        Args:
            location (list of float): The center position of the Plaid.
            size (float): The overall size of the Plaid envelope.
            spatial_frequency (float): The inverse of the wavelength of the cosine factor of both Gabors.
            orientation (float): The orientation of the preferred Gabor.
            phase (float): The phase offset of the cosine factor for both Gabors.
            gamma (float): The spatial aspect ratio reflecting the ellipticity of both Gabors.
            grey_level (float): Mean luminance.
            contrast_preferred (float): Defines the amplitude of the preferred Gabor in %. Takes values from 0 to 1.
            contrast_overlap (float): Defines the amplitude of the orthogonal Gabor in %. Takes values from 0 to 1.
            angle (float): angle of the overlapping Gabor to the preferred Gabor.
            **kwargs: Arbitrary keyword arguments.

        Returns: Pixel intensities of the desired Plaid stimulus as numpy.ndarray.
        """
        gabor_preferred = super().stimulus(
            location=location,
            size=size,
            spatial_frequency=spatial_frequency,
            contrast=contrast_preferred,
            orientation=orientation,
            phase=phase,
            gamma=gamma,
            grey_level=grey_level,
            **kwargs
        )

        gabor_overlap = super().stimulus(
            location=location,
            size=size,
            spatial_frequency=spatial_frequency,
            contrast=contrast_overlap,
            orientation=orientation + angle,
            phase=phase,
            gamma=gamma,
            grey_level=grey_level,
            **kwargs
        )

        plaid = gabor_preferred + gabor_overlap

        return plaid


class DiffOfGaussians(StimuliSet):
    """
    A class to generate Difference of Gaussians (DoG) by subtracting two Gaussian functions of different sizes.
    """
    def __init__(self, canvas_size, locations, sizes, sizes_scale_surround, contrasts, contrasts_scale_surround,
                 grey_levels, pixel_boundaries=None):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height].
            locations (list of list): specifies the center position of the DoG. Can be either of type list or an
                object from parameters.py module. This module has 3 relevant classes: FiniteParameter, FiniteSelection,
                and UniformRange. FiniteParameter objects will be treated exactely like lists. FiniteSelection objects
                will generate n samples from the given list of values from a probability mass function. UniformRange
                objects will sample from a continuous distribution within the defined parameter ranges. If location is
                of type UniformRange, there cannot be an additional argument for the cumulative density distribution.
            sizes (list of float): Standard deviation of the center Gaussian. Can be either a list or an object from
                parameters.py module.
            sizes_scale_surround (list of float): Scaling factor defining how much larger the standard deviation of the
                surround Gaussian is relative to the size of the center Gaussian. Must have values larger than 1. Can be
                either a list or an object from parameters.py module.
            contrasts (list of float): Contrast of the center Gaussian in %. Takes values from -1 to 1. Can be either a
                list or an object from parameters.py module. Negative contrasts yield to inverted stimuli (Gaussian in
                the center is negative, while the surround one is positive).
            contrasts_scale_surround (list of float): Contrast of the surround Gaussian relative to the center Gaussian.
                Should be between 0 and 1. Can be either a list or an object from parameters.py module.
            grey_levels (list of float): The mean luminance/pixel value. Can be either a list or an object from
                parameters.py module.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
        """
        self.arg_dict = locals().copy()

        # Treat all the 'non-stimulus-oriented' arguments
        # canvas_size
        if type(canvas_size) is list:
            self.canvas_size = canvas_size
        else:
            raise TypeError('canvas_size must be of type list.')

        # pixel_boundaries
        if pixel_boundaries is None:
            self.pixel_boundaries = [-1, 1]
        elif type(pixel_boundaries) is list:
            self.pixel_boundaries = pixel_boundaries
        else:
            raise TypeError('pixel_boundaries must be of type list.')

        # Treat the stimulus-relevant arguments
        # locations
        if isinstance(locations, list):
            self.locations = locations
        elif isinstance(locations, FiniteSelection):
            sample = locations.sample()
            if isinstance(sample, list):
                self.locations = sample
            else:
                raise TypeError("locations.sample() must be of type list.")
        elif isinstance(locations, FiniteParameter):
            if type(locations.values) is list:
                if all(isinstance(loc, list) for loc in locations.values):
                    self.locations = locations.values
                else:
                    raise TypeError('all list entries in locations.values have to be lists.')
            else:
                raise TypeError('locations.values has to be a list of lists.')
        elif isinstance(locations, UniformRange):
            sample = locations.sample()
            if isinstance(sample, list):
                self.locations = sample
            else:
                raise TypeError("locations.sample() must be of type list.")
            if isinstance(locations.range, list):
                self.locations_range = locations.range
            else:
                raise TypeError("locations.range must be of type list.")

        # read out the other input arguments and store them in class attributes
        self._parameter_converter()

        # For this class search methods, we want to get the parameters in an ax-friendly format
        type_check = []
        for arg in self.arg_dict:
            if arg in ["self", "canvas_size", "pixel_boundaries"]:
                pass
            else:
                type_check.append(isinstance(self.arg_dict[arg], (FiniteParameter, FiniteSelection, UniformRange)))
        if all(type_check):
            self.auto_params = self._param_dict_for_search(locations=locations,
                                                           sizes=sizes,
                                                           sizes_scale_surround=sizes_scale_surround,
                                                           contrasts=contrasts,
                                                           contrasts_scale_surround=contrasts_scale_surround,
                                                           grey_levels=grey_levels)

    def params(self):
        return [
            (self.locations, 'location'),
            (self.sizes, 'size'),
            (self.sizes_scale_surround, 'size_scale_surround'),
            (self.contrasts, 'contrast'),
            (self.contrasts_scale_surround, 'contrast_scale_surround'),
            (self.grey_levels, 'grey_level')
        ]

    def _parameter_converter(self):
        """ Reads out the type of all the ordinary input arguments and converts them to attributes. """
        for arg_key in self.arg_dict:
            arg_value = self.arg_dict[arg_key]
            if arg_key in ["canvas_size", "pixel_boundaries", "locations", "self"]:
                pass  # exceptions
            else:  # sizes, sizes_scale_surround, contrasts, contrast_scale_surround, grey_levels
                if isinstance(arg_value, list):
                    setattr(self, arg_key, arg_value)
                elif isinstance(arg_value, FiniteSelection):
                    sample = arg_value.sample()  # random sample of n values specified in sizes
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                elif isinstance(arg_value, FiniteParameter):
                    if isinstance(arg_value.values, list):
                        setattr(self, arg_key, arg_value.values)
                    else:
                        raise TypeError("{}.values must be of type list.".format(arg_key))
                elif isinstance(arg_value, UniformRange):
                    sample = arg_value.sample()
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                    if isinstance(arg_value.range, list):
                        setattr(self, arg_key + "_range", arg_value.range)
                    else:
                        raise TypeError("{}.range must be of type list.".format(arg_key))

    @staticmethod
    def gaussian_density(coords, mean, scale):
        """
        Args:
            coords: The evaluation points with shape (#points, 2) as numpy.ndarray.
            mean (int): The mean/location of the Gaussian.
            scale (int): The standard deviation of the Gaussian.

        Returns: Unnormalized Gaussian density values evaluated at the positions in 'coords' as numpy.ndarray.
        """
        mean = np.reshape(mean, [1, -1])
        r2 = np.sum(np.square(coords - mean), axis=1)
        return np.exp(-r2 / (2 * scale**2))

    def stimulus(self, location, size, size_scale_surround, contrast, contrast_scale_surround, grey_level, **kwargs):
        """
        Args:
            location (list of float): The center position of the DoG.
            size (float): Standard deviation of the center Gaussian.
            size_scale_surround (float): Scaling factor defining how much larger the standard deviation of the surround
                Gaussian is relative to the size of the center Gaussian. Must have values larger than 1.
            contrast (float): Contrast of the center Gaussian in %. Takes values from -1 to 1.
            contrast_scale_surround (float): Contrast of the surround Gaussian relative to the center Gaussian.
            grey_level (float): mean luminance.
            **kwargs: Arbitrary keyword arguments.

        Returns: Pixel intensities for desired Difference of Gaussians stimulus as numpy.ndarray.
        """
        if size_scale_surround <= 1:
            raise ValueError("size_surround must be larger than 1.")

        x, y = np.meshgrid(np.arange(self.canvas_size[0]),
                           np.arange(self.canvas_size[1]))
        coords = np.stack([x.flatten(), y.flatten()], axis=-1).reshape(-1, 2)

        center = self.gaussian_density(coords, mean=location, scale=size).reshape(self.canvas_size[::-1])
        surround = self.gaussian_density(coords, mean=location, scale=(size_scale_surround * size)
                                         ).reshape(self.canvas_size[::-1])
        center_surround = center - contrast_scale_surround * surround

        # add contrast
        min_val, max_val = center_surround.min(), center_surround.max()
        amplitude_current = max(np.abs(min_val), np.abs(max_val))
        amplitude_required = contrast * min(np.abs(self.pixel_boundaries[0] - grey_level),
                                            np.abs(self.pixel_boundaries[1] - grey_level))
        contrast_scaling = amplitude_required / amplitude_current

        diff_of_gaussians = contrast_scaling * center_surround + grey_level

        return diff_of_gaussians

    def _param_dict_for_search(self, locations, sizes, sizes_scale_surround, contrasts, contrasts_scale_surround,
                               grey_levels):
        """
        Create a dictionary of all DoG parameters in an ax-friendly format.

        Args:
            locations: object from parameters.py module, defining the center of stimulus.
            sizes: object from parameters.py module, defining the size of the Gaussian envelope.
            sizes_scale_surround: object from parameters.py module, defining the standard deviation of the surround
                Gaussian relative to the size of the center Gaussian.
            contrasts: object from parameters.py module, defining the contrast of the image.
            contrasts_scale_surround: object from parameters.py module, defining the contrast of the surround Gaussian
                relative to the center Gaussian.
            grey_levels: object from parameters.py module, defining the mean pixel intensity of the stimulus.

        Returns:
            dict of dict: dictionary of all parameters and their respective attributes, i.e. 'name, 'type', 'bounds' and
                'log_scale'.
        """
        rn.seed(None)  # truely random samples

        arg_dict = locals().copy()
        del arg_dict['self']

        param_dict = {}
        for arg_key in arg_dict:
            # "finite case" -> 'type' = choice (more than one value) or 'type' = fixed (only one value)
            if isinstance(arg_dict[arg_key], FiniteParameter) or isinstance(arg_dict[arg_key], FiniteSelection):
                # define the type configuration based on the number of list elements
                if type(getattr(self, arg_key)) is list:
                    if len(getattr(self, arg_key)) > 1:
                        name_type = "choice"
                    else:
                        name_type = "fixed"

                if arg_key == 'locations':  # exception handling #1: locations
                    # width
                    if name_type == "choice":
                        name_width = arg_key[:-1] + "_width"
                        param_dict[name_width] = {"name": name_width,
                                                  "type": name_type,
                                                  "values": [float(loc[0]) for loc in getattr(self, arg_key)]}
                        # height
                        name_height = arg_key[:-1] + "_height"
                        param_dict[name_height] = {"name": name_height,
                                                   "type": name_type,
                                                   "values": [float(loc[1]) for loc in getattr(self, arg_key)]}
                    elif name_type == "fixed":
                        name_width = arg_key[:-1] + "_width"
                        param_dict[name_width] = {"name": name_width,
                                                  "type": name_type,
                                                  "value": [float(loc[0]) for loc in getattr(self, arg_key)][0]}
                        # height
                        name_height = arg_key[:-1] + "_height"
                        param_dict[name_height] = {"name": name_height,
                                                   "type": name_type,
                                                   "value": [float(loc[1]) for loc in getattr(self, arg_key)][0]}
                elif arg_key == 'sizes_scale_surround':  # exception handling #2: sizes_scale_surround
                    name = 'size_scale_surround'
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}
                elif arg_key == 'contrasts_scale_surround':  # exception handling #3: contrasts_scale_surround
                    name = 'contrast_scale_surround'
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}
                else:
                    name = arg_key[:-1]
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}

            # "infinite case" -> 'type' = range
            elif isinstance(arg_dict[arg_key], UniformRange):
                if arg_key == 'locations':
                    range_name = arg_key + '_range'
                    # width
                    name_width = arg_key[:-1] + "_width"
                    param_dict[name_width] = {"name": name_width,
                                              "type": "range",
                                              "bounds": getattr(self, range_name)[0]}
                    # height
                    name_height = arg_key[:-1] + "_height"
                    param_dict[name_height] = {"name": name_height,
                                               "type": "range",
                                               "bounds": getattr(self, range_name)[1]}
                elif arg_key == 'sizes_scale_surround':
                    name = 'size_scale_surround'
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}
                elif arg_key == 'contrasts_scale_surround':
                    name = 'contrast_scale_surround'
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}
                else:
                    name = arg_key[:-1]
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}
        return param_dict

    def get_image_from_params(self, auto_params):
        """
        Generates the DoG corresponding to the parameters given in auto_params.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'size': value3, 'sizes_scale_surround' : ...}

        Returns:
            numpy.array: Pixel intensities of the desired DoG stimulus.

        """
        auto_params_copy = auto_params.copy()
        auto_params_copy['location'] = [auto_params_copy['location_width'], auto_params_copy['location_height']]
        del auto_params_copy['location_width'], auto_params_copy['location_height']
        return self.stimulus(**auto_params_copy)

    def train_evaluate(self, auto_params, model, data_key, unit_idx):
        """
        Evaluates the activation of a specific neuron in an evaluated (e.g. nnfabrik) model given the DoG parameters.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'size': value3, 'sizes_scale_surround' : ...}
            model (Encoder): evaluated model (e.g. nnfabrik) of interest.
            data_key (str): session ID.
            unit_idx (int): index of the desired model neuron.

        Returns:
            float: The activation of the DoG image of the model neuron specified in unit_idx.
        """
        auto_params_copy = auto_params.copy()
        image = self.get_image_from_params(auto_params_copy)
        image_tensor = torch.tensor(image).expand(1, 1, self.canvas_size[1], self.canvas_size[0]).float()
        activation = model(image_tensor, data_key=data_key).detach().numpy().squeeze()
        return float(activation[unit_idx])

    def find_optimal_stimulus(self, model, data_key, unit_idx, total_trials=30):
        """
        Runs Bayesian parameter optimization to find optimal DoG stimulus (refer to https://ax.dev/docs/api.html).

        Args:
            model (Encoder): the underlying model of interest.
            data_key (str): session ID of model.
            unit_idx (int): unit index of desired neuron.
            total_trials (int or None): number of optimization steps (default is 30 trials).

        Returns
            - list of dict: The list entries are dictionaries which store the optimal parameter combinations for the
            corresponding unit. It has the variable name in the key and the optimal value in the values, i.e.
            [{'location_width': value1, 'location_height': value2, 'size': value3, ...}, ...]
            - list of tuple: The unit activations of the found optimal DoG of the form [({'activation': mean_unit1},
            {'activation': {'activation': sem_unit1}}), ...].
        """
        if not hasattr(self, "auto_params"):
            raise TypeError("find_optimal_stimulus search method only supports input parameters from module "
                            "parameters.py")

        parameters = list(self.auto_params.values())

        # define helper function as input to 'optimize'
        def train_evaluate_helper(auto_params):
            return partial(self.train_evaluate, model=model, data_key=data_key, unit_idx=unit_idx)(auto_params)

       # run Bayesian search
        best_params, values, _, _ = optimize(parameters=parameters.copy(),
                                             evaluation_function=train_evaluate_helper,
                                             objective_name='activation',
                                             total_trials=total_trials)
        return best_params, values

    def find_optimal_stimulus_bruteforce(self, model, data_key, batch_size=100, return_activations=False, unit_idx=None,
                                         plotflag=False):
        """
        Finds optimal parameter combination for all units based on brute force testing method.

        Args:
            model (Encoder): The evaluated model as an encoder class.
            data_key (char): data key or session ID of model.
            batch_size (int or optional): number of images per batch.
            return_activations (bool or None): return maximal activation alongside its parameter combination
            unit_idx (int or None): unit index of the desired model neuron. If not specified, return the best
                parameters for all model neurons (advised, because search is done for all units anyway).
            plotflag (bool or None): if True, plots the evolution of the maximal activation of the number of images
                tested (default: False).

        Returns
            - params (list of dict): The optimal parameter settings for each of the different units
            - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
        """
        if any([isinstance(par, UniformRange) for par in list(self.arg_dict.values())]):
            raise TypeError('This method needs inputs of type FiniteParameter or FiniteSelection.')

        n_images = np.prod(self.num_params())  # number of all parameter combinations
        n_units = model.readout[data_key].outdims  # number of units

        max_act_evo = np.zeros((n_images + 1, n_units))  # init storage of maximal activation evolution
        activations = np.zeros(n_units)  # init activation array for all tested images

        # divide set of images in batches before showing it to the model
        for batch_idx, batch in enumerate(self.image_batches(batch_size)):

            if batch.shape[0] != batch_size:
                batch_size = batch.shape[0]

            # create images and compute activation for current batch
            images_batch = batch.reshape((batch_size,) + tuple(self.canvas_size))
            images_batch = np.expand_dims(images_batch, axis=1)
            images_batch = torch.tensor(images_batch).float()
            activations_batch = model(images_batch, data_key=data_key).detach().numpy().squeeze()

            if plotflag:  # evolution of maximal activation
                for unit in range(0, n_units):
                    for idx, act in enumerate(activations_batch):
                        i = (idx + 1) + batch_idx * batch_size
                        max_act_evo[i, unit] = max(act[unit], max_act_evo[i - 1, unit])

            # max and argmax for current batch
            activations = np.vstack([activations, activations_batch])

        # delete the first row (only zeros) by which we initialized
        activations = np.delete(activations, 0, axis=0)

        # get maximal activations for each unit
        max_activations = np.amax(activations, axis=0)

        # get the image index of the maximal activations
        argmax_activations = np.argmax(activations, axis=0)

        params = [None] * n_units  # init list with parameter dictionaries
        for unit, opt_param_idx in enumerate(argmax_activations):
            params[unit] = self.params_dict_from_idx(opt_param_idx)

        # plot the evolution of the maximal activation for each additional image
        if plotflag:
            fig, ax = plt.subplots()
            for unit in range(0, n_units):
                ax.plot(np.arange(0, n_images + 1), max_act_evo[:, unit])
            plt.xlabel('Number of Images')
            plt.ylabel('Maximal Activation')

        # catch return options
        if unit_idx is not None:
            if return_activations:
                return params[unit_idx], activations[unit_idx], max_activations[unit_idx]
            else:
                return params[unit_idx], activations[unit_idx]
        else:
            if return_activations:
                return params, activations, max_activations
            else:
                return params, activations


class CenterSurround(StimuliSet):
    """
    A class to generate 'Center-Surround' stimuli with optional center and/or surround gratings.
    """
    def __init__(self, canvas_size, locations, sizes_total, sizes_center, sizes_surround, contrasts_center,
                 contrasts_surround, orientations_center, orientations_surround, spatial_frequencies_center,
                 phases_center, grey_levels, spatial_frequencies_surround=None, phases_surround=None,
                 pixel_boundaries=None):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height].
            locations (list of list): specifies the center position of the stimulus. Can be either of type list or an
                object from parameters.py module. This module has 3 relevant classes: FiniteParameter, FiniteSelection,
                and UniformRange. FiniteParameter objects will be treated exactly like lists. FiniteSelection objects
                will generate n samples from the given list of values from a probability mass function. UniformRange
                objects will sample from a continuous distribution within the defined parameter ranges. If location is
                of type UniformRange, there cannot be an additional argument for the cumulative density distribution.
            sizes_total (list of float): The overall size of the Center-Surround stimulus. Can be either a list or an
                object from parameters.py module.
            sizes_center (list of float): The size of the center as a fraction of the overall size. Takes values from 0
                to 1. 'size_center' is a scaling factor for 'size_total' so that 'size_center' * 'size_total' = radius
                of inner circle. Can be either a list or an object from parameters.py module.
            sizes_surround (list of float): The size of the surround as a fraction of the overall size. Takes values
                from 0 to 1. Can be either a list or an object from parameters.py module.
            contrasts_center (list of float): The contrast of the center grating in %. Takes values from 0 to 1. Can be
                either a list or an object from parameters.py module.
            contrasts_surround (list of float): The contrast of the surround grating in %. Takes values from 0 to 1. Can
                be either a list or an object from parameters.py module.
            orientations_center (list of float): The orientation of the center gratings. Takes values from 0 to pi. Can
                be either a list or an object from parameters.py module.
            orientations_surround (list of float): The orientation of the surround gratings. Takes values from 0 to pi.
                Can be either a list or an object from parameters.py module.
            spatial_frequencies_center (list of float): The inverse of the wavelength of the center gratings in
                absolute units [cycles / pixel]. Can be either a list or an object from parameters.py module.
            phases_center (list of float): The phase offset of the center sinusoidal gratings. Takes values from
                [0, 2*pi) Can be either a list or an object from parameters.py module.
            grey_levels (list of float): The mean luminance/pixel value. Can be either a list or an object from
                parameters.py module.
            spatial_frequencies_surround (list of float or None): The inverse of the wavelength of the center gratings
                in absolute units [cycles / pixel]. If not specified, use same value as in 'spatial_frequencies_center'
                for both generation methods and search methods. In particular, if spatial_frequencies_surround is not
                specified, it will be run over the same value/s as spatial_frequencies_center. In the UniformRange case,
                for the search method, this means that the two parameters are optimized independently from each other
                over the same range. When sizes_center is 1.0, then there is no surround. For this case, the search
                method should not optimize over spatial_frequencies_surround. For this case, the parameter is set to 0.
            phases_surround (list of float or None): The phase offset of the surround sinusoidal gratings. Takes values
                from [0, 2*pi). If not specified, use same value as in 'phases_center' for both generation methods and
                search methods. In particular, if phases_surround is not specified, it will be run over the same value/s
                as phases_center. In the UniformRange case, for the search method, this means that the two parameters
                are optimized independently from each other over the same range. When sizes_center is 1.0, then there is
                no surround. For this case, the search method should not optimize over phases_surround. For this case,
                the parameter is set to 0.
            pixel_boundaries (list of float or None): Range of values the monitor can display. Handed to the class in
                the format [lower pixel value, upper pixel value], default is [-1,1].
        """
        self.arg_dict = locals().copy()

        # Treat all the 'non-stimulus-oriented' arguments
        # canvas_size
        if type(canvas_size) is list:
            self.canvas_size = canvas_size
        else:
            raise TypeError('canvas_size must be of type list.')

        # pixel_boundaries
        if pixel_boundaries is None:
            self.pixel_boundaries = [-1, 1]
        elif type(pixel_boundaries) is list:
            self.pixel_boundaries = pixel_boundaries
        else:
            raise TypeError('pixel_boundaries must be of type list.')

        # Treat all stimulus-relevant arguments
        # locations
        if isinstance(locations, list):
            self.locations = locations
        elif isinstance(locations, FiniteSelection):
            sample = locations.sample()
            if isinstance(sample, list):
                self.locations = sample
            else:
                raise TypeError("locations.sample() must be of type list.")
        elif isinstance(locations, FiniteParameter):
            if type(locations.values) is list:
                if all(isinstance(loc, list) for loc in locations.values):
                    self.locations = locations.values
                else:
                    raise TypeError('all list entries in locations.values have to be lists.')
            else:
                raise TypeError('locations.values has to be a list of lists.')
        elif isinstance(locations, UniformRange):
            sample = locations.sample()
            if isinstance(sample, list):
                self.locations = sample
            else:
                raise TypeError("locations.sample() must be of type list.")
            if isinstance(locations.range, list):
                self.locations_range = locations.range
            else:
                raise TypeError("locations.range must be of type list.")

        # Read out the 'ordinary' input arguments and save them as attributes
        self._parameter_converter()

        # spatial_frequencies_surround
        if spatial_frequencies_surround is None:
            self.spatial_frequencies_surround = [-6666]  # random iterable label of length>0 beyond parameter range
        elif isinstance(spatial_frequencies_surround, list):
            self.spatial_frequencies_surround = spatial_frequencies_surround
        elif isinstance(spatial_frequencies_surround, FiniteSelection):
            sample = spatial_frequencies_surround.sample()
            if isinstance(sample, list):
                self.spatial_frequencies_surround = sample
            else:
                raise TypeError("spatial_frequencies_surround.sample() must be of type list.")
        elif isinstance(spatial_frequencies_surround, FiniteParameter):
            if type(spatial_frequencies_surround.values) is list:
                self.spatial_frequencies_surround = spatial_frequencies_surround.values
            elif isinstance(spatial_frequencies_surround.values, (int or float)):
                self.spatial_frequencies_surround = [spatial_frequencies_surround.values]
            else:
                raise TypeError('spatial_frequencies_surround.values must be of type list, int or float.')
        elif isinstance(spatial_frequencies_surround, UniformRange):
            sample = spatial_frequencies_surround.sample()
            if isinstance(sample, list):
                self.spatial_frequencies_surround = sample
            else:
                raise TypeError("spatial_frequencies_surround.sample() must be of type list.")
            if isinstance(spatial_frequencies_surround.range, list):
                self.spatial_frequencies_surround_range = spatial_frequencies_surround.range
            else:
                raise TypeError("spatial_frequencies_surround.range must be of type list.")

        # phases_surround
        if phases_surround is None:
            self.phases_surround = [-6666]  # arbitrary iterable label of length > 0 outside of valid parameter range
        elif isinstance(phases_surround, list):
            self.phases_surround = phases_surround
        elif isinstance(phases_surround, FiniteSelection):
            sample = phases_surround.sample()
            if isinstance(sample, list):
                self.phases_surround = sample
            else:
                raise TypeError("phases_surround.sample() must be of type list.")
        elif isinstance(phases_surround, FiniteParameter):
            if type(phases_surround.values) is list:
                self.phases_surround = phases_surround.values
            elif type(phases_surround.values) is float:
                self.phases_surround = [phases_surround.values]
            elif type(phases_surround.values) is int:
                self.phases_surround = np.arange(phases_surround.values) * (2 * pi) / phases_surround.values
            else:
                raise TypeError('phases_surround.values must be either of type list, float or int.')
        elif isinstance(phases_surround, UniformRange):
            sample = phases_surround.sample()
            if isinstance(sample, list):
                self.phases_surround = sample
            else:
                raise TypeError("phases_surround.sample() must be of type list.")
            if isinstance(phases_surround.range, list):
                self.phases_surround_range = phases_surround.range
            else:
                raise TypeError("phases_surround.range must be of type list.")

        # For this class' search methods, we want to get the parameters in an ax-friendly format
        type_check = []
        for arg in self.arg_dict:
            if arg in ["self", "canvas_size", "pixel_boundaries", "phases_surround", "spatial_frequencies_surround"]:
                pass
            else:
                type_check.append(isinstance(self.arg_dict[arg], (FiniteParameter, FiniteSelection, UniformRange)))
        if all(type_check):
            self.auto_params = self._param_dict_for_search(locations=locations,
                                                           sizes_total=sizes_total,
                                                           sizes_center=sizes_center,
                                                           sizes_surround=sizes_surround,
                                                           contrasts_center=contrasts_center,
                                                           contrasts_surround=contrasts_surround,
                                                           orientations_center=orientations_center,
                                                           orientations_surround=orientations_surround,
                                                           spatial_frequencies_center=spatial_frequencies_center,
                                                           phases_center=phases_center,
                                                           grey_levels=grey_levels,
                                                           spatial_frequencies_surround=spatial_frequencies_surround,
                                                           phases_surround=phases_surround)

    def params(self):
        # caution when changing the order of the list entries -> conflict with method params_from_idx()
        return [
            (self.locations, 'location'),
            (self.sizes_total, 'size_total'),
            (self.sizes_center, 'size_center'),
            (self.sizes_surround, 'size_surround'),
            (self.contrasts_center, 'contrast_center'),
            (self.contrasts_surround, 'contrast_surround'),
            (self.orientations_center, 'orientation_center'),
            (self.orientations_surround, 'orientation_surround'),
            (self.spatial_frequencies_center, 'spatial_frequency_center'),
            (self.spatial_frequencies_surround, 'spatial_frequency_surround'),
            (self.phases_center, 'phase_center'),
            (self.phases_surround, 'phase_surround'),
            (self.grey_levels, 'grey_level')
        ]

    def params_from_idx(self, idx):
        # Caution changing the class methods: it is crucial that the index of params matches the correct parameter. The
        # index is defined in the params() method.

        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]

        # if phases_surround was not specified, use the value of phases_center
        if self.phases_surround == [-6666]:
            params[11] = params[10]

        # if spatial_frequencies_surround was not specified
        if self.spatial_frequencies_surround == [-6666]:
            params[9] = params[8]
        return params

    def _parameter_converter(self):
        """ Reads out the type of all the ordinary input arguments and converts them to attributes. """
        for arg_key in self.arg_dict:

            arg_value = self.arg_dict[arg_key]

            # case1: canvas_size, pixel_boundaries, locations, self, phases_surround, spatial_frequencies_surround
            if arg_key in ["canvas_size", "pixel_boundaries", "locations", "self", "phases_surround",
                           "spatial_frequencies_surround"]:
                pass  # already implemented in __init__

            # case2: orientations_center, orientations_surround, phases_surround
            elif arg_key in ["orientations_center", "orientations_surround", "phases_center"]:
                if isinstance(arg_value, list):
                    setattr(self, arg_key, arg_value)
                elif isinstance(arg_value, FiniteSelection):
                    sample = arg_value.sample()  # random sample of n values specified in sizes
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                elif isinstance(arg_value, FiniteParameter):
                    if isinstance(arg_value.values, list):
                        setattr(self, arg_key, arg_value.values)
                    elif isinstance(arg_value.values, float):
                        setattr(self, arg_key, [arg_value.values])
                    elif isinstance(arg_value.values, int):
                        setattr(self, arg_key, np.arange(arg_value.values) * pi / arg_value.values)
                    else:
                        raise TypeError("{}.values must be either of type list, float or int.".format(arg_key))
                elif isinstance(arg_value, UniformRange):
                    sample = arg_value.sample()
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                    if isinstance(arg_value.range, list):
                        setattr(self, arg_key + '_range', arg_value.range)
                    else:
                        raise TypeError("{}.range must be of type list.".format(arg_key))

            # case3: sizes_total, sizes_center, sizes_surround, contrasts_center, contrasts_surround, grey_levels,
            # spatial_frequencies_center
            else:
                if isinstance(arg_value, list):
                    setattr(self, arg_key, arg_value)
                elif isinstance(arg_value, FiniteSelection):
                    sample = arg_value.sample()  # random sample of n values specified in sizes
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                elif isinstance(arg_value, FiniteParameter):
                    if isinstance(arg_value.values, list):
                        setattr(self, arg_key, arg_value.values)
                    elif isinstance(arg_value.values, (int or float)):
                        setattr(self, arg_key, [arg_value.values])
                    else:
                        raise TypeError('sizes_total.values must be of type list, int or float.')
                elif isinstance(arg_value, UniformRange):
                    sample = arg_value.sample()
                    if isinstance(sample, list):
                        setattr(self, arg_key, sample)
                    else:
                        raise TypeError("{}.sample() must be of type list.".format(arg_key))
                    if isinstance(arg_value.range, list):
                        setattr(self, arg_key + '_range', arg_value.range)
                    else:
                        raise TypeError("{}.range must be of type list.".format(arg_key))

    def stimulus(self, location, size_total, size_center, size_surround, contrast_center, contrast_surround,
                 orientation_center, orientation_surround, spatial_frequency_center, spatial_frequency_surround,
                 phase_center, phase_surround, grey_level):
        """
        Args:
            location (list of float): The center position of the Center-Surround stimulus.
            size_total (float): The overall size of the Center-Surround stimulus as the radius in [number of pixels].
            size_center (float): The size of the center as a fraction of the overall size.
            size_surround (float): The size of the surround as a fraction of the overall size.
            contrast_center (float): The contrast of the center grating in %. Takes values from 0 to 1.
            contrast_surround (float): The contrast of the surround grating in %. Takes values from 0 to 1.
            orientation_center (float): The orientation of the center grating.
            orientation_surround (float): The orientation of the surround grating.
            spatial_frequency_center (float): The inverse of the wavelength of the center gratings.
            spatial_frequency_surround (float): The inverse of the wavelength of the surround gratings.
            phase_center (float): The cosine phase-offset of the center grating.
            phase_surround (float): The cosine phase-offset of the surround grating.
            grey_level (float): The mean luminance.

        Returns: Pixel intensities of the desired Center-Surround stimulus as numpy.ndarray.
        """

        if size_center > size_surround:
            raise ValueError("size_center cannot be larger than size_surround.")

        x, y = np.meshgrid(np.arange(self.canvas_size[0]) - location[0],
                           np.arange(self.canvas_size[1]) - location[1])

        R_center = np.array([[np.cos(orientation_center), -np.sin(orientation_center)],
                             [np.sin(orientation_center),  np.cos(orientation_center)]])

        R_surround = np.array([[np.cos(orientation_surround), -np.sin(orientation_surround)],
                               [np.sin(orientation_surround),  np.cos(orientation_surround)]])

        coords = np.stack([x.flatten(), y.flatten()])
        x_center, y_center = R_center.dot(coords).reshape((2, ) + x.shape)
        x_surround, y_surround = R_surround.dot(coords).reshape((2, ) + x.shape)

        norm_xy_center = np.sqrt(x_center ** 2 + y_center ** 2)
        norm_xy_surround = np.sqrt(x_surround ** 2 + y_surround ** 2)

        envelope_center = (norm_xy_center <= size_center * size_total)
        envelope_surround = (norm_xy_surround > size_surround * size_total) * (norm_xy_surround <= size_total)

        grating_center = np.cos(spatial_frequency_center * x_center * (2*pi) + phase_center)
        grating_surround = np.cos(spatial_frequency_surround * x_surround * (2*pi) + phase_surround)

        # add contrast
        amplitude_center = contrast_center * min(abs(self.pixel_boundaries[0] - grey_level),
                                                 abs(self.pixel_boundaries[1] - grey_level))
        amplitude_surround = contrast_surround * min(abs(self.pixel_boundaries[0] - grey_level),
                                                     abs(self.pixel_boundaries[1] - grey_level))

        grating_center_contrast = amplitude_center * grating_center
        grating_surround_contrast = amplitude_surround * grating_surround

        return envelope_center * grating_center_contrast + envelope_surround * grating_surround_contrast

    def _param_dict_for_search(self, locations, sizes_total, sizes_center, sizes_surround, contrasts_center,
                               contrasts_surround, orientations_center, orientations_surround,
                               spatial_frequencies_center, phases_center, grey_levels,
                               spatial_frequencies_surround=None, phases_surround=None):
        """
        Create a dictionary of all bar stimulus parameters in an ax-friendly format.

        Args:
            locations: object from parameters.py module, defining the center of the stimulus.
            sizes_total: object from parameters.py module, defining the total size of the stimulus in [#pixels].
            sizes_center: object from parameters.py module, defining the relative size of the center.
            sizes_surround: object from parameters.py module, defining the relative size of the surround.
            contrasts_center: object from parameters.py module, defining the contrast for the center.
            contrasts_surround: object from parameters.py module, defining the contrast of the surround.
            orientations_center: object from parameters.py module, defining the orientation of the grating of the
                center.
            orientations_surround: object from parameters.py module, defining the orientation of the grating in the
                surround.
            spatial_frequencies_center: object from parameters.py module, defining the spatial frequency in the center.
            phases_center: object from parameters.py module, defining the phase offset of the sinusoidal in the center.
            grey_levels: object from parameters.py module, defining the mean luminance of the image.
            spatial_frequencies_surround: object from parameters.py module, defining the spatial frequency in the
                surround.
            phases_surround: object from parameters.py module, defining the phase offset of the sinusoidal in the
                surround.

        Returns:
            dict of dict: dictionary of all parameters and their respective attributes, i.e. 'name, 'type', 'bounds' and
                'log_scale'.

        """
        rn.seed(None)  # truely random samples

        arg_dict = locals().copy()
        del arg_dict['self']

        param_dict = {}
        for arg_key in arg_dict:
            # "finite case" -> 'type' = choice (more than one value) or 'type' = fixed (only one value)
            if isinstance(arg_dict[arg_key], FiniteParameter) or isinstance(arg_dict[arg_key], FiniteSelection):
                # define the type configuration based on the number of list elements
                if type(getattr(self, arg_key)) is list:
                    if len(getattr(self, arg_key)) > 1:
                        name_type = "choice"
                    else:
                        name_type = "fixed"

                if arg_key == "locations":  # exception handling #1: locations
                    # width
                    if name_type == "choice":
                        name_width = arg_key[:-1] + "_width"
                        param_dict[name_width] = {"name": name_width,
                                                  "type": name_type,
                                                  "values": [float(loc[0]) for loc in getattr(self, arg_key)]}
                        # height
                        name_height = arg_key[:-1] + "_height"
                        param_dict[name_height] = {"name": name_height,
                                                   "type": name_type,
                                                   "values": [float(loc[1]) for loc in getattr(self, arg_key)]}
                    elif name_type == "fixed":
                        name_width = arg_key[:-1] + "_width"
                        param_dict[name_width] = {"name": name_width,
                                                  "type": name_type,
                                                  "value": [float(loc[0]) for loc in getattr(self, arg_key)][0]}
                        # height
                        name_height = arg_key[:-1] + "_height"
                        param_dict[name_height] = {"name": name_height,
                                                   "type": name_type,
                                                   "value": [float(loc[1]) for loc in getattr(self, arg_key)][0]}
                elif arg_key in ["spatial_frequencies_center", "spatial_frequencies_surround"]:
                    arg_key_split = arg_key.split('_')
                    name = "spatial_frequency" + "_" + arg_key_split[-1]
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}
                elif arg_key == "grey_levels":  # grey_levels
                    name = arg_key[:-1]
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}
                else:  # all other inputs
                    arg_key_split = arg_key.split('_')
                    name = arg_key_split[0][:-1] + "_" + arg_key_split[1]
                    if name_type == "choice":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "values": getattr(self, arg_key)}
                    elif name_type == "fixed":
                        param_dict[name] = {"name": name,
                                            "type": name_type,
                                            "value": getattr(self, arg_key)[0]}
            # "infinite case" -> 'type' = range
            elif isinstance(arg_dict[arg_key], UniformRange):
                if arg_key == 'locations':
                    range_name = arg_key + '_range'
                    # width
                    name_width = arg_key[:-1] + "_width"
                    param_dict[name_width] = {"name": name_width,
                                              "type": "range",
                                              "bounds": getattr(self, range_name)[0]}
                    # height
                    name_height = arg_key[:-1] + "_height"
                    param_dict[name_height] = {"name": name_height,
                                               "type": "range",
                                               "bounds": getattr(self, range_name)[1]}
                elif arg_key in ["spatial_frequencies_center", "spatial_frequencies_surround"]:
                    arg_key_split = arg_key.split('_')
                    name = "spatial_frequency" + "_" + arg_key_split[-1]
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}
                elif arg_key == "grey_levels":
                    name = arg_key[:-1]
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}
                else:  # all other inputs
                    arg_key_split = arg_key.split('_')
                    name = arg_key_split[0][:-1] + "_" + arg_key_split[1]
                    range_name = arg_key + "_range"
                    param_dict[name] = {"name": name,
                                        "type": "range",
                                        "bounds": getattr(self, range_name)}

        if spatial_frequencies_surround is None:
            if self.sizes_center == [1.0]:  # circular patch
                param_dict["spatial_frequency_surround"] = {"name": "spatial_frequency_surround",
                                                            "type": "fixed",
                                                            "value": 0.0}
            else:
                param_dict["spatial_frequency_surround"] = param_dict["spatial_frequency_center"].copy()
                param_dict["spatial_frequency_surround"]["name"] = "spatial_frequency_surround"

        if phases_surround is None:
            if self.sizes_center == [1.0]:  # circular patch
                param_dict["phase_surround"] = {"name": "phase_surround",
                                                "type": "fixed",
                                                "value": 0.0}
            else:
                param_dict["phase_surround"] = param_dict["phase_center"].copy()
                param_dict["phase_surround"]["name"] = "phase_surround"

        return param_dict

    def get_image_from_params(self, auto_params):
        """
        Generates the bar stimulus corresponding to the parameters given in auto_params.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'lengths': value3, 'widths' : ...}

        Returns:
            numpy.array: Pixel intensities of the desired bar stimulus.

        """
        auto_params_copy = auto_params.copy()
        auto_params_copy['location'] = [auto_params_copy['location_width'], auto_params_copy['location_height']]
        del auto_params_copy['location_width'], auto_params_copy['location_height']
        return self.stimulus(**auto_params_copy)

    def train_evaluate(self, auto_params, model, data_key, unit_idx):
        """
        Evaluates the activation of a specific neuron in an evaluated (e.g. nnfabrik) model given the bar stimulus
        parameters.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'length': value3, 'width' : ...}
            model (Encoder): evaluated model (e.g. nnfabrik) of interest.
            data_key (str): session ID.
            unit_idx (int): index of the desired model neuron.

        Returns:
            float: The activation of the bar stimulus image of the model neuron specified in unit_idx.
        """
        auto_params_copy = auto_params.copy()
        image = self.get_image_from_params(auto_params_copy)
        image_tensor = torch.tensor(image).expand(1, 1, self.canvas_size[1], self.canvas_size[0]).float()
        activation = model(image_tensor, data_key=data_key).detach().numpy().squeeze()
        return float(activation[unit_idx])

    def find_optimal_stimulus(self, model, data_key, unit_idx, total_trials=30):
        """
        Runs Bayesian parameter optimization to find optimal bar stimulus (refer to https://ax.dev/docs/api.html).

        Args:
            model (Encoder): the underlying model of interest.
            data_key (str): session ID of model.
            unit_idx (int): unit index of desired neuron.
            total_trials (int or None): number of optimization steps (default is 30 trials).

        Returns
            - list of dict: The list entries are dictionaries which store the optimal parameter combinations for the
            corresponding unit. It has the variable name in the key and the optimal value in the values, i.e.
            [{'location_width': value1, 'location_height': value2, 'length': value3, ...}, ...]
            - list of tuple: The unit activations of the found optimal bar stimulus of the form
            [({'activation': mean_unit1}, {'activation': {'activation': sem_unit1}}), ...].
        """
        if not hasattr(self, "auto_params"):
            raise TypeError("find_optimal_stimulus search method only supports input parameters from module "
                            "parameters.py")

        parameters = list(self.auto_params.values())

        # define helper function as input to 'optimize'
        def train_evaluate_helper(auto_params):
            return partial(self.train_evaluate, model=model, data_key=data_key, unit_idx=unit_idx)(auto_params)

        # run Bayesian search
        best_params, values, _, _ = optimize(parameters=parameters.copy(),
                                             evaluation_function=train_evaluate_helper,
                                             objective_name='activation',
                                             total_trials=total_trials)
        return best_params, values

    def find_optimal_stimulus_bruteforce(self, model, data_key, batch_size=100, return_activations=False, unit_idx=None,
                                         plotflag=False):
        """
        Finds optimal parameter combination for all units based on brute force testing method.

        Args:
            model (Encoder): The evaluated model as an encoder class.
            data_key (char): data key or session ID of model.
            batch_size (int or optional): number of images per batch.
            return_activations (bool or None): return maximal activation alongside its parameter combination
            unit_idx (int or None): unit index of the desired model neuron. If not specified, return the best
                parameters for all model neurons (advised, because search is done for all units anyway).
            plotflag (bool or None): if True, plots the evolution of the maximal activation of the number of images
                tested (default: False).

        Returns
            - params (list of dict): The optimal parameter settings for each of the different units
            - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
        """
        if any([isinstance(par, UniformRange) for par in list(self.arg_dict.values())]):
            raise TypeError('This method needs inputs of type FiniteParameter or FiniteSelection.')

        n_images = np.prod(self.num_params())  # number of all parameter combinations
        n_units = model.readout[data_key].outdims  # number of units

        max_act_evo = np.zeros((n_images + 1, n_units))  # init storage of maximal activation evolution
        activations = np.zeros(n_units)  # init activation array for all tested images

        # divide set of images in batches before showing it to the model
        for batch_idx, batch in enumerate(self.image_batches(batch_size)):

            if batch.shape[0] != batch_size:
                batch_size = batch.shape[0]

            # create images and compute activation for current batch
            images_batch = batch.reshape((batch_size,) + tuple(self.canvas_size))
            images_batch = np.expand_dims(images_batch, axis=1)
            images_batch = torch.tensor(images_batch).float()
            activations_batch = model(images_batch, data_key=data_key).detach().numpy().squeeze()

            if plotflag:  # evolution of maximal activation
                for unit in range(0, n_units):
                    for idx, act in enumerate(activations_batch):
                        i = (idx + 1) + batch_idx * batch_size
                        max_act_evo[i, unit] = max(act[unit], max_act_evo[i - 1, unit])

            # max and argmax for current batch
            activations = np.vstack([activations, activations_batch])

        # delete the first row (only zeros) by which we initialized
        activations = np.delete(activations, 0, axis=0)

        # get maximal activations for each unit
        max_activations = np.amax(activations, axis=0)

        # get the image index of the maximal activations
        argmax_activations = np.argmax(activations, axis=0)

        params = [None] * n_units  # init list with parameter dictionaries
        for unit, opt_param_idx in enumerate(argmax_activations):
            params[unit] = self.params_dict_from_idx(opt_param_idx)

        # plot the evolution of the maximal activation for each additional image
        if plotflag:
            fig, ax = plt.subplots()
            for unit in range(0, n_units):
                ax.plot(np.arange(0, n_images + 1), max_act_evo[:, unit])
            plt.xlabel('Number of Images')
            plt.ylabel('Maximal Activation')

        # catch return options
        if unit_idx is not None:
            if return_activations:
                return params[unit_idx], activations[unit_idx], max_activations[unit_idx]
            else:
                return params[unit_idx], activations[unit_idx]
        else:
            if return_activations:
                return params, activations, max_activations
            else:
                return params, activations


class PlaidsGratingSet(CenterSurround):
    """ A class to generate overlapping gratings in a circular patch. """
    def __init__(self, canvas_size, locations, sizes_total, contrasts_preferred, contrasts_overlap, spatial_frequencies,
                 orientations, phases, grey_levels, angles=None, pixel_boundaries=None):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height].
            locations (list of list): The center position of the stimulus.
            sizes_total (list of float): the overall size of the stimulus.
            contrasts_preferred (list of float): the contrast of the preferred circular grating.
            contrasts_overlap (list of float): the contrast of the overlapping circular grating.
            spatial_frequencies (list of float): the spatial frequency of the grating [cycles / pixel].
            orientations (list of float): the orientation of the grating in radians, takes values from [0, 2*pi).
            phases (list of float): the phase offset of the sinusoidal grating.
            grey_levels (list of float): the mean pixel intensity.
            angles (list of float or None): rotation angle [rad] of the overlapping grating relative to the preferred
                grating, defaults to pi/2 (orthogonal).
            pixel_boundaries (list of float or None): Range of values the monitor can display. Handed to the class in
                the format [lower pixel value, upper pixel value], default is [-1,1].
        """

        super().__init__(
            canvas_size=canvas_size,
            locations=locations,
            sizes_total=sizes_total,
            sizes_center=[1],
            sizes_surround=[1],
            contrasts_center=contrasts_preferred,
            contrasts_surround=[0],
            orientations_center=orientations,
            orientations_surround=[0],
            spatial_frequencies_center=spatial_frequencies,
            spatial_frequencies_surround=None,
            phases_center=phases,
            phases_surround=None,
            grey_levels=grey_levels,
            pixel_boundaries=pixel_boundaries
        )

        # Delete unnecessary entries
        # sizes
        del self.sizes_center, self.sizes_surround

        # contrasts_center
        self.contrasts_preferred = self.contrasts_center
        del self.contrasts_center, self.contrasts_surround

        # orientations
        self.orientations = self.orientations_center
        del self.orientations_center, self.orientations_surround

        # phases
        self.phases = self.phases_center
        del self.phases_center, self.phases_surround

        # spatial_frequencies
        self.spatial_frequencies = self.spatial_frequencies_center
        del self.spatial_frequencies_center, self.spatial_frequencies_surround

        # Add attributes not present in __init__ of CenterSurround class
        # contrasts_overlap
        if isinstance(contrasts_overlap, list):
            self.contrasts_overlap = contrasts_overlap
        elif isinstance(contrasts_overlap, FiniteSelection):
            sample = contrasts_overlap.sample()
            if isinstance(sample, list):
                self.contrasts_overlap = sample
            else:
                raise TypeError("contrasts_overlap.sample() must be of type list.")
        elif isinstance(contrasts_overlap, FiniteParameter):
            if type(contrasts_overlap.values) is list:
                self.contrasts_overlap = contrasts_overlap.values
            else:
                raise TypeError('contrasts_orthogonal.values must be of type list.')
        elif isinstance(contrasts_overlap, UniformRange):
            sample = contrasts_overlap.sample()
            if isinstance(sample, list):
                self.contrasts_overlap = sample
            else:
                raise TypeError("contrasts_overlap.sample() must be of type list.")

        # angles
        if angles is None:
            self.angles = [pi / 2]  # orthogonal, 90°
        elif isinstance(angles, list):
            self.angles = angles
        elif isinstance(angles, FiniteSelection):
            sample = angles.sample()
            if isinstance(sample, list):
                self.angles = sample
            else:
                raise TypeError("angles.sample() must be of type list.")
        elif isinstance(angles, FiniteParameter):
            if type(angles.values) is list:
                self.angles = angles.values
            elif type(angles.values) is int:  # linearly spaced angle values from 0° to 180°
                self.angles = np.arange(angles.values) * pi / angles.values
            else:
                raise TypeError('angles.values must be either of type list, float or int.')
        elif isinstance(angles, UniformRange):
            sample = angles.sample()
            if isinstance(sample, list):
                self.angles = sample
            else:
                raise TypeError("angles.sample() must be of type list.")
            if isinstance(angles.range, list):
                self.angles_range = angles.range
            else:
                raise TypeError("angles.range must be of type list.")

    def params(self):
        return [
            (self.locations, 'location'),
            (self.sizes_total, 'size_total'),
            (self.contrasts_preferred, 'contrast_preferred'),
            (self.contrasts_overlap, 'contrast_overlap'),
            (self.spatial_frequencies, 'spatial_frequency'),
            (self.orientations, 'orientation'),
            (self.phases, 'phase'),
            (self.angles, 'angle'),
            (self.grey_levels, 'grey_level')
        ]

    def params_from_idx(self, idx):
        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]
        return params

    def stimulus(self, location, size_total, contrast_preferred, contrast_overlap, orientation, angle,
                 spatial_frequency, phase, grey_level):
        """
        Args:
            location (list of float): center position of stimulus.
            size_total (float or int): radius of the circular stimulus in [#pixels].
            contrast_preferred (float or int): contrast of the preferred grating patch.
            contrast_overlap (float or int): contrast of the overlapping gating patch.
            orientation (float or int): orientation of preferred gating in radians.
            angle (float or int): rotation angle [rad] of the overlapping grating relative to the preferred grating.
            spatial_frequency (float or int): the spatial frequency of the grating.
            phase (float or it): the phase offset of the grating.
            grey_level (float or int): the mean luminance of the image.

        Returns: Pixel intensities of the desired Plaid stimulus as numpy.ndarray.
        """
        circular_grating_preferred = super().stimulus(
            location=location,
            size_total=size_total,
            size_center=1,
            size_surround=1,
            contrast_center=contrast_preferred,
            contrast_surround=0,  # arbitrary float or int
            orientation_center=orientation,
            orientation_surround=0,  # arbitrary float or int
            spatial_frequency_center=spatial_frequency,
            spatial_frequency_surround=0,  # arbitrary float or int
            phase_center=phase,
            phase_surround=0,  # arbitrary float or int
            grey_level=grey_level
            )

        circular_grating_overlap = super().stimulus(
            location=location,
            size_total=size_total,
            size_center=1,
            size_surround=1,
            contrast_center=contrast_overlap,
            contrast_surround=0,  # arbitrary float or int
            orientation_center=orientation + angle,
            orientation_surround=0,  # arbitrary float or int
            spatial_frequency_center=spatial_frequency,
            spatial_frequency_surround=0,  # arbitrary float or int
            phase_center=phase,
            phase_surround=0,  # arbitrary float or int
            grey_level=grey_level
        )

        plaid = circular_grating_preferred + circular_grating_overlap

        return plaid

