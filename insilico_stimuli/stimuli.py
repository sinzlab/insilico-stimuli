# when adding a stimulus class, always add at least the methods "params" and "stimulus".

import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import pi
from numpy import random as rn
from ax.service.managed_loop import optimize
from functools import partial

class StimuliSet:
    """
    Base class for all other stimuli classes.
    """
    def __init__(self):
        pass

    def params(self):
        raise NotImplementedError

    def num_params(self):
        '''
        Returns:
            list: Number of different input parameters for each parameter from the 'params' method.
        '''
        return [len(p[0]) for p in self.params()]

    def stimulus(self, *args, **kwargs):
        raise NotImplementedError

    def params_from_idx(self, idx):
        '''
        labels the different parameter combinations.

        Args:
            idx (int): The index of the desired parameter combination

        Returns:
            list: parameter combinations of the desired index 'idx'
        '''
        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]  # p[0] is parameter content
        return params

    def params_dict_from_idx(self, idx):
        '''
        Args:
            idx (int): The index of the desired parameter combination

        Returns:
            dict: dictionary of the parameter combination specified in 'idx'
        '''
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
        '''
        Generates the images for the desired stimuli.

        Returns: The images of all possible parameter combinations as numpy.ndarray with shape
        ('total number of parameter combinations', 'image height', 'image width')
        '''
        num_stims = np.prod(self.num_params())
        return np.array([self.stimulus_from_idx(i) for i in range(num_stims)])


class GaborSet(StimuliSet):
    """
    A class to generate Gabor stimuli as sinusoidal gratings modulated by Gaussian envelope.
    """
    def __init__(self, canvas_size, center_range, sizes, spatial_frequencies, contrasts, orientations, phases,
                 grey_level, pixel_boundaries=None, eccentricities=None, locations=None, relative_sf=True):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height].
            center_range (list of int): The start and end locations for the center positions of the Gabor [x_start, x_end,
                y_start, y_end].
            sizes (list of float): Controls the size of the Gabor envelope in direction of the longer axis of the
                ellipse. It is measured in pixels (pixel radius). The size corresponds to 4*SD of the Gaussian envelope
                (+/- 2 SD of envelope).
            spatial_frequencies (list of float): The inverse of the wavelength of the cosine factor entered in
                [cycles / pixel]. By setting the parameter 'relative_sf'=True, the spatial frequency depends on size,
                namely [cycles / envelope]. In this case, the value for the spatial frequency reflects how many periods
                fit into the length of 'size' from the center.
                In order to prevent the occurrence of undesired effects at the image borders, the wavelength value
                should be smaller than one fifth of the input image size.
            contrasts (list of float): Defines the amplitude of the stimulus in %. Takes values from 0 to 1. For a
                grey_level=-0.2 and pixel_boundaries=[-1,1], a contrast of 1 (=100%) means the amplitude of the Gabor
                stimulus is 0.8.
            orientations (list or int): The orientation of the normal to the parallel stripes of a Gabor function. Its
                values are given in [rad] and can range from [0,pi). If orientations is handed to the class as an
                integer, e.g. orientations = 4, then the range from [0,pi) will be divided in 4 evenly spaced
                orientations, namely 0*pi/4, 1*pi/4, 2*pi/4 and 3*pi/4.
            phases (list or int): The phase offset in the cosine factor of the Gabor function. Its values are given in
                [rad] and can range from [0,pi). If phases is handed to the class as an integer, e.g. phases = 4, then
                the range from [0,2*pi) will be divided in 4 evenly spaced phase offsets, namely 0*2pi/4, 1*2pi/4,
                2*2pi/4 and 3*2pi/4.
            grey_level (float): Mean luminance/pixel value.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
            eccentricities (list or None): The eccentricity determining the ellipticity of the Gabor. Takes values from
                [0,1].
            locations (list of list or None): list of lists specifying the position of the Gabor. If 'locations' is not
                specified, the Gabors are centered around the grid specified in 'center_range'.
            relative_sf (bool or None): Scale 'spatial_frequencies' by size (True, default) or use absolute units
                (False).
        """
        self.canvas_size = canvas_size
        self.cr = center_range

        if locations is None:
            self.locations = np.array([[x, y] for x in range(self.cr[0], self.cr[1])
                                              for y in range(self.cr[2], self.cr[3])])
        else:
            self.locations = locations

        self.sizes = sizes
        self.spatial_frequencies = spatial_frequencies
        self.contrasts = contrasts
        self.grey_level = grey_level

        if pixel_boundaries is None:
            self.pixel_boundaries = [-1, 1]
        else:
            self.pixel_boundaries = pixel_boundaries

        if type(orientations) is not list:
            self.orientations = np.arange(orientations) * pi / orientations
        else:
            self.orientations = orientations

        if type(phases) is not list:
            self.phases = np.arange(phases) * (2*pi) / phases
        else:
            self.phases = phases

        if eccentricities is None:
            self.gammas = [1]
        else:
            self.gammas = [1 - e ** 2 for e in eccentricities]

        self.relative_sf = relative_sf

    def params(self):
        return [
            (self.locations, 'location'),
            (self.sizes, 'size'),
            (self.spatial_frequencies, 'spatial_frequency'),
            (self.contrasts, 'contrast'),
            (self.orientations, 'orientation'),
            (self.phases, 'phase'),
            (self.gammas, 'gamma')
        ]

    def params_from_idx(self, idx):
        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]
        # Caution changing the class methods: it is crucial that the index of params matches the correct parameter
        if self.relative_sf:
            params[2] /= params[1]  # params[2] is spatial_frequency and params[1] is size.
        return params

    def density(self, xy, gammas, R, size):
        """
        compute the density value for given scalars x and y.

        Args:
            xy (numpy.array): 1x2 vector with the data point of interest
            gammas (list of float): eccentricity parameters
            R (numpy.array): rotation matrix
            size (float): corresponds to the size of the Gaussian

        Returns (numpy.array): density value for the point [x,y]
        """
        S_inv = np.diag(1 / np.array(gammas))
        return np.exp(-0.5 * xy.T @ R @ S_inv @ R.T @ xy / (size / 4)**2)

    def p(self, X, Y, gammas, R, size):
        """
        This function reshapes the grid X and Y into an array of points, computes the density values on these points
        and reshapes them back into a 2D matrix.

        Args:
            X (numpy.array): values of the grid in x-direction
            Y (numpy.array): values of the grid in y-direction
            gammas (list of float): eccentricity parameters
            R (numpy.array): rotation matrix
            size (float): corresponds to the size of the Gaussian

        Returns (numpy.array): 2D matrix containing the density values for the rotated, elliptical Gaussian

        """
        shape = X.shape
        return np.reshape([self.density(np.array([x, y]), gammas, R, size) for x, y in zip(X.ravel(), Y.ravel())],shape)

    def stimulus(self, location, size, spatial_frequency, contrast, orientation, phase, gamma, **kwargs):
        """
        Args:
            location (list of int): The center position of the Gabor.
            size (float): The length of the longer axis of the ellipse of the Gabor envelope.
            spatial_frequency (float): The inverse of the wavelength of the cosine factor.
            contrast (float): Defines the amplitude of the stimulus in %. Takes values from 0 to 1.
            orientation (float): The orientation of the normal to the parallel stripes.
            phase (float): The phase offset of the cosine factor.
            gamma (float): The spatial aspect ratio reflecting the ellipticity of the Gabor.
            **kwargs: Arbitrary keyword arguments.

        Returns: Image of the desired Gabor stimulus as numpy.ndarray.
        """
        x, y = np.meshgrid(np.arange(self.canvas_size[0]) - location[0],
                           np.arange(self.canvas_size[1]) - location[1])
        coords = np.stack([x.flatten(), y.flatten()])

        # rotation matrix R for envelope
        R_env = np.array([[np.cos(-orientation - np.pi/2), -np.sin(-orientation - np.pi/2)],
                      [np.sin(-orientation - np.pi/2),  np.cos(-orientation - np.pi/2)]])
        envelope = self.p(x, y, gammas=[1, gamma], R=R_env, size=size)

        # rotation matrix for grating
        R = np.array([[np.cos(orientation), -np.sin(orientation)],
                      [np.sin(orientation),  np.cos(orientation)]])
        x, y = R.dot(coords).reshape((2, ) + x.shape)
        grating = np.cos(spatial_frequency * (2*pi) * x + phase)

        # add contrast
        gabor_no_contrast = envelope * grating
        amplitude = contrast * min(abs(self.pixel_boundaries[0] - self.grey_level),
                                   abs(self.pixel_boundaries[1] - self.grey_level))
        gabor = amplitude * gabor_no_contrast + self.grey_level

        return gabor

    def dict_param_infinite(self, location=None, size=None, spatial_frequency=[1e-2, 0.25], contrast=[0.4, 1.0],
                            orientation=[0.0, pi], phase=[0.0, pi], gamma=[1e-1, 1.0], grey_level=[-1e-2, 1e-2]):
        """
        Create a dictionary of all Gabor parameters to an ax-friendly format.

        Args:
            location (list of list or None): center of stimulus, default for width is [10.0, # horizontal pixels - 10.0]
                and default for height is [10.0, # vertical pixels - 10.0].
            size (list of float or None): size of envelope, default is [10.0, max(self.canvas_size)].
            spatial_frequency (list of float or None): spatial frequency of grating, default is [1e-2, 0.25].
            contrast (list of float or None): contrast of the image, default is [0.4, 1.0].
            orientation (list of float or None): orientation of grating relative to envelope, default is [0.0, pi].
            phase (list of float or None): phase offset of the grating, default is [0.0, pi].
            gamma (list of float or None): eccentricity parameter of the envelope, default is [1e-1, 1.0].
            grey_level (list of float or None): mean pixel intensity of the stimulus, default is [-1e-2, 1e-2].

        Returns:
            dict of dict: dictionary of all parameters and their respective attributes, i.e. 'name, 'type', 'bounds' and
                'log_scale'.
        """
        if location is None:
            location_width_range = [0.0 + 10.0, float(self.canvas_size[0]) - 10.0]
            location_height_range = [0.0 + 10.0, float(self.canvas_size[1]) - 10.0]
        else:
            location_width_range = location[0]
            location_height_range = location[1]
        location_width = {"name": "location_width", "type": "range", "bounds": location_width_range, "log_scale": False}
        location_height = {"name": "location_height", "type": "range", "bounds": location_height_range,
                           "log_scale": False}

        if size is None:
            size_range = [10.0, float(max(self.canvas_size))]
        else:
            size_range = size
        size = {"name": "size", "type": "range", "bounds": size_range, "log_scale": False}

        sf_range = spatial_frequency
        spatial_frequency = {"name": "spatial_frequency", "type": "range", "bounds": sf_range, "log_scale": False}

        contrast_range = contrast
        contrast = {"name": "contrast", "type": "range", "bounds": contrast_range, "log_scale": False}

        orientation_range = orientation
        orientation = {"name": "orientation", "type": "range", "bounds": orientation_range, "log_scale": False}

        phase_range = phase
        phase = {"name": "phase", "type": "range", "bounds": phase_range, "log_scale": False}

        gamma_range = gamma
        gamma = {"name": "gamma", "type": "range", "bounds": gamma_range, "log_scale": False}

        grey_level_range = grey_level
        grey_level = {"name": "grey_level", "type": "fixed", "bounds": grey_level_range, "log_scale": False}

        param_dict = {"location_width": location_width,
                      "location_height": location_height,
                      "size": size,
                      "spatial_frequency": spatial_frequency,
                      "contrast": contrast,
                      "orientation": orientation,
                      "phase": phase,
                      "gamma": gamma,
                      "grey_level": grey_level}
        return param_dict

    def gen_params_infinite(self, location=None, size=None, spatial_frequency=None, contrast=None, orientation=None,
                            phase=None, gamma=None, grey_level=None):
        """
        Generates random sample for each parameter.

        Args:
            location (list of list or None): center of stimulus, default for width is [0.0, # horizontal pixels]
                and default for height is [0.0, # vertical pixels].
            size (list of float or None): size of envelope, default is [0.0, max(self.canvas_size)].
            spatial_frequency (list of float or None): spatial frequency of grating, default is [5e-3, 0.5].
            contrast (list of float or None): contrast of the image, default is [0.0, 1.0].
            orientation (list of float or None): orientation of grating relative to envelope, default is [0.0, pi].
            phase (list of float or None): phase offset of the grating, default is [0.0, pi].
            gamma (list of float or None): eccentricity parameter of the envelope, default is [1e-1, 1.0].
            grey_level (list of float or None): mean pixel intensity of the stimulus, default is [-1e-2, 1e-2].

        Returns:
            dict: A dictionary containing the parameters with their sampled values.
        """
        rn.seed(None)  # truely random samples
        param_dict = self.dict_param_infinite(location, size, spatial_frequency, contrast, orientation, phase, gamma,
                                              grey_level)
        auto_param_dict = {}
        for param in param_dict:
            if param_dict[param]['type'] == 'range':
                if param == 'location':
                    low_bound1 = param_dict[param]['bounds'][0][0]
                    low_bound2 = param_dict[param]['bounds'][1][0]
                    high_bound1 = param_dict[param]['bounds'][0][1] + 1
                    high_bound2 = param_dict[param]['bounds'][1][1] + 1
                    auto_param_dict[param] = list(rn.randint([low_bound1, low_bound2], [high_bound1, high_bound2]))
                else:
                    low_bound = param_dict[param]['bounds'][0]
                    high_bound = param_dict[param]['bounds'][1]
                    auto_param_dict[param] = [rn.uniform(low_bound, high_bound)]
            elif param_dict[param]['type'] == 'choice':
                n = len(param_dict[param]['bounds'])
                u = rn.uniform(0, n)
                auto_param_dict[param] = param_dict[param]['bounds'][int(np.floor(u))]
            elif param_dict[param]['type'] == 'fixed':
                auto_param_dict[param] = param_dict[param]['bounds']
        return auto_param_dict

    def get_image_from_params(self, auto_params):
        """
        Generates the Gabor corresponding to the parameters given in auto_params.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'size': value3, 'spatial_frequency' : ...}

        Returns:
            numpy.array: Pixel intensities of the desired Gabor stimulus.

        """
        auto_params['location'] = [auto_params['location_width'], auto_params['location_height']]
        del auto_params['location_width'], auto_params['location_height']
        return self.stimulus(**auto_params)

    def train_evaluate(self, auto_params, model, data_key, unit_idx):
        """
        Evaluates the activation of a specific neuron in an evaluated (e.g. nnfabrik) model given the Gabor parameters.

        Args:
            auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
                {'location_width': value1, 'location_height': value2, 'size': value3, 'spatial_frequency' : ...}
            model (Encoder): evaluated model of interest.
            data_key (str): session ID.
            unit_idx (int): index of the desired model neuron.

        Returns:
            float: The activation of the Gabor image of the model neuron specified in unit_idx.
        """
        image = self.get_image_from_params(auto_params)
        image_tensor = torch.tensor(image).expand(1, 1, self.canvas_size[1], self.canvas_size[0]).float()
        activation = model(image_tensor, data_key=data_key).detach().numpy().squeeze()
        return float(activation[unit_idx])

    def find_optimal_gabor_bayes(self, model, data_key, unit_idx, total_trials=30,
                                 location=None, size=None, spatial_frequency=[1e-2, 0.25], contrast=[0.4, 1.0],
                                 orientation=[0.0, pi], phase=[0.0, pi], gamma=[1e-1, 1.0], grey_level=[-1e-2, 1e-2]):
        """
        Runs Bayesian parameter optimization to find optimal Gabor (refer to https://ax.dev/docs/api.html).

        Args:
            model (Encoder): the underlying model of interest.
            data_key (str): session ID of model.
            unit_idx (int): unit index of desired neuron.
            total_trials (int or None): number of optimization steps (default is 30 trials).
            location (list of list or None): center of stimulus, default for width is [10.0, # horizontal pixels - 10.0]
                and default for height is [10.0, # vertical pixels - 10.0].
            size (list of float or None): size of envelope, default is [10.0, max(self.canvas_size)].
            spatial_frequency (list of float or None): spatial frequency of grating, default is [1e-2, 0.25].
            contrast (list of float or None): contrast of the image, default is [0.4, 1.0].
            orientation (list of float or None): orientation of grating relative to envelope, default is [0.0, pi].
            phase (list of float or None): phase offset of the grating, default is [0.0, pi].
            gamma (list of float or None): eccentricity parameter of the envelope, default is [1e-1, 1.0].
            grey_level (list of float or None): mean pixel intensity of the stimulus, default is [-1e-2, 1e-2].

        Returns
            - list of dict: The list entries are dictionaries which store the optimal parameter combinations for the
            corresponding unit. It has the variable name in the key and the optimal value in the values, i.e.
            [{'location_width': value1, 'location_height': value2, 'size': value3, ...}, ...]
            - list of tuple: The unit activations of the found optimal Gabor of the form [({'activation': mean_unit1},
            {'activation': {'activation': sem_unit1}}), ...].
        """
        auto_param_dict = self.dict_param_infinite(location, size, spatial_frequency, contrast, orientation, phase,
                                                   gamma, grey_level)
        parameters = list(auto_param_dict.values())

        # define helper function as input to 'optimize'
        def train_evaluate_helper(auto_params):
            return partial(self.train_evaluate, model=model, data_key=data_key, unit_idx=unit_idx)(auto_params)

        best_params, values, _, _ = optimize(parameters=parameters,
                                             evaluation_function=train_evaluate_helper,
                                             objective_name='activation',
                                             total_trials=total_trials)
        return best_params, values

    def find_optimal_gabor_bruteforce(self, model, data_key, batch_size=100, return_activations=False,
                                      unit_idx=None, plotflag=False):
        """
        Finds optimal parameter combination for all units based on brute force testing method.

        Args:
            model (Encoder): The evaluated model as an encoder class.
            data_key (char): data key or session ID of model.
            batch_size (int, optional): number of images per batch.
            return_activations (bool or None): return maximal activation alongside its parameter combination
            unit_idx (int or None): unit index of the desired model neuron. If not specified, return the best
                parameters for all model neurons.
            plotflag (bool or None): if True, plots the evolution of the maximal activation of the number of images
                tested (default: False).

        Returns
            - params (list of dict): The optimal parameter settings for each of the different units
            - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
        """
        N_images = np.prod(self.num_params())  # number of all parameter combinations
        N_units = model.readout[data_key].outdims  # number of units

        max_act_evo = np.zeros((N_images + 1, N_units))  # init storage of maximal activation evolution
        activations = np.zeros(N_units)  # init activation array for all tested images

        # divide set of images in batches before showing it to the model
        for batch_idx, batch in enumerate(self.image_batches(batch_size)):

            if batch.shape[0] != batch_size:
                batch_size = batch.shape[0]

            # create images and compute activation for current batch
            images_batch = batch.reshape((batch_size,) + tuple(self.canvas_size))
            images_batch = np.expand_dims(images_batch, axis=1)
            images_batch = torch.tensor(images_batch).float()
            activations_batch = model(images_batch, data_key=data_key).detach().numpy().squeeze()

            # evolution of maximal activation
            for unit in range(0, N_units):
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

        params = [None] * N_units  # init list with parameter dictionaries
        for unit, opt_param_idx in enumerate(argmax_activations):
            params[unit] = self.params_dict_from_idx(opt_param_idx)

        # plot the evolution of the maximal activation for each additional image
        if plotflag:
            fig, ax = plt.subplots()
            for unit in range(0, N_units):
                ax.plot(np.arange(0, N_images + 1), max_act_evo[:, unit])
            plt.xlabel('Number of Images')
            plt.ylabel('Maximal Activation')

        # catch return options
        if unit_idx is not None:
            if return_activations:
                return params[unit_idx], activations[unit_idx], max_activations
            else:
                return params[unit_idx], activations[unit_idx]
        else:
            if return_activations:
                return params, activations, max_activations
            else:
                return params, activations



class PlaidsSet(GaborSet):
    """
    A class to generate Plaid stimuli by adding two orthogonal Gabors.
    """
    def __init__(self, canvas_size, center_ranges, sizes, spatial_frequencies, orientations, phases,
                 contrasts_preferred, contrasts_orthogonal, grey_level, pixel_boundaries=None, eccentricities=None,
                 locations=None, relative_sf=True):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height]
            center_ranges (list of int): The ranges for the center locations of the Plaid [x_start, x_end, y_start,
                y_end]
            sizes (list of float): The overall size of the Plaid. Corresponds to 4*SD (+/- 2 SD) of the Gaussian
                envelope.
            spatial_frequencies (list of float): The inverse of the wavelength of the cosine factor entered in
                [cycles / pixel]. By setting the parameter 'relative_sf'=True, the spatial frequency depends on size,
                namely [cycles / envelope]. In this case, the value for the spatial frequency reflects how many periods
                fit into the length of 'size' from the center.
            orientations (list or int): The orientation of the preferred Gabor.
            phases (list or int): The phase offset of the cosine factor of the Plaid. Same value is used for both
                preferred and orthogonal Gabor.
            contrasts_preferred (list of float): Defines the amplitude of the preferred Gabor in %. Takes values from 0
                to 1. For grey_level=-0.2 and pixel_boundaries=[-1,1], a contrast of 1 (=100%) means the amplitude of
                the Gabor stimulus is 0.8.
            contrasts_orthogonal (list of float): Defines the amplitude of the orthogonal Gabor in %. Takes values from
                0 to 1. For grey_level=-0.2 and pixel_boundaries=[-1,1], a contrast of 1 (=100%) means the amplitude
                of the Gabor stimulus is 0.8.
            grey_level (float): Mean luminance/pixel value.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
            eccentricities (list or None): The ellipticity of the Gabor (default: 0). Same value for both preferred and
                orthogonal Gabor. Takes values from [0,1].
            locations (list of list or None): list of lists specifying the location of the Plaid. If 'locations' is not
                specified, the Plaid centers are generated from 'center_ranges' (default is None).
            relative_sf (bool or None): Scale 'spatial_frequencies' by size (True, by default) or use absolute units
                (False).
        """
        self.canvas_size = canvas_size
        self.cr = center_ranges

        if locations is None:
            self.locations = np.array([[x, y] for x in range(self.cr[0], self.cr[1])
                                              for y in range(self.cr[2], self.cr[3])])
        else:
            self.locations = locations

        self.sizes = sizes
        self.spatial_frequencies = spatial_frequencies

        if type(orientations) is not list:
            self.orientations = np.arange(orientations) * pi / orientations
        else:
            self.orientations = orientations

        if type(phases) is not list:
            self.phases = np.arange(phases) * (2*pi) / phases
        else:
            self.phases = phases

        if eccentricities is None:
            self.gammas = [1]
        else:
            self.gammas = [1 - e ** 2 for e in eccentricities]

        self.contrasts_preferred = contrasts_preferred
        self.contrasts_orthogonal = contrasts_orthogonal
        self.grey_level = grey_level

        if pixel_boundaries is None:
            self.pixel_boundaries = [-1, 1]
        else:
            self.pixel_boundaries = pixel_boundaries

        self.relative_sf = relative_sf

    def params(self):
        return [
            (self.locations, 'location'),
            (self.sizes, 'size'),
            (self.spatial_frequencies, 'spatial_frequency'),
            (self.orientations, 'orientation'),
            (self.phases, 'phase'),
            (self.gammas, 'gamma'),
            (self.contrasts_preferred, 'contrast_preferred'),
            (self.contrasts_orthogonal, 'contrast_orthogonal')
        ]

    def stimulus(self, location, size, spatial_frequency, orientation, phase, gamma,
                 contrast_preferred, contrast_orthogonal, **kwargs):
        """
        Args:
            location (list of int): The center position of the Plaid.
            size (float): The overall size of the Plaid envelope.
            spatial_frequency (float): The inverse of the wavelength of the cosine factor of both Gabors.
            orientation (float): The orientation of the preferred Gabor.
            phase (float): The phase offset of the cosine factor for both Gabors.
            gamma (float): The spatial aspect ratio reflecting the ellipticity of both Gabors.
            contrast_preferred (float): Defines the amplitude of the preferred Gabor in %. Takes values from 0 to 1.
            contrast_orthogonal (float): Defines the amplitude of the orthogonal Gabor in %. Takes values from 0 to 1.
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
            **kwargs
        )

        gabor_orthogonal = super().stimulus(
            location=location,
            size=size,
            spatial_frequency=spatial_frequency,
            contrast=contrast_orthogonal,
            orientation=orientation + np.pi/2,
            phase=phase,
            gamma=gamma,
            **kwargs
        )

        plaid = gabor_preferred + gabor_orthogonal

        return plaid


class DiffOfGaussians(StimuliSet):
    """
    A class to generate Difference of Gaussians (DoG) by subtracting two Gaussian functions of different sizes.
    """
    def __init__(self, canvas_size, center_range, sizes, sizes_scale_surround, contrasts, contrasts_scale_surround,
                 grey_level, pixel_boundaries=None, locations=None):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height].
            center_range (list of int): The grid of ranges for the center locations [x_start, x_end, y_start, y_end].
            sizes (list of float): Standard deviation of the center Gaussian.
            sizes_scale_surround (list of float): Scaling factor defining how much larger the standard deviation of the
                surround Gaussian is relative to the size of the center Gaussian. Must have values larger than 1.
            contrasts (list of float): Contrast of the center Gaussian in %. Takes values from -1 to 1.
            contrasts_scale_surround (list of float): Contrast of the surround Gaussian relative to the center Gaussian.
                Should be between 0 and 1.
            grey_level (float): The mean luminance/pixel value.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
            locations (list of list or None): list of lists specifying the center locations. If 'locations' is not
                specified, the center positions are generated from 'center_range' (default is None).
        """
        self.canvas_size = canvas_size
        self.cr = center_range

        if locations is None:
            self.locations = np.array([[x, y] for x in range(self.cr[0], self.cr[1])
                                              for y in range(self.cr[2], self.cr[3])])
        else:
            self.locations = locations

        self.sizes = sizes
        self.sizes_scale_surround = sizes_scale_surround
        self.contrasts = contrasts
        self.grey_level = grey_level
        self.contrasts_scale_surround = contrasts_scale_surround

        if pixel_boundaries is None:
            self.pixel_boundaries = [-1, 1]
        else:
            self.pixel_boundaries = pixel_boundaries

    def params(self):
        return [
            (self.locations, 'location'),
            (self.sizes, 'size'),
            (self.sizes_scale_surround, 'size_scale_surround'),
            (self.contrasts, 'contrast'),
            (self.contrasts_scale_surround, 'contrast_scale_surround')
        ]

    def gaussian_density(self, coords, mean, scale):
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

    def stimulus(self, location, size, size_scale_surround, contrast, contrast_scale_surround, **kwargs):
        """
        Args:
            location (list of int): The center position of the DoG.
            size (float): Standard deviation of the center Gaussian.
            size_scale_surround (float): Scaling factor defining how much larger the standard deviation of the surround
                Gaussian is relative to the size of the center Gaussian. Must have values larger than 1.
            contrast (float): Contrast of the center Gaussian in %. Takes values from 0 to 1.
            contrast_scale_surround (float): Contrast of the surround Gaussian relative to the center Gaussian.
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
        amplitude_required = contrast * min(np.abs(self.pixel_boundaries[0] - self.grey_level),
                                            np.abs(self.pixel_boundaries[1] - self.grey_level))
        contrast_scaling = amplitude_required / amplitude_current

        diff_of_gaussians = contrast_scaling * center_surround + self.grey_level

        return diff_of_gaussians


class CenterSurround(StimuliSet):
    """
    A class to generate 'Center-Surround' stimuli with optional center and/or surround gratings.
    """
    def __init__(self, canvas_size, center_range, sizes_total, sizes_center, sizes_surround, contrasts_center,
                 contrasts_surround, orientations_center, orientations_surround, spatial_frequencies_center,
                 phases_center, grey_level, spatial_frequencies_surround=None, phases_surround=None,
                 pixel_boundaries=None, locations=None):
        """
        Args:
            canvas_size (list of int): The canvas size [width, height].
            center_range (list of int): The grid of ranges for the center locations [x_start, x_end, y_start, y_end].
            sizes_total (list of float): The overall size of the Center-Surround stimulus.
            sizes_center (list of float): The size of the center as a fraction of the overall size. Takes values from 0
                to 1. 'size_center' is a scaling factor for 'size_total' so that 'size_center' * 'size_total' = radius
                of inner circle.
            sizes_surround (list of float): The size of the surround as a fraction of the overall size. Takes values
                from 0 to 1.
            contrasts_center (list of float): The contrast of the center grating in %. Takes values from 0 to 1.
            contrasts_surround (list of float): The contrast of the surround grating in %. Takes values from 0 to 1.
            orientations_center (list or int): The orientation of the center gratings. Takes values from 0 to pi. If
                orientations_center is handed to the class as an integer, e.g. orientations_center = 3, then the range
                from [0,pi) will be divided into 3 evenly spaced orientations, namely 0*pi/3, 1*pi/3 and 2*pi/3.
            orientations_surround (list or int): The orientation of the surround gratings. Takes values from 0 to pi. If
                orientations_surround is handed to the class as an integer, e.g. orientations_surround = 3, then the
                range from [0,pi) will be divided into 3 evenly spaced orientations, namely 0*pi/3, 1*pi/3 and 2*pi/3.
            spatial_frequencies_center (list of float): The inverse of the wavelength of the center gratings in
                absolute units [cycles / pixel].
            spatial_frequencies_surround (list of float or None): The inverse of the wavelength of the center gratings
                in absolute units [cycles / pixel]. If not specified, use same value as in 'spatial_frequencies_center'.
            phases_center (list or int): The phase offset of the center sinusoidal gratings. Takes values from -pi to
                pi.
            phases_surround (list or int or None): The phase offset of the surround sinusoidal gratings. Takes values
                from -pi to pi. If not specified, use same value as in 'phases_center'.
            grey_level (float): The mean luminance/pixel value.
            pixel_boundaries (list of float or None): Range of values the monitor can display. Handed to the class in
                the format [lower pixel value, upper pixel value], default is [-1,1].
            locations (list of list or None): list of lists specifying the center locations (default: None). If
                'locations' is not specified, the center positions are generated from 'center_range'.
        """
        self.canvas_size = canvas_size
        self.cr = center_range

        if locations is None:
            self.locations = np.array([[x, y] for x in range(self.cr[0], self.cr[1])
                                              for y in range(self.cr[2], self.cr[3])])
        else:
            self.locations = locations

        self.sizes_total = sizes_total
        self.sizes_center = sizes_center
        self.sizes_surround = sizes_surround
        self.contrasts_center = contrasts_center
        self.contrasts_surround = contrasts_surround
        self.grey_level = grey_level

        if pixel_boundaries is None:
            self.pixel_boundaries = [-1, 1]
        else:
            self.pixel_boundaries = pixel_boundaries

        if type(orientations_center) is not list:
            self.orientations_center = np.arange(orientations_center) * pi / orientations_center
        else:
            self.orientations_center = orientations_center

        if type(orientations_surround) is not list:
            self.orientations_surround = np.arange(orientations_surround) * pi / orientations_surround
        else:
            self.orientations_surround = orientations_surround

        self.spatial_frequencies_center = spatial_frequencies_center

        if spatial_frequencies_surround is None:
            self.spatial_frequencies_surround = [-6666]  # random iterable label of length>0 beyond parameter range
        else:
            self.spatial_frequencies_surround = spatial_frequencies_surround

        if type(phases_center) is not list:
            self.phases_center = np.arange(phases_center) * (2*pi) / phases_center
        else:
            self.phases_center = phases_center

        if phases_surround is None:
            self.phases_surround = [-6666]  # arbitrary iterable label of length > 0 outside of valid parameter range
        elif type(phases_surround) is not list:
            self.phases_surround = np.arange(phases_surround) * (2 * pi) / phases_surround
        else:
            self.phases_surround = phases_surround

    def params(self):
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
            (self.phases_surround, 'phase_surround')
        ]

    def params_from_idx(self, idx):
        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]
        # Caution changing the class methods: it is crucial that the index of params matches the correct parameter
        if self.phases_surround == [-6666]:  # if phases_surround was not specified, use the value of phases_center
            params[11] = params[10]
        if self.spatial_frequencies_surround == [-6666]:  # if spatial_frequencies_surround was not specified
            params[9] = params[8]
        return params

    def stimulus(self, location, size_total, size_center, size_surround, contrast_center, contrast_surround,
                 orientation_center, orientation_surround, spatial_frequency_center, spatial_frequency_surround,
                 phase_center, phase_surround):
        """
        Args:
            location (list of int): The center position of the Center-Surround stimulus.
            size_total (float): The overall size of the Center-Surround stimulus.
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

        Returns: Pixel intensities of the desired Center-Surround stimulus as numpy.ndarray.
        """

        if size_center > size_surround:
            raise ValueError("size_center cannot be larger than size_surround")

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
        amplitude_center = contrast_center * min(abs(self.pixel_boundaries[0] - self.grey_level),
                                                 abs(self.pixel_boundaries[1] - self.grey_level))
        amplitude_surround = contrast_surround * min(abs(self.pixel_boundaries[0] - self.grey_level),
                                                     abs(self.pixel_boundaries[1] - self.grey_level))

        grating_center_contrast = amplitude_center * grating_center
        grating_surround_contrast = amplitude_surround * grating_surround

        return envelope_center * grating_center_contrast + envelope_surround * grating_surround_contrast

