import numpy as np

import torch

import matplotlib.pyplot as plt

from insilico_stimuli.parameters import *

def bruteforce(stimulus_set, model, data_key, batch_size=100, return_activations=False, unit_idx=None,
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
    if any([isinstance(par, UniformRange) for par in list(stimulus_set.arg_dict.values())]):
        raise TypeError('This method needs inputs of type FiniteParameter or FiniteSelection.')

    n_images = np.prod(stimulus_set.num_params())  # number of all parameter combinations
    n_units = model.readout[data_key].outdims  # number of units

    max_act_evo = np.zeros((n_images + 1, n_units))  # init storage of maximal activation evolution
    activations = np.zeros(n_units)  # init activation array for all tested images

    # divide set of images in batches before showing it to the model
    for batch_idx, batch in enumerate(stimulus_set.image_batches(batch_size)):

        if batch.shape[0] != batch_size:
            batch_size = batch.shape[0]

        # create images and compute activation for current batch
        images_batch = batch.reshape((batch_size,) + tuple(stimulus_set.canvas_size))
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
        params[unit] = stimulus_set.params_dict_from_idx(opt_param_idx)

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