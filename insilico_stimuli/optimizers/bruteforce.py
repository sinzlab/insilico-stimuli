import numpy as np

import torch

from insilico_stimuli.parameters import *

from tqdm import tqdm

def bruteforce(StimulusSet, model, batch_size=100, unit_idx=None):
    """
    Finds optimal parameter combination for all units based on brute force testing method.

    Args:
        StimulusSet (StimulusSet): Instance of the StimulusSet class.
        model (Encoder): The evaluated model as an encoder class.
        batch_size (int or optional): number of images per batch.
        unit_idx (int or None): unit index of the desired model neuron. If not specified, return the best
            parameters for all model neurons (advised, because search is done for all units anyway).

    Returns
        - params (list of dict): The optimal parameter settings for each of the different units
        - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
    """
    if any([isinstance(par, UniformRange) for par in list(StimulusSet.arg_dict.values())]):
        raise TypeError('This method needs inputs of type FiniteParameter or FiniteSelection.')

    n_images = np.prod(StimulusSet.num_params())  # number of all parameter combinations
    # n_units = model.readout[data_key].outdims  # number of units

    argmax_activations = None
    max_activations = None

    # divide set of images in batches before showing it to the model
    for batch_idx, batch in tqdm(enumerate(StimulusSet.image_batches(batch_size)), total=n_images // batch_size):

        if batch.shape[0] != batch_size:
            batch_size = batch.shape[0]

        # create images and compute activation for current batch
        images_batch = batch.reshape((batch_size,) + tuple(StimulusSet.canvas_size))
        images_batch = np.expand_dims(images_batch, axis=1)
        images_batch = torch.tensor(images_batch).float()
        activations_batch = model(images_batch).detach().numpy().squeeze()

        # Initialize the activation vectors from the initial pass through the model
        if not argmax_activations or not max_activations:
            n_units = activations_batch.shape[1]

            argmax_activations = np.zeros(n_units).astype(int)
            max_activations = np.zeros(n_units)

        new_max_idx = activations_batch.argmax(0)
        new_max_activations = activations_batch.T[range(len(new_max_idx)), new_max_idx]

        argmax_activations[new_max_activations > max_activations] = new_max_idx[
                                                                        new_max_activations > max_activations] \
                                                                    + batch_size * batch_idx
        max_activations[new_max_activations > max_activations] = new_max_activations[
            new_max_activations > max_activations]

    params = [None] * n_units  # init list with parameter dictionaries
    for unit, opt_param_idx in enumerate(argmax_activations):
        params[unit] = StimulusSet.params_dict_from_idx(opt_param_idx)

    # catch return options
    if unit_idx is not None:
        return params[unit_idx], max_activations[unit_idx]
    else:
        return params, max_activations