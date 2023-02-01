import torch

from insilico_stimuli.parameters import *

from ax.service.managed_loop import optimize
from functools import partial


def param_dict_for_search(StimuliSet):
    """
        Create a dictionary of all stimulus parameters in an ax-friendly format.

        Args:
            StimuliSet (StimuliSet): instance of the StimuliSet class.

        Returns:
            dict of dict: dictionary of all parameters and their respective attributes, i.e. 'name, 'type', 'bounds' and
                'log_scale'.
    """
    arg_dict = StimuliSet.arg_dict

    param_dict = {}
    for arg_key in arg_dict:
        # "finite case" -> 'type' = choice (more than one value) or 'type' = fixed (only one value)
        if isinstance(arg_dict[arg_key], FiniteParameter) or isinstance(arg_dict[arg_key], FiniteSelection):
            # define the type configuration based on the number of list elements
            if type(getattr(StimuliSet, arg_key)) is list:
                if len(getattr(StimuliSet, arg_key)) > 1:
                    name_type = "choice"
                else:
                    name_type = "fixed"

            if arg_key == 'location' or arg_key == 'center':  # exception handling #1: locations
                # width
                if name_type == "choice":
                    name_width = arg_key + "_width"
                    param_dict[name_width] = {"name": name_width,
                                              "type": name_type,
                                              "values": [float(loc[0]) for loc in getattr(StimuliSet, arg_key)]}
                    # height
                    name_height = arg_key + "_height"
                    param_dict[name_height] = {"name": name_height,
                                               "type": name_type,
                                               "values": [float(loc[1]) for loc in getattr(StimuliSet, arg_key)]}
                elif name_type == "fixed":
                    name_width = arg_key + "_width"
                    param_dict[name_width] = {"name": name_width,
                                              "type": name_type,
                                              "value": [float(loc[0]) for loc in getattr(StimuliSet, arg_key)][0]}
                    # height
                    name_height = arg_key + "_height"
                    param_dict[name_height] = {"name": name_height,
                                               "type": name_type,
                                               "value": [float(loc[1]) for loc in getattr(StimuliSet, arg_key)][0]}
            else:
                name = arg_key
                if name_type == "choice":
                    param_dict[name] = {"name": name,
                                        "type": name_type,
                                        "values": getattr(StimuliSet, arg_key)}
                elif name_type == "fixed":
                    param_dict[name] = {"name": name,
                                        "type": name_type,
                                        "value": getattr(StimuliSet, arg_key)[0]}

        # "infinite case" -> 'type' = range
        elif isinstance(arg_dict[arg_key], UniformRange):
            if arg_key == 'location' or arg_key == 'center':
                range_name = arg_key + '_range'
                # width
                name_width = arg_key + "_width"
                param_dict[name_width] = {"name": name_width,
                                          "type": "range",
                                          "bounds": getattr(StimuliSet, range_name)[0]}
                # height
                name_height = arg_key + "_height"
                param_dict[name_height] = {"name": name_height,
                                           "type": "range",
                                           "bounds": getattr(StimuliSet, range_name)[1]}
            else:
                name = arg_key
                range_name = arg_key + "_range"
                param_dict[name] = {"name": name,
                                    "type": "range",
                                    "bounds": getattr(StimuliSet, range_name)}
    return param_dict


def train_evaluate(auto_params, StimuliSet, model, unit):
    """
    Evaluates the activation of a specific neuron in an evaluated (e.g. nnfabrik) model given the bar stimulus
    parameters.

    Args:
        auto_params (dict): A dictionary which has the parameter names as keys and their realization as values, i.e.
            {'location_width': value1, 'location_height': value2, 'length': value3, 'width' : ...}
        StimuliSet (StimuliSet): instance of the StimuliSet class.
        model (Encoder): evaluated model (e.g. nnfabrik) of interest.
        unit (int): index of the desired model neuron.

    Returns:
        float: The activation of the bar stimulus image of the model neuron specified in unit_idx.
    """
    auto_params_copy = auto_params.copy()

    if 'location_width' in auto_params_copy and 'location_height' in auto_params_copy:
        auto_params_copy['location'] = [auto_params_copy['location_width'], auto_params_copy['location_height']]
        del auto_params_copy['location_width'], auto_params_copy['location_height']

    if 'center_width' in auto_params_copy and 'center_height' in auto_params_copy:
        auto_params_copy['center'] = [auto_params_copy['center_width'], auto_params_copy['center_height']]
        del auto_params_copy['center_width'], auto_params_copy['center_height']

    image = StimuliSet.stimulus(StimuliSet.canvas_size, **auto_params_copy)()

    image_tensor = torch.tensor(image).float()

    activation = model(image_tensor).detach().cpu().numpy().squeeze()

    return float(activation[unit])


def bayesian_search(stimulus, model, unit, seed, total_trials=30, **kwargs):
    """
        Finds optimal parameter combination for all units based on the gradient descent method.

        Args:
            stimulus (stimulus): Instance of a Stimulus class.
            model (Encoder): The evaluated model as an encoder class.
            unit (int or None) (optional): unit index of the desired model neuron. If not specified, return the best
                parameters for all model neurons (advised, because search is done for all units anyway).
            seed (int): random seed for reproducibility
            total_trials (int): total number of trials in the optimization, default=30
        Returns
            - stimuli_params (list of dict): The optimal parameter settings for each of the different units
            - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
    """

    StimuliSet = stimulus.to_set()

    auto_params = param_dict_for_search(StimuliSet)

    parameters = list(auto_params.values())

    # define helper function as input to 'optimize'
    def train_evaluate_helper(auto_params):
        return partial(train_evaluate, StimuliSet=StimuliSet, model=model, unit=unit)(auto_params)

    # run Bayesian search
    best_params, values, _, _ = optimize(parameters=parameters.copy(),
                                         evaluation_function=train_evaluate_helper,
                                         objective_name='score',
                                         total_trials=total_trials,
                                         random_seed=seed)

    return best_params, values[0]['score']
