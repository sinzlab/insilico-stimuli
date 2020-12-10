"""This module contains mix-ins for the main tables and table templates."""

from __future__ import annotations

import datajoint as dj
import numpy as np

from typing import Callable, Mapping, Dict, Any

from torch.utils.data import DataLoader

from nnfabrik.utility.dj_helpers import make_hash
from nnfabrik.utility.nnf_helper import dynamic_import, split_module_name, FabrikCache

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

def import_module(path):
    return dynamic_import(*split_module_name(path))

class InsilicoStimuliSetMixin:
    definition = """
    # contains stimuli sets and their configurations.
    stimulus_fn                           : varchar(64)   # name of the set function
    stimulus_hash                         : varchar(32)   # hash of the set config
    ---
    stimulus_config                       : longblob      # set configuration object
    stimulus_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    stimulus_comment                      : varchar(256)  # a short comment describing the set
    """

    insert1: Callable[[Mapping], None]
    __and__: Callable[[Mapping], InsilicoStimuliSetMixin]
    fetch1: Callable

    import_func = staticmethod(import_module)

    def add_set(self, stimulus_fn: str, stimulus_config: Mapping, comment: str = "") -> None:
        self.insert1(
            dict(
                stimulus_fn=stimulus_fn,
                stimulus_hash=make_hash(stimulus_config),
                stimulus_config=stimulus_config,
                stimulus_comment=comment,
            )
        )

    def parse_stimulus_config(self, stimulus_config):
        """
        Parsing of the set config attributes to args and kwargs format, which is passable to the stimulus_fn
        expecting it to be of the format
        {
            parameter1: {
                path: path_to_type_function,
                args: list_of_arguments (optional),
                kwargs: dict_of_keyword_arguments (optional)
            },
            parameter1: {
                path: path_to_type_function,
                args: list_of_arguments (optional),
                kwargs: dict_of_keyword_arguments (optional)
            }
        }
        """

        for key, value in stimulus_config.items():
            if not isinstance(value, dict):
                continue

            attr_fn = self.import_func(value['path'])

            if not 'args' in value:
                value['args'] = []
            if not 'kwargs' in value:
                value['kwargs'] = {}

            attr = attr_fn(*value['args'], **value['kwargs'])
            stimulus_config[key] = attr

        return stimulus_config

    def images(self, key: Key, canvas_size: [int, int]) -> np.ndarray:
        """
        Returns the stimuli images given the set config.
        """
        stimulus_fn, stimulus_config = (self & key).fetch1("stimulus_fn", "stimulus_config")
        stimulus_fn = self.import_func(stimulus_fn)

        stimulus_config = self.parse_stimulus_config(stimulus_config)
        stimulus_config['canvas_size'] = canvas_size

        StimulusSet = stimulus_fn(**stimulus_config)

        return StimulusSet.images()

class StimuliOptimizeMethodMixin:
    definition = """
    # contains methods for optimizing stimuli
    method_fn                           : varchar(128)   # name of the set function
    method_hash                         : varchar(32)   # hash of the set config
    ---
    method_config                       : longblob      # set configuration object
    method_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    method_comment                      : varchar(256)  # a short comment describing the set
    """

    insert1: Callable[[Mapping], None]
    __and__: Callable[[Mapping], StimuliOptimizeMethodMixin]
    fetch1: Callable

    import_func = staticmethod(import_module)

    def add_method(self, method_fn: str, method_config: Mapping, comment: str = "") -> None:
        self.insert1(
            dict(
                method_fn=method_fn,
                method_hash=make_hash(method_config),
                method_config=method_config,
                method_comment=comment,
            )
        )

    def parse_method_config(self, method_config):
        """
        Parsing of the set config attributes to args and kwargs format, which is passable to the stimulus_fn
        expecting it to be of the format
        {
            parameter1: {
                path: path_to_type_function,
                args: list_of_arguments (optional),
                kwargs: dict_of_keyword_arguments (optional)
            },
            parameter2: {
                path: path_to_type_function,
                args: list_of_arguments (optional),
                kwargs: dict_of_keyword_arguments (optional)
            }
        }
        """

        for key, value in method_config.items():
            if not isinstance(value, dict):
                continue

            attr_fn = self.import_func(value['path'])

            if not 'args' in value:
                value['args'] = []
            if not 'kwargs' in value:
                value['kwargs'] = {}

            attr = attr_fn(*value['args'], **value['kwargs'])
            method_config[key] = attr

        return method_config

class OptimisedStimuliTemplateMixin:
    definition = """
    # contains optimal stimuli
    -> self.optimisation_method_table
    -> self.StimulusSet_table
    -> self.trained_model_table
    ---
    average_score = 0.      : float        # average score depending on the used method function
    """

    trained_model_table = None
    optimisation_method_table = None
    unit_table = None
    StimulusSet_table = None

    model_loader_class = FabrikCache

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)

    class Units(dj.Part):
        definition = """
        # Scores for Individual Neurons
        -> master
        -> master.unit_table
        ---
        stimulus         : longblob
        score            : float
        """

    def get_stimulus_set(self, key):
        StimulusSet = self.StimulusSet_table()

        stimulus_config, stimulus_fn = (StimulusSet & key).fetch1('stimulus_config', 'stimulus_fn')
        stimulus_config = StimulusSet.parse_stimulus_config(stimulus_config)

        stimulus_fn = StimulusSet.import_func(stimulus_fn)

        return stimulus_config, stimulus_fn

    def get_method(self, key):
        method = self.optimisation_method_table()

        method_config, method_fn = (method & key).fetch1('method_config', 'method_fn')
        method_config = method.parse_method_config(method_config)

        method_fn = method.import_func(method_fn)

        return method_config, method_fn

    def make(self, key: Key) -> None:
        stimulus_config, stimulus_fn = self.get_stimulus_set(key)
        method_config, method_fn = self.get_method(key)

        dataloaders, model = self.model_loader.load(key=key)

        batch = next(iter(list(dataloaders['test'].values())[0]))
        _, _, w, h = batch.inputs.shape
        canvas_size = [w, h]
        stimulus_config['canvas_size'] = canvas_size

        data_keys = list(dataloaders['test'].keys())

        self.insert1(key, ignore_extra_fields=True)

        for data_key in data_keys:
            stimuli, scores = method_fn(
                stimulus_fn(**stimulus_config),
                model,
                data_key,
                **method_config
            )
            for idx, (stimulus, score) in enumerate(zip(stimuli, scores)):
                unit_key = dict(unit_index=idx, data_key=data_key)
                unit_type = ((self.unit_table & key) & unit_key).fetch1("unit_type")
                unit_id = ((self.unit_table & key) & unit_key).fetch1("unit_id")

                stimuli_entity = dict(
                    data_key=data_key,
                    stimulus=stimulus,
                    score=score,
                    unit_type=unit_type,
                    unit_id=unit_id,
                    **key
                )
                self.Units.insert1(stimuli_entity, ignore_extra_fields=True)
