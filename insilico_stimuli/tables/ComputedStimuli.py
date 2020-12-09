"""This module contains mix-ins for the main tables and table templates."""

from __future__ import annotations

import datajoint as dj
import numpy as np

from typing import Callable, Mapping, Dict, Any

from torch.utils.data import DataLoader

from nnfabrik.utility.dj_helpers import make_hash

from . import integration

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

class StimuliSetsMixin:
    definition = """
    # contains stimuli sets and their configurations.
    set_fn                           : varchar(64)   # name of the set function
    set_hash                         : varchar(32)   # hash of the set config
    ---
    set_config                       : longblob      # set configuration object
    set_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    set_comment                      : varchar(256)  # a short comment describing the set
    """

    insert1: Callable[[Mapping], None]
    __and__: Callable[[Mapping], StimuliSetsMixin]
    fetch1: Callable

    import_func = staticmethod(integration.import_module)

    def add_set(self, set_fn: str, set_config: Mapping, comment: str = "") -> None:
        self.insert1(
            dict(
                set_fn=set_fn,
                set_hash=make_hash(set_config),
                set_config=set_config,
                set_comment=comment,
            )
        )

    def parse_set_config(self, set_config):
        """
        Parsing of the set config attributes to args and kwargs format, which is passable to the set_fn
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

        for key, value in set_config.items():
            if 'path' not in value:
                set_config[key] = value['args']
                continue

            attr_fn = self.import_func(value['path'])

            if not 'args' in value:
                value['args'] = []
            if not 'kwargs' in value:
                value['kwargs'] = {}

            attr = attr_fn(*value['args'], **value['kwargs'])
            set_config[key] = attr

        return set_config

    def generate_stimuli(self, key: Key) -> np.ndarray:
        """
        Returns the stimuli images given the set config.
        """
        set_fn, set_config = (self & key).fetch1("set_fn", "set_config")
        set_fn = self.import_func(set_fn)

        set_config = self.parse_set_config(set_config)

        stimuli_set = set_fn(**set_config)

        return stimuli_set.images()

class ComputedStimuliTemplateMixin:
    definition = """
    # contains optimal stimuli
    -> self.method_table
    -> self.trained_model_table
    ---
    stimuli             : longblob     # the stimuli parameters
    average_score       : float        # average score depending on the used method function
    """

    trained_model_table = None
    method_table = None
    unit_table = None

    model_loader_class = integration.ModelLoader

    insert1: Callable[[Mapping], None]

    class Units(dj.Part):
        definition = """
        # Scores for Individual Neurons
        -> master
        -> master.unit_table
        ---
        score           : float
        """

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)

    def make(self, key: Key) -> None:
        dataloaders, model = self.model_loader.load(key=key)

        method = self.method_table()
        method_config = method.parse_method_config(key['method_config'])

        proccess_fn = method.import_func(method_config['process_fn']['path'])

        if not 'args' in method_config['process_fn']:
            method_config['process_fn']['args'] = []
        if not 'kwargs' in method_config['process_fn']:
            method_config['process_fn']['kwargs'] = {}

        stimuli_entities = proccess_fn(
            method(**method_config['attributes']),
            model,
            key['data_key'],
            *method_config['process_fn']['args'],
            **method_config['process_fn']['kwargs']
        )

        self.insert1(key, ignore_extra_fields=True)

        for unit_index, stimuli_entity in enumerate(stimuli_entities):
            if "unit_id" in key: key.pop("unit_id")
            if "unit_type" in key: key.pop("unit_type")

            unit_key = dict(unit_index=unit_index, data_key=key['data_key'])
            unit_type = ((self.unit_table & key) & unit_key).fetch1("unit_type")
            unit_id = ((self.unit_table & key) & unit_key).fetch1("unit_id")

            stimuli_entity["unit_id"] = unit_id
            stimuli_entity["unit_type"] = unit_type

            self.Units.insert1(stimuli_entity, ignore_extra_fields=True)
