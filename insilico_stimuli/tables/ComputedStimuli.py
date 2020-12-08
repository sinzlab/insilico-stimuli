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

class StimuliMethodMixin:
    definition = """
    # contains methods for finding Stimuli and their configurations.
    method_fn                           : varchar(64)   # name of the method function
    method_hash                         : varchar(32)   # hash of the method config
    ---
    method_config                       : longblob      # method configuration object
    method_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    method_comment                      : varchar(256)  # a short comment describing the method
    """

    insert1: Callable[[Mapping], None]
    __and__: Callable[[Mapping], StimuliMethodMixin]
    fetch1: Callable

    import_func = staticmethod(integration.import_module)

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
        Parsing of the method config attributes
        expecting it to be of the format
        {
            attributes: {
                attr1: {
                    path: path_to_function,
                    args: list_of_arguments (optional),
                    kwargs: dict_of_keyword_arguments (optional)
                }
            },
            process_fn: {
                path: path_to_processing_function,
                args: list_of_arguments (optional),
                kwargs: dict_of_keyword_arguments (optional)
            }
        }
        """
        attributes = method_config['attributes']

        for key, value in attributes.items():
            if 'path' not in value:
                attributes[key] = value['args']
                continue

            attr_fn = self.import_func(value['path'])

            if not 'args' in value:
                value['args'] = []
            if not 'kwargs' in value:
                value['kwargs'] = {}

            attr = attr_fn(*value['args'], **value['kwargs'])
            attributes[key] = attr

        return method_config

    def generate_stimuli(self, key: Key) -> np.ndarray:
        """
        Returns the stimuli images given the method config.
        """
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        method_fn = self.import_func(method_fn)

        method_config = self.parse_method_config(method_config)

        stimuli_set = method_fn(method_config['attributes'])

        return stimuli_set.images()

class ComputedStimuliTemplateMixin:
    definition = """
    # contains optimal stimuli
    -> self.method_table
    -> self.trained_model_table
    ---
    stimuli             : longblob      # the stimuli parameters
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
            method(method_config['attributes'],
            model,
            key['data_key'],
            *method_config['process_fn']['args'],
            **method_config['process_fn']['kwargs']
        ))

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
