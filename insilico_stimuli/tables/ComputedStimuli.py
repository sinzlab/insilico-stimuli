"""This module contains mix-ins for the main tables and table templates."""

from __future__ import annotations
import os
import tempfile
from typing import Callable, Mapping, Optional, Dict, Any
from string import ascii_letters
from random import choice

import torch
from torch.nn import Module
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

    def generate_stimuli(self, dataloaders: Dataloaders, model: Module, key: Key) -> Dict[str, Any]:
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        method_fn = self.import_func(method_fn)
        stimuli, score, output = method_fn(dataloaders, model, method_config)
        return dict(key, stimuli=stimuli, score=score, output=output)

class ComputedStimuliTemplateMixin:
    definition = """
    # contains optimal stimuli
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    ---
    stimuli             : attach@minio  # the stimuli as a tensor
    score               : float         # some score depending on the used method function
    output              : attach@minio  # object returned by the method function
    """

    trained_model_table = None
    selector_table = None
    method_table = None
    model_loader_class = integration.ModelLoader
    save = staticmethod(torch.save)
    get_temp_dir = tempfile.TemporaryDirectory

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)

    def make(self, key: Key) -> None:
        dataloaders, model = self.model_loader.load(key=key)
        output_selected_model = self.selector_table().get_output_selected_model(model, key)
        stimuli_entity = self.method_table().generate_stimuli(dataloaders, output_selected_model, key)
        self._insert_stimuli(stimuli_entity)

    def _insert_stimuli(self, stimuli_entity: Dict[str, Any]) -> None:
        """Saves the stimuli to a temporary directory and inserts the prepared entity into the table."""
        with self.get_temp_dir() as temp_dir:
            for name in ("stimuli", "output"):
                self._save_to_disk(stimuli_entity, temp_dir, name)
            self.insert1(stimuli_entity)

    def _save_to_disk(self, stimuli_entity: Dict[str, Any], temp_dir: str, name: str) -> None:
        data = stimuli_entity.pop(name)
        filename = name + "_" + self._create_random_filename() + ".pth.tar"
        filepath = os.path.join(temp_dir, filename)
        self.save(data, filepath)
        stimuli_entity[name] = filepath

    @staticmethod
    def _create_random_filename(length: Optional[int] = 32) -> str:
        return "".join(choice(ascii_letters) for _ in range(length))