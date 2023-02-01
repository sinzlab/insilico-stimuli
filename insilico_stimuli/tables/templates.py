"""This module the table templates."""

from __future__ import annotations

import datajoint as dj
import numpy as np

from typing import Callable, Mapping, Dict, Any
from functools import partial

from torch.utils.data import DataLoader

from nnfabrik.utility.nnf_helper import FabrikCache

from .main import ExperimentMethod, ExperimentSeed, InsilicoStimuliSet

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]


class ExperimentTemplate(dj.Computed):
    trained_model_table = None
    unit_table = None
    previous_experiment_table = None
    experiment_method_table = ExperimentMethod
    StimulusSet_table = InsilicoStimuliSet

    definition = """
    # contains optimal stimuli
    -> self.experiment_method_table
    -> self.StimulusSet_table
    -> self.trained_model_table
    ---
    average_score = 0.:                float        # average score depending on the used method function
    """

    model_loader_class = FabrikCache

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)
        super().__init__(*args, **kwargs)

    class Units(dj.Part):
        definition = """
        # Scores for Individual Neurons
        -> master
        -> master.unit_table
        ---
        output           : longblob
        score            : float
        """

    def get_method(self, key):
        method = self.experiment_method_table()

        method_config, method_fn = (method & key).fetch1('method_config', 'method_fn')
        method_config = method.parse_method_config(method_config)

        method_fn = method.import_func(method_fn)

        return method_config, method_fn

    def get_experiment_output(self, key, dataloaders, model, method_fn, method_config, stimulus_set):
        data_keys = list(dataloaders['test'].keys())

        stimuli_entities = []
        for data_key in data_keys:
            model.forward = partial(model.forward, data_key=data_key)

            outputs, scores = method_fn(
                stimulus_set,
                model,
                **method_config
            )

            for idx, (output, score) in enumerate(zip(outputs, scores)):
                unit_key = dict(unit_index=idx, data_key=data_key)
                unit_type = ((self.unit_table & key) & unit_key).fetch1("unit_type")
                unit_id = ((self.unit_table & key) & unit_key).fetch1("unit_id")

                stimuli_entity = dict(
                    data_key=data_key,
                    output=output,
                    score=score,
                    unit_type=unit_type,
                    unit_id=unit_id,
                    **key
                )

                stimuli_entities.append(stimuli_entity)

        return stimuli_entities

    @staticmethod
    def compute_average_score(experiment_entities):
        scores = [experiment_entity['score'] for experiment_entity in experiment_entities]
        average_score = np.mean(scores)
        return average_score

    def make(self, key: Key) -> None:
        stimulus_set = self.StimulusSet_table().load(key=key)
        method_config, method_fn = self.get_method(key)

        dataloaders, model = self.model_loader.load(key=key)
        model.cuda().eval()

        experiment_entities = self.get_experiment_output(key,
                                                         dataloaders, model,
                                                         method_fn, method_config,
                                                         stimulus_set)

        key['average_score'] = self.compute_average_score(experiment_entities)

        self.insert1(key, ignore_extra_fields=True)
        self.Units.insert(experiment_entities, ignore_extra_fields=True)


class ExperimentPerUnitTemplate(dj.Computed):
    trained_model_table = None
    unit_table = None
    seed_table = ExperimentSeed
    experiment_method_table = ExperimentMethod
    StimulusSet_table = InsilicoStimuliSet

    definition = """
    # contains optimal stimuli
    -> self.experiment_method_table
    -> self.StimulusSet_table
    -> self.trained_model_table
    -> self.unit_table
    -> self.seed_table
    ---
    output           : longblob
    score            : float
    """

    model_loader_class = FabrikCache

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)
        super().__init__(*args, **kwargs)

    def get_method(self, key):
        method = self.experiment_method_table()

        method_config, method_fn = (method & key).fetch1('method_config', 'method_fn')
        method_config = method.parse_method_config(method_config)

        method_fn = method.import_func(method_fn)

        return method_config, method_fn

    def get_experiment_output(self, key, model, method_fn, method_config, stimulus_set):
        unit_index = (self.unit_table & key).fetch1('unit_index')
        seed = (self.seed_table & key).fetch1('seed')

        model.forward = partial(model.forward, data_key=key['data_key'])

        output, score = method_fn(
            stimulus_set,
            model,
            unit=unit_index,
            seed=seed,
            **method_config
        )

        stimuli_entity = dict(
            output=output,
            score=score,
            **key
        )

        return stimuli_entity

    def make(self, key: Key) -> None:
        stimulus_set = self.StimulusSet_table().load(key=key)
        method_config, method_fn = self.get_method(key=key)

        dataloaders, model = self.model_loader.load(key=key)
        model.cuda().eval()

        experiment_entity = self.get_experiment_output(key, model, method_fn, method_config, stimulus_set)

        self.insert1(experiment_entity, ignore_extra_fields=True)

