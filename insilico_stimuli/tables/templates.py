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

    @property
    def prev_primary_keys(self):
        prev_primary_keys = None
        if self.previous_experiment_table:
            prev_primary_keys = self.previous_experiment_table.primary_key
        return prev_primary_keys

    @property
    def definition(self):
        if self.previous_experiment_table:
            projection = '.proj(' + ', '.join([f'prev_{key}="{key}"' for key in self.prev_primary_keys]) + ')'

        definition = """
        # contains optimal stimuli
        -> self.experiment_method_table
        -> self.StimulusSet_table
        -> self.trained_model_table
        {}
        ---
        average_score = 0.:                float        # average score depending on the used method function
        """.format('-> self.previous_experiment_table' + projection if self.previous_experiment_table else '')

        return definition

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

    def get_stimulus_set(self, key):
        StimulusSet = self.StimulusSet_table()

        stimulus_config, stimulus_fn = (StimulusSet & key).fetch1('stimulus_config', 'stimulus_fn')
        stimulus_config = StimulusSet.parse_stimulus_config(stimulus_config)

        stimulus_fn = StimulusSet.import_func(stimulus_fn)

        return stimulus_config, stimulus_fn

    def get_method(self, key):
        method = self.experiment_method_table()

        method_config, method_fn = (method & key).fetch1('method_config', 'method_fn')
        method_config = method.parse_method_config(method_config)

        method_fn = method.import_func(method_fn)

        return method_config, method_fn

    def get_experiment_output(self, key,
                              dataloaders, model,
                              method_fn, method_config,
                              stimulus_fn, stimulus_config):
        data_keys = list(dataloaders['test'].keys())

        stimuli_entities = []
        for data_key in data_keys:
            if self.previous_experiment_table:
                prev_key = {prev_key.strip('prev_'): key[prev_key] for prev_key in self.prev_primary_keys}

                previous_experiment = (
                            (self.previous_experiment_table & prev_key).Units() & dict(data_key=data_key)).fetch(as_dict=True)

                outputs, scores = method_fn(
                    stimulus_fn(**stimulus_config),
                    partial(model, data_key=data_key),
                    previous_experiment=previous_experiment,
                    **method_config
                )
            else:
                outputs, scores = method_fn(
                    stimulus_fn(**stimulus_config),
                    partial(model, data_key=data_key),
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
        stimulus_config, stimulus_fn = self.get_stimulus_set(key)
        method_config, method_fn = self.get_method(key)

        dataloaders, model = self.model_loader.load(key=key)
        model.cuda().eval()

        experiment_entities = self.get_experiment_output(key,
                                                         dataloaders, model,
                                                         method_fn, method_config,
                                                         stimulus_fn, stimulus_config)

        key['average_score'] = self.compute_average_score(experiment_entities)

        self.insert1(key, ignore_extra_fields=True)
        self.Units.insert(experiment_entities, ignore_extra_fields=True)

class ExperimentPerUnitTemplate(dj.Computed):
    trained_model_table = None
    unit_table = None
    previous_experiment_table = None
    seed_table = ExperimentSeed
    experiment_method_table = ExperimentMethod
    StimulusSet_table = InsilicoStimuliSet

    @property
    def prev_primary_keys(self):
        prev_primary_keys = None
        if self.previous_experiment_table:
            prev_primary_keys = self.previous_experiment_table.primary_key
        return prev_primary_keys

    @property
    def definition(self):
        if self.previous_experiment_table:
            projection = '.proj(' + ', '.join([f'prev_{key}="{key}"' for key in self.prev_primary_keys]) + ')'

        definition = """
        # contains optimal stimuli
        -> self.experiment_method_table
        -> self.StimulusSet_table
        -> self.trained_model_table
        -> self.unit_table
        -> self.seed_table
        {}
        ---
        output           : longblob
        score            : float
        """.format('-> self.previous_experiment_table' + projection if self.previous_experiment_table else '')

        return definition

    model_loader_class = FabrikCache

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)
        super().__init__(*args, **kwargs)

    def get_stimulus_set(self, key):
        StimulusSet = self.StimulusSet_table()

        stimulus_config, stimulus_fn = (StimulusSet & key).fetch1('stimulus_config', 'stimulus_fn')
        stimulus_config = StimulusSet.parse_stimulus_config(stimulus_config)

        stimulus_fn = StimulusSet.import_func(stimulus_fn)

        return stimulus_config, stimulus_fn

    def get_method(self, key):
        method = self.experiment_method_table()

        method_config, method_fn = (method & key).fetch1('method_config', 'method_fn')
        method_config = method.parse_method_config(method_config)

        method_fn = method.import_func(method_fn)

        return method_config, method_fn

    def get_experiment_output(self, key, model,
                              method_fn, method_config,
                              stimulus_fn, stimulus_config):
        unit_index = (self.unit_table & key).fetch1('unit_index')
        seed = (self.seed_table & key).fetch1('seed')

        if self.previous_experiment_table:
            prev_key = {prev_key.strip('prev_'): key[prev_key] for prev_key in self.prev_primary_keys}

            previous_experiment = (
                        (self.previous_experiment_table & prev_key).Units() & dict(data_key=key['data_key'])).fetch(as_dict=True)

            output, score = method_fn(
                stimulus_fn(**stimulus_config),
                partial(model, data_key=key['data_key']),
                previous_experiment=previous_experiment,
                unit=unit_index,
                seed=seed,
                **method_config
            )
        else:
            output, score = method_fn(
                stimulus_fn(**stimulus_config),
                partial(model, data_key=key['data_key']),
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
        stimulus_config, stimulus_fn = self.get_stimulus_set(key)
        method_config, method_fn = self.get_method(key)

        dataloaders, model = self.model_loader.load(key=key)
        model.cuda().eval()

        experiment_entity = self.get_experiment_output(key, model,
                                                         method_fn, method_config,
                                                         stimulus_fn, stimulus_config)

        self.insert1(experiment_entity, ignore_extra_fields=True)
