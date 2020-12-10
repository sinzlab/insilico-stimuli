"""This module the table templates."""

from __future__ import annotations

import datajoint as dj
import numpy as np

from typing import Callable, Mapping, Dict, Any

from torch.utils.data import DataLoader

from nnfabrik.utility.nnf_helper import FabrikCache

from .main import StimuliOptimizeMethod, InsilicoStimuliSet

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

class OptimisedStimuliTemplate(dj.Computed):
    definition = """
    # contains optimal stimuli
    -> self.optimisation_method_table
    -> self.StimulusSet_table
    -> self.trained_model_table
    ---
    average_score = 0.      : float        # average score depending on the used method function
    """

    trained_model_table = None
    unit_table = None
    optimisation_method_table = StimuliOptimizeMethod
    StimulusSet_table = InsilicoStimuliSet

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

        stimuli_entities = []
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

                stimuli_entities.append(stimuli_entity)

        scores = [stimuli_entity['score'] for stimuli_entity in stimuli_entities]
        average_score = np.mean(scores)

        key['average_score'] = average_score

        self.insert1(key, ignore_extra_fields=True)
        self.Units.insert(stimuli_entities, ignore_extra_fields=True)
