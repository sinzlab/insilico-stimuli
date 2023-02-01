import datajoint as dj
import numpy as np

import warnings

from typing import Callable, Mapping, Dict, Any

from torch.utils.data import DataLoader

from nnfabrik.main import Dataset, schema
from nnfabrik.utility.dj_helpers import make_hash

from .utils import parse_config, import_module

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]


@schema
class InsilicoStimuliSet(dj.Lookup):
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
    fetch1: Callable

    import_func = staticmethod(import_module)
    parse_stimulus_config = staticmethod(parse_config)

    def add_set(self, stimulus_fn: str, stimulus_config: Mapping, comment: str = "",
                skip_duplicates: bool = False) -> None:
        key = dict(
            stimulus_fn=stimulus_fn,
            stimulus_hash=make_hash(stimulus_config),
            stimulus_config=stimulus_config,
            stimulus_comment=comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key


    def load(self, key: Key) -> np.ndarray:
        """
        Returns the stimuli set instance.
        """
        stimulus_fn, stimulus_config = (self & key).fetch1("stimulus_fn", "stimulus_config")
        stimulus_fn = self.import_func(stimulus_fn)

        stimulus_config = self.parse_stimulus_config(stimulus_config)

        stimulus = stimulus_fn(**stimulus_config)

        return stimulus


@schema
class ExperimentMethod(dj.Lookup):
    """Table that contains experiment methods and their configurations."""
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
    fetch1: Callable

    import_func = staticmethod(import_module)
    parse_method_config = staticmethod(parse_config)

    def add_method(self, method_fn: str, method_config: Mapping, comment: str = "",
                   skip_duplicates: bool = False) -> None:
        key = dict(
            method_fn=method_fn,
            method_hash=make_hash(method_config),
            method_config=method_config,
            method_comment=comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key


@schema
class ExperimentSeed(dj.Lookup):
    definition = """
        # contains seeds used to make the experiments reproducible
        seed                                : int   # experiment seed
        """
