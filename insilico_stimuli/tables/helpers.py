from __future__ import annotations

from typing import Callable, Mapping, Dict, Any
from collections.abc import Iterable

from torch.nn import Module

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

def get_scores(self, dataloaders: Dataloaders, model: Module, key: Key) -> Dict[str, Any]:
    method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
    method_fn = self.import_func(method_fn)
    stimuli, score = method_fn(dataloaders, model, method_config)

    if isinstance(stimuli, Iterable):
        for stim, scr in zip(stimuli, score):
            yield dict(key, stimuli=stim, score=scr)
    else:
        yield dict(key, stimuli=stimuli, score=score)