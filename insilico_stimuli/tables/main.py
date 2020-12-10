from nnfabrik.main import Dataset, schema
import datajoint as dj

from .mixins import InsilicoStimuliSetMixin, StimuliOptimizeMethodMixin, OptimisedStimuliTemplateMixin

@schema
class InsilicoStimuliSet(InsilicoStimuliSetMixin, dj.Lookup):
    """Table that contains Stimuli sets and their configurations."""

@schema
class StimuliOptimizeMethod(StimuliOptimizeMethodMixin, dj.Lookup):
    """Table that contains Stimuli sets and their configurations."""


class OptimisedStimuliTemplate(OptimisedStimuliTemplateMixin, dj.Computed):
    """Stimuli table template.
    To create a functional "Stimuli" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "StimuliMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    optimisation_method_table = StimuliOptimizeMethod
    stimuli_set_table = InsilicoStimuliSet