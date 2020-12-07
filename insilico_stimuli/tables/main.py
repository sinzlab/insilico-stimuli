from nnfabrik.main import Dataset, schema
import datajoint as dj

from .ComputedStimuli import StimuliMethodMixin, ComputedStimuliTemplateMixin

@schema
class StimuliMethod(StimuliMethodMixin, dj.Lookup):
    """Table that contains Stimuli methods and their configurations."""


class ComputedStimuliTemplate(ComputedStimuliTemplateMixin, dj.Computed):
    """Stimuli table template.
    To create a functional "Stimuli" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "StimuliMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    method_table = StimuliMethod