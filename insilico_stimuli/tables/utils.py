from nnfabrik.utility.nnf_helper import dynamic_import, split_module_name


def import_module(path):
    return dynamic_import(*split_module_name(path))

def parse_config(config):
    """
    Parsing of the set config attributes to args and kwargs format, which is passable to the stimulus_fn
    expecting it to be of the format
    {
        parameter1: {
            path: path_to_type_function,
            args: list_of_arguments (optional),
            kwargs: dict_of_keyword_arguments (optional)
        },
        parameter2: {
            path: path_to_type_function,
            args: list_of_arguments (optional),
            kwargs: dict_of_keyword_arguments (optional)
        }
    }
    """

    for key, value in config.items():
        if not isinstance(value, dict):
            continue

        if 'path' not in value:
            if 'args' in value:
                config[key] = value['args']
            continue

        attr_fn = import_module(value['path'])

        if not 'args' in value:
            value['args'] = []
        if not 'kwargs' in value:
            value['kwargs'] = {}

        attr = attr_fn(*value['args'], **value['kwargs'])
        config[key] = attr

    return config