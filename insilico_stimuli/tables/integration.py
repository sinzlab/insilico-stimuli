from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.dj_helpers import make_hash

def import_module(path):
    return dynamic_import(*split_module_name(path))

class ModelLoader:
    def __init__(self, model_table, cache_size_limit=10):
        self.model_table = model_table
        self.cache_size_limit = cache_size_limit
        self.cache = dict()

    def load(self, key):
        if self.cache_size_limit == 0:
            return self._load_model(key)
        if not self._is_cached(key):
            self._cache_model(key)
        return self._get_cached_model(key)

    def _load_model(self, key):
        return self.model_table().load_model(key=key)

    def _is_cached(self, key):
        if self._hash_trained_model_key(key) in self.cache:
            return True
        return False

    def _cache_model(self, key):
        """Caches a model and makes sure the cache is not bigger than the specified limit."""
        self.cache[self._hash_trained_model_key(key)] = self._load_model(key)
        if len(self.cache) > self.cache_size_limit:
            del self.cache[list(self.cache)[0]]

    def _get_cached_model(self, key):
        return self.cache[self._hash_trained_model_key(key)]

    def _hash_trained_model_key(self, key):
        """Creates a hash from the part of the key corresponding to the primary key of the trained model table."""
        return make_hash({k: key[k] for k in self.model_table().primary_key})