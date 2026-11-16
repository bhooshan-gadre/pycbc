from .matchedfilter import *
from .resample import *


def __getattr__(name):
    # Importing the package must not import JAX. The live filter pulls this
    # class in only on ranks that select the JAX backend.
    if name == 'JaxLiveBatchMatchedFilter':
        from .jax_matchedfilter import JaxLiveBatchMatchedFilter
        return JaxLiveBatchMatchedFilter
    raise AttributeError('module %r has no attribute %r' % (__name__, name))
