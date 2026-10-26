"""Compatibility alias; use kvbytepack.tensor."""
import sys
from kvbytepack import tensor as _module
sys.modules[__name__] = _module
