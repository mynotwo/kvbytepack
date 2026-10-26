"""Compatibility alias; use kvbytepack.formats."""
import sys
from kvbytepack import formats as _module
sys.modules[__name__] = _module
