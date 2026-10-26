"""Compatibility alias; use kvbytepack.capture."""
import sys
from kvbytepack import capture as _module
sys.modules[__name__] = _module
