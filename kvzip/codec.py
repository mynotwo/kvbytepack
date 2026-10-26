"""Compatibility alias; use kvbytepack.codec."""
import sys
from kvbytepack import codec as _module
sys.modules[__name__] = _module
