"""Compatibility alias; use kvbytepack.inputs."""
import sys
from kvbytepack import inputs as _module
sys.modules[__name__] = _module
