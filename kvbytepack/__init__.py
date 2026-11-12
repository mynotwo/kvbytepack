"""Bit-exact KV-cache compression, plus the cost model that says when it pays."""
from .bench import measure, report
from .costmodel import Measurement, crossover, pipelined_goodput, serial_seconds
from .record import read_header
from .tensor import pack, unpack

__version__ = "0.3.0"
__all__ = ["pack", "unpack", "read_header", "measure", "report", "Measurement",
           "crossover", "pipelined_goodput", "serial_seconds"]
