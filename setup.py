"""Optional CPU byte-layout kernel; NumPy remains the portable fallback."""
from setuptools import Extension, setup

setup(ext_modules=[Extension("kvbytepack._decode", ["kvbytepack/_decode.c"], optional=True)])
