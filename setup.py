"""Optional C kernel for the 2-byte channel-plane merge; NumPy is the portable fallback."""
from setuptools import Extension, setup

setup(ext_modules=[Extension("kvbytepack._native", ["kvbytepack/_native.c"], optional=True)])
