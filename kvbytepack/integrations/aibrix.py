"""AIBrix L2 marshaller that stores KV blocks as KVB2 records.

AIBrix chains marshallers by subclassing ``BaseMarshaller`` and overriding
``_marshal``/``_unmarshal`` on ``bytes``. Pass AIBrix's own class so this module
does not import or copy AIBrix::

    from aibrix_kvcache.l2.marshallers.marshaller import BaseMarshaller
    m = make_marshaller(BaseMarshaller, tokens=16, channels=1024, width=2)

The marshalled bytes must be exactly one ``[tokens, channels]`` block of
``width``-byte elements (no index header): that is what makes the channel
layout meaningful. Mismatched input is rejected rather than silently packed.
"""
from .. import record


def make_marshaller(base_cls, *, tokens, channels, width, inner=None,
                    layout="channel-planes", codec="zstd:1"):
    if any(type(v) is not int or v <= 0 for v in (tokens, channels, width)):
        raise ValueError("tokens, channels and width must be positive integers")
    shape = (tokens, channels)
    size = tokens * channels * width

    class KVBytePackMarshaller(base_cls):
        def _marshal(self, data):
            if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) != size:
                raise ValueError(f"expected {size} bytes for a {tokens}x{channels}x{width} block")
            return record.encode(data, shape, width, layout=layout, codec=codec)

        def _unmarshal(self, data):
            h, raw = record.decode(data)
            if h.shape != shape or h.width != width:
                raise ValueError(f"record geometry {h.shape}x{h.width} != {shape}x{width}")
            return raw

    return KVBytePackMarshaller(inner)
