"""General-purpose byte codecs, addressed by a stable one-byte id in records.

A codec spec is ``"<name>"`` or ``"<name>:<level>"``, e.g. ``"zstd:1"``,
``"zstd:-3"``, ``"lz4"``, ``"store"``. Ids are part of the record format: never
renumber them.
"""
import lzma
import threading
import zlib

import zstandard as _zstd

try:
    import lz4.frame as _lz4
except ImportError:
    _lz4 = None

_local = threading.local()


def _zstd_c(level):
    cache = _local.__dict__.setdefault("zc", {})
    c = cache.get(level)
    if c is None:
        c = cache[level] = _zstd.ZstdCompressor(level=level)
    return c


def _zstd_d():
    d = getattr(_local, "zd", None)
    if d is None:
        d = _local.zd = _zstd.ZstdDecompressor()
    return d


class Codec:
    __slots__ = ("name", "id", "default_level", "_compress", "_decompress")

    def __init__(self, name, cid, default_level, compress, decompress):
        self.name, self.id, self.default_level = name, cid, default_level
        self._compress, self._decompress = compress, decompress

    def compress(self, data, level):
        return self._compress(data, level)

    def decompress(self, data, size):
        out = self._decompress(data, size)
        if len(out) != size:
            raise ValueError(f"{self.name}: decoded {len(out)} bytes, expected {size}")
        return out


def _zstd_decompress(data, size):
    try:
        return _zstd_d().decompress(data, max_output_size=size, allow_extra_data=False)
    except _zstd.ZstdError as exc:
        raise ValueError(f"zstd: corrupt plane ({exc})") from exc


def _zlib_decompress(data, size):
    d = zlib.decompressobj()
    out = d.decompress(data, size)
    if d.unconsumed_tail or not d.eof:
        raise ValueError("zlib: corrupt or oversized plane")
    return out


_REGISTRY = {}


def _register(codec):
    _REGISTRY[codec.name] = codec


_register(Codec("store", 0, None, lambda d, _: bytes(d), lambda d, _: bytes(d)))
_register(Codec("zstd", 1, 1, lambda d, lv: _zstd_c(lv).compress(d), _zstd_decompress))
_register(Codec("zlib", 2, 6, lambda d, lv: zlib.compress(d, lv), _zlib_decompress))
_register(Codec("lzma", 3, 6, lambda d, lv: lzma.compress(d, preset=lv),
                lambda d, _: lzma.decompress(d)))
if _lz4 is not None:
    _register(Codec("lz4", 4, 0,
                    lambda d, lv: _lz4.compress(d, compression_level=lv, store_size=False),
                    lambda d, _: _lz4.decompress(d)))

_BY_ID = {c.id: c for c in _REGISTRY.values()}


def available():
    return sorted(_REGISTRY)


def parse(spec):
    """``"zstd:3"`` -> (Codec, 3). Missing level uses the codec default."""
    name, _, lv = spec.partition(":")
    if name not in _REGISTRY:
        hint = " (pip install lz4)" if name == "lz4" else ""
        raise ValueError(f"codec not available: {name!r}{hint}; have {available()}")
    codec = _REGISTRY[name]
    if codec.default_level is None:
        if lv:
            raise ValueError(f"codec {name!r} takes no level")
        return codec, 0
    level = int(lv) if lv else codec.default_level
    if not -128 <= level <= 127:
        raise ValueError("codec level must fit in a signed byte")
    return codec, level


def by_id(cid):
    try:
        return _BY_ID[cid]
    except KeyError:
        raise ValueError(f"record uses codec id {cid}, not available here") from None


def spec(codec, level):
    return codec.name if codec.default_level is None else f"{codec.name}:{level}"
