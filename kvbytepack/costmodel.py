"""When does a smaller KV record pay for its codec time?

A *measurement* is (raw bytes, stored bytes, encode seconds, decode seconds) for
one way of storing the same data. Bandwidths are bytes/second.

Two transfer models bracket real systems:

serial      encode, then send, then decode, one block at a time:
                t = encode + stored / bw + decode
pipelined   blocks stream through the three stages concurrently, so the slowest
            stage sets throughput (in raw bytes/second):
                goodput = min(raw / encode, raw / decode, bw * raw / stored)

Neither models queueing, contention with the model's own compute, or memory
bandwidth. They give a necessary condition, not a speedup claim.
"""
import math
from typing import NamedTuple


class Measurement(NamedTuple):
    name: str
    raw_bytes: int
    stored_bytes: int
    encode_s: float
    decode_s: float

    @property
    def ratio(self):
        return self.raw_bytes / self.stored_bytes if self.stored_bytes else math.inf

    @property
    def codec_s(self):
        return self.encode_s + self.decode_s


def uncompressed(raw_bytes):
    """Baseline: send the bytes as they are."""
    return Measurement("uncompressed", raw_bytes, raw_bytes, 0.0, 0.0)


def serial_seconds(m, bw):
    return m.encode_s + m.stored_bytes / bw + m.decode_s


def pipelined_goodput(m, bw):
    stages = [bw * m.raw_bytes / m.stored_bytes if m.stored_bytes else math.inf]
    if m.encode_s:
        stages.append(m.raw_bytes / m.encode_s)
    if m.decode_s:
        stages.append(m.raw_bytes / m.decode_s)
    return min(stages)


class Crossover(NamedTuple):
    """Serial model verdict for ``candidate`` against ``baseline``.

    ``wins`` is one of ``"always"``, ``"never"``, ``"below"`` (faster when the
    link is slower than ``bandwidth``) or ``"above"``.
    """
    wins: str
    bandwidth: float


def crossover(candidate, baseline):
    saved = baseline.stored_bytes - candidate.stored_bytes   # bytes the candidate avoids sending
    extra = candidate.codec_s - baseline.codec_s             # CPU seconds it adds
    if saved >= 0 and extra <= 0:
        return Crossover("never" if saved == 0 and extra == 0 else "always", math.nan)
    if saved <= 0 and extra >= 0:
        return Crossover("never", math.nan)
    return Crossover("below" if saved > 0 else "above", saved / extra)


def pipelined_ceiling(m):
    """Highest raw-byte goodput this codec can sustain, whatever the link speed."""
    return pipelined_goodput(m, math.inf)
