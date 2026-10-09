"""Capture KV cache and activations (per-layer hidden states) from a HF causal LM.

Every captured tensor is returned as float32 [S, D] (tokens x channels):
  KV:          per layer, heads are flattened into channels (D = n_kv_heads * head_dim)
  activation:  per layer residual-stream hidden state (D = hidden_size), index i
               = output of block i+1. The embedding output is skipped: it is
               recomputable from token ids, and with repeated input text its rows
               repeat exactly, which inflates the ratio (~28x at 2K tokens).
"""
import torch


def build_inputs(tokenizer, text, target_len):
    repeated = text * (target_len // max(len(text.split()), 1) + 1)
    return tokenizer(repeated, return_tensors="pt", max_length=target_len, truncation=True)


def _iter_kv(pkv):
    # transformers >= 4.56 DynamicCache exposes .layers; older versions .key_cache;
    # legacy caches are tuples of (k, v).
    if hasattr(pkv, "layers"):
        for layer in pkv.layers:
            yield layer.keys, layer.values
    elif hasattr(pkv, "key_cache"):
        yield from zip(pkv.key_cache, pkv.value_cache)
    else:
        for layer in pkv:
            yield layer[0], layer[1]


def _to_sd(t):
    if t.dim() == 4:  # [B, H, S, Dh] -> [S, H*Dh]
        _, H, S, Dh = t.shape
        return t[0].permute(1, 0, 2).reshape(S, H * Dh)
    return t.reshape(-1, t.shape[-1])


@torch.no_grad()
def capture(model, tokenizer, text, target_len, want_kv=True, want_act=True):
    """Run one forward pass. Returns dict with optional 'K', 'V', 'act' lists and 'tokens'."""
    inputs = build_inputs(tokenizer, text, target_len).to(model.device)
    out = model(**inputs, use_cache=want_kv, output_hidden_states=want_act)
    res = {"tokens": inputs["input_ids"].shape[1]}
    if want_kv:
        res["K"], res["V"] = [], []
        for k, v in _iter_kv(out.past_key_values):
            res["K"].append(_to_sd(k).float().cpu())
            res["V"].append(_to_sd(v).float().cpu())
    if want_act:
        res["act"] = [_to_sd(h).float().cpu() for h in out.hidden_states[1:]]
    del out
    return res
