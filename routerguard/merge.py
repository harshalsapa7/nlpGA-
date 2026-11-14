"""Merge redundant expert pairs instead of deleting them.

Why merging
-----------
The similarity matrix already identifies redundant *pairs*. Deletion throws the
pair's combined capacity away; averaging the two experts' weights keeps a coarse
version of it: if ``f_i`` and ``f_j`` are similar, ``f_merged = (f_i + f_j) / 2`` is
close to both, so routing to the merged expert loses less than dropping one entirely.

Semantics, stated exactly
-------------------------
1. Pair selection mirrors the greedy pruner in ``prune.greedy_redundant_keeps``:
   primary term is similarity (most redundant pair first), secondary term is the
   pair's balanced utilization, weighted by the same ``utilization_weight`` knob
   (``protocol.yaml: design.utilization_weight``). Each merge removes exactly one
   expert, so any target count is reachable.
2. Survivor within a pair: the higher-utilization expert (ties -> lower index).
3. Weights: every parameter of the survivor becomes ``(w_dst + w_src) / 2``, and the
   survivor's gate row is averaged the same way. The removed slot is then sliced out
   with ``moe.prune_block_inplace``, so the merged model is the *native* pruned one
   (softmax over kept experts, ``top_k`` clamped) — same semantics as deletion.
4. The reduced similarity matrix approximates the merged expert's row as the mean of
   the pair's rows. This is an approximation (cosine of an average != average of
   cosines), stated here and never claimed as exact.

What merging does NOT guarantee
-------------------------------
* **No mask-simulation equivalence.** Merging changes weights, so ``moe.MaskSim``
  cannot preview it; its perplexity must be measured on real merged weights.
* **Unquantized experts only.** bitsandbytes stores 4/8-bit experts as packed
  integers; averaging packed weights is meaningless. This module refuses any expert
  whose parameters are not floating point (or carry a ``quant_state``), so the merge
  arm requires a bf16/fp16 load (OLMoE bf16 on an A100/L4-class card, or CPU) and
  falls back to deletion on 8/4-bit runs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from . import env as rgenv
from . import moe
from .prune import num_keep
from .stats import layer_index_from_name


def _normalize_util(utilization: Sequence[float] | None, n: int) -> torch.Tensor | None:
    """Same min-max normalization the greedy pruner uses; None -> None."""
    if utilization is None:
        return None
    util = torch.as_tensor(list(utilization), dtype=torch.float32)
    if util.numel() != n:
        raise ValueError(f"utilization has {util.numel()} entries, similarity has {n} experts")
    span = float(util.max() - util.min())
    return (util - util.min()) / span if span > 0 else torch.ones_like(util)


def greedy_merge_plan(
    sim: Any,
    ratio: float,
    utilization: Sequence[float] | None = None,
    utilization_weight: float = 0.25,
    max_steps: int | None = None,
) -> tuple[list[tuple[int, int]], list[int], dict[str, Any]]:
    """Plan which experts merge, without touching any weights.

    Returns ``(pairs, survivors, info)`` where ``pairs`` is a list of
    ``(dst, src)`` merges in original expert indices and ``survivors`` is the
    ascending list of original indices that remain after applying all merges.
    """
    matrix = torch.as_tensor(np.asarray(sim, dtype=np.float32)).clone()
    n = int(matrix.shape[0])
    target = num_keep(n, ratio)
    survivors = list(range(n))
    pairs: list[tuple[int, int]] = []
    steps = 0
    truncated = False

    util = _normalize_util(utilization, n)
    weight = float(utilization_weight)

    while len(survivors) > target:
        if max_steps is not None and steps >= int(max_steps):
            truncated = True
            break
        idx = torch.as_tensor(survivors, dtype=torch.long)
        sub = matrix.index_select(0, idx).index_select(1, idx)
        m = sub.shape[0]
        if m < 2:
            break
        # Off-diagonal similarity of every ordered pair (a < b), min-max normalized
        # over those entries, exactly like the pruner normalizes its redundancy term.
        mask = torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)
        vals = sub[mask]
        span = float(vals.max() - vals.min()) if vals.numel() else 0.0
        norm = (sub - vals.min()) / span if span > 0 else torch.zeros_like(sub)
        if util is not None:
            keep_util = util[idx]
            pair_util = (keep_util[:, None] + keep_util[None, :]) / 2.0
            priority = (1.0 - weight) * norm + weight * (1.0 - pair_util)
        else:
            priority = norm
        priority = priority.masked_fill(~mask, float("-inf"))
        flat = int(torch.argmax(priority.flatten()))
        a, b = flat // m, flat % m
        dst_local, src_local = (a, b) if a < b else (b, a)
        dst, src = survivors[dst_local], survivors[src_local]
        if util is not None and util[src] > util[dst]:
            dst, src = src, dst  # the more-used expert keeps the slot
        pairs.append((dst, src))
        # Merged expert's row ≈ mean of the pair's rows (documented approximation).
        row = (matrix[dst] + matrix[src]) / 2.0
        matrix[dst] = row
        matrix[:, dst] = row
        survivors.remove(src)
        steps += 1

    info = {
        "steps": steps,
        "merges": len(pairs),
        "target_keep": target,
        "survivors": len(survivors),
        "truncated": truncated,
        "utilization_weight": float(utilization_weight),
    }
    return pairs, survivors, info


def merge_keeps(
    sim_per_layer: dict[int, Any],
    ratio: float | dict[int, float],
    per_layer_counts: dict[int, Sequence[int]] | None = None,
    utilization_weight: float = 0.25,
    max_steps: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Merge plans per layer, shaped like ``prune.redundancy_keeps``.

    ``ratio`` may be a single fraction or a per-layer mapping (layer-wise budget).
    """
    plans: dict[int, dict[str, Any]] = {}
    for layer in sorted(sim_per_layer):
        layer_ratio = ratio[layer] if isinstance(ratio, dict) else ratio
        util = None
        if per_layer_counts is not None and layer in per_layer_counts:
            util = per_layer_counts[layer]
        pairs, survivors, info = greedy_merge_plan(
            sim_per_layer[layer], layer_ratio, util, utilization_weight, max_steps
        )
        plans[int(layer)] = {"pairs": pairs, "survivors": survivors, "info": info}
    return plans


def _assert_mergable(block) -> None:
    """Refuse quantized (bitsandbytes) or integer expert weights up front."""
    for name, param in block.experts[0].named_parameters():
        if hasattr(param, "quant_state"):
            raise RuntimeError(
                f"expert parameter {name} is bitsandbytes-quantized; merging requires "
                "unquantized experts. Load the checkpoint in bf16 (or run on CPU) or "
                "use deletion instead — see merge.py module docstring."
            )
        if not param.is_floating_point():
            raise RuntimeError(
                f"expert parameter {name} has non-floating dtype {param.dtype}; "
                "merging requires unquantized experts"
            )


def merge_experts_inplace(block, pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Average each ``(dst, src)`` pair of experts and their gate rows, in place."""
    if not pairs:
        return {"merged": 0}
    _assert_mergable(block)

    named = [dict(expert.named_parameters()) for expert in block.experts]
    gate_weight = block.gate.weight.detach()
    gate_bias = block.gate.bias.detach() if block.gate.bias is not None else None

    with torch.no_grad():
        for dst, src in pairs:
            dst_params, src_params = named[dst], named[src]
            if set(dst_params) != set(src_params):
                raise RuntimeError(
                    f"experts {dst} and {src} have different parameter sets; cannot merge"
                )
            for key in dst_params:
                dst_p, src_p = dst_params[key], src_params[key]
                averaged = (dst_p.detach().float() + src_p.detach().float()) / 2.0
                dst_p.copy_(averaged.to(dst_p.dtype))
            row = (gate_weight[dst].float() + gate_weight[src].float()) / 2.0
            gate_weight[dst].copy_(row.to(gate_weight.dtype))
            if gate_bias is not None:
                bias_row = (gate_bias[dst].float() + gate_bias[src].float()) / 2.0
                gate_bias[dst].copy_(bias_row.to(gate_bias.dtype))

    return {"merged": len(pairs)}


def apply_merge(
    model: torch.nn.Module,
    plans: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Merge every layer's pairs, then slice the survivors out natively.

    After this call the model is a *native* pruned model with averaged weights:
    fewer experts, sliced gate, ``top_k`` clamped — the same shape deletion leaves,
    so every downstream tool (evaluation, refit, further pruning) works unchanged.
    """
    blocks = rgenv.find_moe_blocks(model)
    by_layer = {layer_index_from_name(name): block for name, block in blocks}
    report: dict[str, Any] = {"layers": {}, "survivors": {}}

    for layer, block in sorted(by_layer.items()):
        plan = plans.get(layer)
        if plan is None:
            continue
        merge_info = merge_experts_inplace(block, plan["pairs"])
        removal = moe.prune_block_inplace(block, plan["survivors"])
        report["layers"][layer] = {**merge_info, **removal}
        report["survivors"][layer] = [int(i) for i in plan["survivors"]]

    counts = {info["num_experts_after"] for info in report["layers"].values()}
    if counts and len(counts) == 1:
        n = counts.pop()
        if hasattr(model.config, "num_experts"):
            model.config.num_experts = n
    report["num_experts"] = sorted({info["num_experts_after"] for info in report["layers"].values()})
    report["merged_total"] = sum(info.get("merged", 0) for info in report["layers"].values())
    report["removed_total"] = sum(len(info["removed"]) for info in report["layers"].values())
    return report
