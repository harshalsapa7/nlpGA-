"""Expert rankings and pruning. Step 3 ships the *baselines* only (random, frequency);
the redundancy-aware greedy search arrives in Step 4 and plugs into the same API.

Ranking convention: ``scores[layer][e]`` — **higher means more important** (kept first).
``ratio`` is always the fraction of *routed* experts removed in that layer.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from . import env as rgenv
from . import moe
from .stats import layer_index_from_name


def num_keep(num_experts: int, ratio: float) -> int:
    """How many experts survive a removal ratio. Never returns 0."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    return max(1, int(round(num_experts * (1.0 - ratio))))


def top_k_indices(scores: torch.Tensor, k: int) -> list[int]:
    """Indices of the ``k`` highest scores, returned in ascending expert order."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, scores.numel())
    order = torch.argsort(torch.as_tensor(scores).float(), descending=True, stable=True)
    return sorted(int(i) for i in order[:k])


# ------------------------------------------------------------------ score sources


def frequency_scores(per_layer_counts: dict[int, Sequence[int]]) -> dict[int, torch.Tensor]:
    """Utilization baseline: an expert's calibration load is its importance."""
    return {int(layer): torch.as_tensor(counts, dtype=torch.float32) for layer, counts in per_layer_counts.items()}


def random_scores(experts_per_layer: dict[int, int], seed: int = 0) -> dict[int, torch.Tensor]:
    """Random baseline: a fixed permutation per layer from an explicit seed."""
    generator = torch.Generator().manual_seed(int(seed))
    scores: dict[int, torch.Tensor] = {}
    for layer in sorted(experts_per_layer):
        n = int(experts_per_layer[layer])
        scores[layer] = torch.randperm(n, generator=generator).float()
    return scores


def experts_per_layer(model: torch.nn.Module) -> dict[int, int]:
    return {
        layer_index_from_name(name): int(block.num_experts)
        for name, block in rgenv.find_moe_blocks(model)
    }


# ------------------------------------------------------------------- keep sets


def keeps_from_scores(scores: dict[int, torch.Tensor], ratio: float | dict[int, float]) -> dict[int, list[int]]:
    """``ratio`` may be a single fraction or a per-layer mapping {layer: ratio}."""
    keeps: dict[int, list[int]] = {}
    for layer, layer_scores in scores.items():
        layer_ratio = ratio[layer] if isinstance(ratio, dict) else ratio
        n = int(layer_scores.numel())
        keeps[int(layer)] = top_k_indices(layer_scores, num_keep(n, layer_ratio))
    return keeps


def frequency_keeps(per_layer_counts: dict[int, Sequence[int]], ratio: float) -> dict[int, list[int]]:
    return keeps_from_scores(frequency_scores(per_layer_counts), ratio)


def random_keeps(experts: dict[int, int], ratio: float, seed: int = 0) -> dict[int, list[int]]:
    return keeps_from_scores(random_scores(experts, seed), ratio)


def keeps_for(
    model: torch.nn.Module,
    kind: str,
    ratio: float | dict[int, float],
    per_layer_counts: dict[int, Sequence[int]] | None = None,
    seed: int = 0,
    scores: dict[int, torch.Tensor] | None = None,
    sim: dict[int, Any] | None = None,
    utilization_weight: float = 0.25,
    max_steps: int | None = None,
) -> dict[int, list[int]]:
    """Dispatch for the rankings the protocol registers."""
    if scores is not None:
        return keeps_from_scores(scores, ratio)
    if kind == "frequency":
        if per_layer_counts is None:
            raise ValueError("frequency pruning needs per_layer_counts from calibration")
        return frequency_keeps(per_layer_counts, ratio)
    if kind == "random":
        return random_keeps(experts_per_layer(model), ratio, seed=seed)
    if kind == "redundancy":
        if sim is None:
            raise ValueError(
                "redundancy pruning needs similarity matrices; run similarity.build_similarity first"
            )
        keeps, info = redundancy_keeps(
            sim, ratio, per_layer_counts,
            utilization_weight=utilization_weight, max_steps=max_steps,
        )
        return keeps
    raise ValueError(f"unknown pruning kind {kind!r}")


# ------------------------------------------------------------------ redundancy


def greedy_redundant_keeps(
    sim: Any,
    ratio: float,
    utilization: Sequence[float] | None = None,
    utilization_weight: float = 0.25,
    max_steps: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Greedy removal by redundancy, recomputed after every removal.

    Primary term: how similar the expert is to the *surviving* experts (redundant ones
    go first). Secondary term: its balanced utilization, so a redundant-but-heavily
    used expert is protected. Scores are recomputed from the similarity submatrix each
    step, which is exact for this quantity and costs no model forward passes.
    """
    from .similarity import pairwise_mean_similarity

    matrix = torch.as_tensor(np.asarray(sim, dtype=np.float32))
    n = int(matrix.shape[0])
    target = num_keep(n, ratio)
    remaining = list(range(n))
    steps = 0
    truncated = False

    util = None
    if utilization is not None:
        util = torch.as_tensor(list(utilization), dtype=torch.float32)
        if util.numel() != n:
            raise ValueError(f"utilization has {util.numel()} entries, similarity has {n} experts")
        span = float(util.max() - util.min())
        util = (util - util.min()) / span if span > 0 else torch.ones_like(util)

    while len(remaining) > target:
        if max_steps is not None and steps >= int(max_steps):
            truncated = True
            break
        redundancy = pairwise_mean_similarity(matrix, remaining)
        span = float(redundancy.max() - redundancy.min())
        normalized = (redundancy - redundancy.min()) / span if span > 0 else torch.zeros_like(redundancy)
        if util is not None:
            keep_util = util[torch.as_tensor(remaining, dtype=torch.long)]
            priority = (1.0 - utilization_weight) * normalized + utilization_weight * (1.0 - keep_util)
        else:
            priority = normalized
        if not torch.isfinite(priority).all():
            raise RuntimeError("non-finite removal priority; similarity matrix is degenerate")
        drop = int(torch.argmax(priority))  # ties resolve to the lowest remaining expert
        remaining.pop(drop)
        steps += 1

    info = {
        "steps": steps,
        "target_keep": target,
        "kept": len(remaining),
        "truncated": truncated,
        "utilization_weight": float(utilization_weight),
    }
    return remaining, info


def redundancy_keeps(
    sim_per_layer: dict[int, Any],
    ratio: float | dict[int, float],
    per_layer_counts: dict[int, Sequence[int]] | None = None,
    utilization_weight: float = 0.25,
    max_steps: int | None = None,
) -> tuple[dict[int, list[int]], dict[int, dict[str, Any]]]:
    keeps: dict[int, list[int]] = {}
    infos: dict[int, dict[str, Any]] = {}
    for layer in sorted(sim_per_layer):
        layer_ratio = ratio[layer] if isinstance(ratio, dict) else ratio
        util = None
        if per_layer_counts is not None and layer in per_layer_counts:
            util = per_layer_counts[layer]
        keeps[int(layer)], infos[int(layer)] = greedy_redundant_keeps(
            sim_per_layer[layer], layer_ratio, util, utilization_weight, max_steps
        )
    return keeps, infos


# ------------------------------------------------------- layer-wise removal budgets


def layer_redundancy(sim: Any) -> float:
    """Mean off-diagonal cosine of one layer's similarity matrix.

    This is the per-layer quantity the layer-wise budget allocates by: a layer
    whose experts are all close to each other can lose more of them.
    """
    matrix = torch.as_tensor(np.asarray(sim, dtype=np.float32))
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool)
    return float(matrix[mask].mean())


def layer_wise_ratios(
    sim_per_layer: dict[int, Any],
    total_ratio: float,
    max_layer_ratio: float | None = None,
) -> dict[int, float]:
    """Allocate removals across layers in proportion to measured redundancy.

    Instead of a uniform ratio everywhere, the *total* removal budget
    (``round(total_ratio * total_experts)``) is distributed by largest-remainder
    with weights proportional to each layer's mean off-diagonal similarity (shifted
    so the least redundant layer still has a small positive weight; identical
    layers split the budget uniformly). Constraints, enforced in order:

    * every layer keeps at least one expert;
    * ``max_layer_ratio`` (optional) caps how much any single layer loses;
    * the total number of removals is conserved exactly — whatever a clamp frees
      up is re-allocated to layers that still have slack, in descending redundancy.

    Deterministic: ties break by higher redundancy, then lower layer index.
    """
    if not 0.0 <= total_ratio < 1.0:
        raise ValueError(f"total_ratio must be in [0, 1), got {total_ratio}")
    layers = sorted(sim_per_layer)
    sizes = {layer: int(np.asarray(sim_per_layer[layer]).shape[0]) for layer in layers}
    total_experts = sum(sizes.values())
    removals = {layer: 0 for layer in layers}
    if total_experts <= len(layers):
        return {layer: 0.0 for layer in layers}
    budget = min(int(round(total_ratio * total_experts)), total_experts - len(layers))
    if budget <= 0:
        return {layer: 0.0 for layer in layers}

    red = {layer: layer_redundancy(sim_per_layer[layer]) for layer in layers}
    lowest = min(red.values())
    spread = max(red.values()) - lowest
    # Shift so the least redundant layer keeps a small positive weight; without the
    # shift a single zero-redundancy layer would zero out the whole allocation.
    eps = 1e-6 * max(spread, 1.0)
    weights = {layer: red[layer] - lowest + eps for layer in layers}
    weight_sum = sum(weights.values())

    # Largest-remainder allocation for exact budget conservation.
    shares = {layer: budget * weights[layer] / weight_sum for layer in layers}
    for layer in layers:
        cap = sizes[layer] - 1
        if max_layer_ratio is not None:
            cap = min(cap, int(max_layer_ratio * sizes[layer]))
        removals[layer] = min(int(shares[layer]), cap)
    remainder = budget - sum(removals.values())
    order = sorted(layers, key=lambda layer: (-red[layer], layer))
    while remainder > 0:
        progressed = False
        for layer in order:
            if remainder <= 0:
                break
            cap = sizes[layer] - 1
            if max_layer_ratio is not None:
                cap = min(cap, int(max_layer_ratio * sizes[layer]))
            if removals[layer] < cap:
                removals[layer] += 1
                remainder -= 1
                progressed = True
        if not progressed:  # every layer is at its cap; budget was infeasible
            break
    return {layer: removals[layer] / sizes[layer] for layer in layers}


# --------------------------------------------------------------------- apply


def prune_with_keeps(model: torch.nn.Module, keeps: dict[int, list[int]]) -> dict[str, Any]:
    """Real removal: drop experts and slice routers. Returns the removal report."""
    report = moe.apply_removal(model, keeps)
    report["kept_per_layer"] = {int(k): len(v) for k, v in keeps.items()}
    return report


def prune_by_kind(
    model: torch.nn.Module,
    kind: str,
    ratio: float,
    per_layer_counts: dict[int, Sequence[int]] | None = None,
    seed: int = 0,
    scores: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    keeps = keeps_for(model, kind, ratio, per_layer_counts=per_layer_counts, seed=seed, scores=scores)
    return keeps, prune_with_keeps(model, keeps)


def removal_counts(experts: dict[int, int], ratio: float) -> dict[int, tuple[int, int]]:
    """``{layer: (n_before, n_after)}`` for a ratio, without touching a model."""
    return {layer: (n, num_keep(n, ratio)) for layer, n in sorted(experts.items())}
