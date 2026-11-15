"""One-process pipeline for Steps 1-4 (plus the extension arms), runbook-accurate.

Why this exists
---------------
The notebooks repeat model loading in every file, which wastes several minutes of a
Colab session per stage and makes runs hard to diff. This module executes the same
call sequence as ``notebooks/01 -> 02 -> 03`` — verified cell-by-cell against
``tools/preflight_step3.py`` / ``preflight_step4.py`` — in a single process, with the
model loaded once, and writes everything into one ``results/<run_id>/`` directory.

Sequence (matching the notebooks, plus the opt-in extension arms):

1.  protocol -> Run bookkeeping -> model load -> env dump
2.  splits (wikitext calib/train + eval/test, cached)
3.  load statistics on both splits; ``gini_skip_balance`` decision gate
4.  balance: baseline -> fit biases -> guards (``max_kl_nats``, ``min_balance_transfer_ratio``)
5.  similarity (cached, resumable); recompute under the fitted bias iff mean
    routing KL >= 1e-2 (layer 0 is router-invariant by construction)
6.  2xN ablation by MaskSim: {unbalanced, balanced} x {random, frequency, redundancy}
    x ratios (+ the layer-wise-budget redundancy arm), evaluated as perplexity
7.  ``min_gain_over_frequency_pts`` gate on the balanced@max-ratio cell
8.  materialize the headline arm (real removal) -> full evaluation bundle
9.  extension: refit the biases on the pruned model (protocol v2, opt-in)
10. extension: merge instead of remove, on a freshly reloaded model (opt-in)

Testability: ``PipelineSpec.loader`` and ``PipelineSpec.splits`` are injection
points, so the whole pipeline runs on CPU against tiny models with no downloads.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from . import balance as rgbalance
from . import data as rgdata
from . import env as rgenv
from . import eval as rgeval
from . import merge as rgmerge
from . import models as rgmodels
from . import moe
from . import prune
from . import run as rgrun
from . import similarity as rgsim
from . import stats as rgstats


Loader = Callable[[], tuple[Any, Any, Any]]


@dataclass
class PipelineSpec:
    """Everything one pipeline run needs; CLI flags map 1:1 onto these fields."""

    model: str = "olmoe"                 # olmoe | qwen
    model_id: str | None = None
    precision: str = "auto"              # auto | bf16 | 8bit | 4bit (olmoe only for auto/8bit)
    device: str = "cuda"
    tag: str = "pipeline"
    ratios: Sequence[float] | None = None        # None = protocol pruning.ratios
    seeds: Sequence[int] | None = None           # random-baseline seeds; None = protocol
    kinds: Sequence[str] = ("random", "frequency", "redundancy")
    update_mode: str | None = None               # None = protocol balance.update_mode
    with_refit: bool = True              # protocol v2 arm: refit biases after pruning
    with_merge: bool = False             # protocol v2 arm: merge instead of remove
    layer_wise_budget: bool = False      # allocate redundancy removals by layer redundancy
    do_mmlu: bool = True
    do_throughput: bool = True
    stats_batch_size: int = 4
    ppl_batch_size: int = 1
    mmlu_batch_size: int = 8
    similarity_batch_size: int = 4
    max_calib_tokens: int | None = None  # caps for smoke/fast runs (None = full protocol)
    max_eval_tokens: int | None = None
    max_tokens_per_layer: int | None = None      # None = protocol budget
    overrides: dict[str, Any] = field(default_factory=dict)      # protocol overrides
    results_root: str | Any = None
    loader: Loader | None = None         # injection point for tests
    splits: dict[str, rgdata.TokenSplit] | None = None           # injection point for tests
    protocol: dict[str, Any] | None = None                       # injection point for tests
    progress: bool = True

    def resolved_ratios(self, protocol: dict[str, Any]) -> list[float]:
        return [float(r) for r in (self.ratios if self.ratios is not None else protocol["pruning"]["ratios"])]

    def resolved_seeds(self, protocol: dict[str, Any]) -> list[int]:
        return [int(s) for s in (self.seeds if self.seeds is not None else protocol["pruning"]["seeds"])]


def default_loader(spec: PipelineSpec) -> Loader:
    """Load OLMoE or Qwen through ``models.py`` with the spec's precision/device."""
    def loader() -> tuple[Any, Any, Any]:
        if spec.model == "olmoe":
            model_id = spec.model_id or rgmodels.OLMOE_ID
            return rgmodels.load_olmoe(
                model_id, precision=spec.precision, device=spec.device
            )
        if spec.model == "qwen":
            model_id = spec.model_id or rgmodels.QWEN_ID
            return rgmodels.load_qwen(model_id, device=spec.device)
        raise ValueError(f"unknown model {spec.model!r} (expected 'olmoe' or 'qwen')")
    return loader


def _assert_env(spec: PipelineSpec) -> dict[str, Any]:
    """Version pins always; CUDA requirement only when a GPU run was asked for."""
    strict = spec.device.startswith("cuda") and torch.cuda.is_available()
    return rgenv.assert_env(strict=strict)


def _build_splits(spec: PipelineSpec, tokenizer: Any, protocol: dict[str, Any], run: rgrun.Run):
    if spec.splits is not None:
        return spec.splits
    return rgdata.build_splits(
        tokenizer, protocol, cache_dir=run.dir / "splits",
    )


def _maybe_mmlu_prompts(spec: PipelineSpec, protocol: dict[str, Any]) -> tuple[Any, list[str]]:
    """Build the protocol's MMLU subset; a failure degrades to ppl-only, loudly."""
    if not spec.do_mmlu:
        return None, ["mmlu disabled by flag"]
    try:
        cfg = protocol["metrics"]["mmlu"]
        prompts = rgdata.build_mmlu_prompts(
            n_questions=int(cfg.get("n_questions", 500)),
            n_shot=int(cfg.get("n_shot", 3)),
            seed=int(cfg.get("seed", 42)),
        )
        return prompts, []
    except Exception as exc:  # noqa: BLE001 - missing dataset must not kill the run
        return None, [f"mmlu prompts unavailable ({type(exc).__name__}: {exc}); proceeding ppl-only"]


def _resolve_protocol(spec: PipelineSpec) -> dict[str, Any]:
    return spec.protocol if spec.protocol is not None else rgrun.load_protocol(
        overrides=spec.overrides
    )


def _bootstrap(spec: PipelineSpec, run: rgrun.Run, protocol: dict[str, Any]):
    """Env assert + model load + splits: the shared prologue of every stage."""
    _assert_env(spec)
    run.record_env()
    loader = spec.loader or default_loader(spec)
    model, tokenizer, info = loader()
    precision = getattr(info, "precision", "bf16")
    info_dict = info.to_dict() if hasattr(info, "to_dict") else {"repr": str(info)}
    run.save_json("load", info_dict)
    if spec.progress:
        print(f"[pipeline] model loaded: {info}", flush=True)
    splits = _build_splits(spec, tokenizer, protocol, run)
    run.save_json("splits", {name: split.describe() for name, split in splits.items()})
    if spec.progress:
        for split in splits.values():
            print(f"[pipeline] {split.describe()}", flush=True)
    return model, tokenizer, splits, precision


def _device(spec: PipelineSpec):
    """Real device only when CUDA exists; None keeps tensors where they are."""
    return spec.device if torch.cuda.is_available() else None


def _ppl(model, eval_ids, spec: PipelineSpec, device=None) -> dict[str, Any]:
    return rgeval.wikitext_perplexity(
        model, eval_ids, batch_size=spec.ppl_batch_size, device=_device(spec),
        progress=spec.progress,
    )


def _eval_bundle(
    model, tokenizer, eval_ids, prompts, spec: PipelineSpec, precision: str,
    protocol: dict[str, Any], run: rgrun.Run,
) -> dict[str, Any]:
    return rgeval.evaluate_all(
        model,
        tokenizer,
        eval_ids=eval_ids,
        prompts=prompts,
        protocol=protocol,
        precision=precision,
        device=spec.device if torch.cuda.is_available() else None,
        batch_size=spec.ppl_batch_size,
        mmlu_batch_size=spec.mmlu_batch_size,
        do_mmlu=prompts is not None,
        do_throughput=spec.do_throughput,
        progress=spec.progress,
        on_metric=lambda metrics: run.save_json("pruned_eval_partial", metrics),
    )


def _similarity_under_bias(
    model,
    calib_ids,
    run: rgrun.Run,
    spec: PipelineSpec,
    counts_unbalanced: dict[str, Any],
    counts_balanced: dict[str, Any],
    biases: dict[int, torch.Tensor],
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Notebook 03's rule: recompute similarity under the bias iff routing KL >= 1e-2.

    Layer 0's inputs are the embeddings, so no router bias can change them; deeper
    layers shift when earlier routing decisions change.
    """
    per_layer = counts_unbalanced["per_layer_counts"]
    kl = {
        int(layer): rgstats.kl(per_layer[layer], counts_balanced["per_layer_counts"][layer])
        for layer in per_layer
    }
    mean_kl = sum(kl.values()) / len(kl) if kl else 0.0
    report: dict[str, Any] = {"mean_kl": mean_kl, "recomputed": False}

    sim_unbalanced = rgsim.build_similarity(
        model, calib_ids, run.dir / "sim_unbalanced",
        max_tokens_per_layer=spec.max_tokens_per_layer or 100_000,
        batch_size=spec.similarity_batch_size, device=spec.device,
        progress=spec.progress,
    ).matrices

    sim_balanced = sim_unbalanced
    if mean_kl >= 1e-2:
        with torch.no_grad(), rgbalance.apply_bias(model, biases, device=spec.device):
            sim_balanced = rgsim.build_similarity(
                model, calib_ids, run.dir / "sim_balanced",
                max_tokens_per_layer=spec.max_tokens_per_layer or 100_000,
                batch_size=spec.similarity_batch_size, device=spec.device,
                progress=spec.progress,
            ).matrices
        report["recomputed"] = True
    report["per_layer_kl"] = {int(k): float(v) for k, v in kl.items()}
    return sim_unbalanced, sim_balanced, report


def _ratio_for(
    spec: PipelineSpec,
    kind: str,
    ratio: float,
    sim: dict[int, Any] | None,
) -> float | dict[int, float]:
    """Layer-wise budget applies only to the redundancy arm (baselines stay uniform)."""
    if spec.layer_wise_budget and kind == "redundancy" and sim is not None:
        return prune.layer_wise_ratios(sim, total_ratio=ratio)
    return ratio


def run_ablation(
    model,
    run: rgrun.Run,
    spec: PipelineSpec,
    protocol: dict[str, Any],
    baseline_ppl: float,
    counts_unbalanced: dict[str, Any],
    counts_balanced: dict[str, Any],
    biases: dict[int, torch.Tensor],
    sim_unbalanced: dict[int, Any],
    sim_balanced: dict[int, Any],
    eval_ids: torch.Tensor,
) -> tuple[dict[tuple[str, str, float], float], list[dict[str, Any]]]:
    """The 2xN ablation, evaluated by MaskSim (non-destructive). Notebook 03, cell 08."""
    rows: dict[tuple[str, str, float], float] = {}
    cells: list[dict[str, Any]] = []
    weight = float(protocol.get("design", {}).get("utilization_weight", 0.25))
    max_steps = int(protocol.get("budget", {}).get("max_greedy_steps", 64))

    for router_state in ("unbalanced", "balanced"):
        bias = None if router_state == "unbalanced" else biases
        counts = counts_unbalanced if router_state == "unbalanced" else counts_balanced
        sim = sim_unbalanced if router_state == "unbalanced" else sim_balanced
        for ranking in spec.kinds:
            if ranking == "random":
                for seed in spec.resolved_seeds(protocol):
                    for ratio in spec.resolved_ratios(protocol):
                        if spec.progress:
                            print(f"[ablation] {router_state} random seed={seed} ratio={ratio}", flush=True)
                        keeps = prune.keeps_for(model, "random", ratio, seed=seed)
                        with torch.no_grad(), moe.MaskSim(model, keeps, bias=bias):
                            ppl = _ppl(model, eval_ids, spec)
                        label = f"random[s{seed}]"
                        rows[(router_state, label, ratio)] = ppl["perplexity"]
                        cells.append(_cell(router_state, label, ratio, ppl, baseline_ppl, keeps, layer_budget=False))
            elif ranking in ("frequency", "redundancy"):
                for ratio in spec.resolved_ratios(protocol):
                    if spec.progress:
                        print(f"[ablation] {router_state} {ranking} ratio={ratio}", flush=True)
                    eff_ratio = _ratio_for(spec, ranking, ratio, sim)
                    keeps = prune.keeps_for(
                        model, ranking, eff_ratio,
                        per_layer_counts=counts["per_layer_counts"],
                        sim=sim, utilization_weight=weight, max_steps=max_steps,
                    )
                    with torch.no_grad(), moe.MaskSim(model, keeps, bias=bias):
                        ppl = _ppl(model, eval_ids, spec)
                    rows[(router_state, ranking, ratio)] = ppl["perplexity"]
                    cells.append(
                        _cell(router_state, ranking, ratio, ppl, baseline_ppl, keeps,
                              layer_budget=isinstance(eff_ratio, dict))
                    )
            else:
                raise ValueError(f"unknown ranking kind {ranking!r}")

    run.save_json("ablation_2x2", {"baseline_ppl": baseline_ppl, "cells": cells})
    return rows, cells


def _cell(router, ranking, ratio, ppl, baseline_ppl, keeps, layer_budget: bool) -> dict[str, Any]:
    return {
        "router": router,
        "ranking": ranking,
        "ratio": ratio,
        "perplexity": ppl["perplexity"],
        "delta_vs_baseline": ppl["perplexity"] - baseline_ppl,
        "kept_per_layer": len(keeps[sorted(keeps)[0]]),
        "layer_budget": layer_budget,
    }


def gain_over_frequency(
    rows: dict[tuple[str, str, float], float], ratios: Sequence[float]
) -> dict[str, float]:
    """ppl points saved by redundancy vs frequency, per router state and ratio."""
    gains: dict[str, float] = {}
    for router in ("unbalanced", "balanced"):
        for ratio in ratios:
            f = rows.get((router, "frequency", ratio))
            r = rows.get((router, "redundancy", ratio))
            if f is not None and r is not None:
                gains[f"{router}@{ratio:.2f}"] = f - r
    return gains


def run_pipeline(spec: PipelineSpec) -> dict[str, Any]:
    """Execute the full Steps 1-4 sequence (plus extension arms) in one process."""
    started = time.time()
    protocol = _resolve_protocol(spec)
    run = rgrun.new_run(spec.tag, protocol=protocol, results_root=spec.results_root)

    state: dict[str, Any] = {"run_id": run.run_id, "run_dir": str(run.dir)}
    notes: list[str] = []

    model, tokenizer, splits, precision = _bootstrap(spec, run, protocol)
    calib_ids = splits["calib"].input_ids
    eval_ids = splits["eval"].input_ids

    # -- 3. baseline load stats + gini decision gate ------------------------------
    counts_unbalanced = rgbalance.measure_model(
        model, calib_ids, None, spec.stats_batch_size, spec.device, spec.max_calib_tokens
    )
    counts_unbalanced_eval = rgbalance.measure_model(
        model, eval_ids, None, spec.stats_batch_size, spec.device, spec.max_eval_tokens
    )
    run.save_json("load_stats_calib", counts_unbalanced)
    run.save_json("load_stats_eval", counts_unbalanced_eval)
    gini = counts_unbalanced["gini_mean"]
    if spec.progress:
        print(f"[pipeline] baseline gini_mean={gini:.4f} (skip threshold "
              f"{protocol['gates']['gini_skip_balance']})", flush=True)

    # -- 4. balance ---------------------------------------------------------------
    balance_result = rgbalance.evaluate_balance(
        model, calib_ids, eval_ids, protocol,
        batch_size=spec.stats_batch_size, device=spec.device,
        max_tokens=spec.max_calib_tokens, progress=spec.progress,
        mode=spec.update_mode,
    )
    run.save_json("balance", balance_result.summary())
    balance_result.save(run.dir / "balance_biases.pt")
    biases = balance_result.biases
    if spec.progress:
        for warning in balance_result.warnings():
            print(f"[pipeline] balance WARNING: {warning}", flush=True)

    # -- 5. similarity ------------------------------------------------------------
    sim_unbalanced, sim_balanced, sim_report = _similarity_under_bias(
        model, calib_ids, run, spec,
        counts_unbalanced, balance_result.calib_after, biases,
    )
    run.save_json("similarity", {
        "unbalanced": {
            "layers": sorted(int(k) for k in sim_unbalanced),
            "num_experts_per_layer": {int(k): int(v.shape[0]) for k, v in sim_unbalanced.items()},
            "top_pairs": {
                int(k): rgsim.redundancy_pairs(v, top=3) for k, v in sim_unbalanced.items()
            },
        },
        "decision": sim_report,
    })

    baseline_ppl = _ppl(model, eval_ids, spec)["perplexity"]
    run.save_json("baseline_ppl", {"perplexity": baseline_ppl})

    # -- 6./7. ablation + gates -----------------------------------------------------
    rows, cells = run_ablation(
        model, run, spec, protocol, baseline_ppl,
        counts_unbalanced, balance_result.calib_after, biases,
        sim_unbalanced, sim_balanced, eval_ids,
    )
    ratios = spec.resolved_ratios(protocol)
    gains = gain_over_frequency(rows, ratios)
    headline = f"balanced@{max(ratios):.2f}"
    measured = {
        "gini_skip_balance": gini,
        "max_kl_nats": balance_result.kl_mean,
        "min_balance_transfer_ratio": balance_result.transfer_ratio,
    }
    if headline in gains:
        measured["min_gain_over_frequency_pts"] = gains[headline]
    gates = rgrun.summarize_gates(protocol, measured)
    run.save_json("gates", {"gains": gains, "headline_cell": headline, "gates": gates})
    if spec.progress:
        print(rgrun.format_gates(gates), flush=True)

    # -- 8. materialize the headline arm -------------------------------------------
    ranking = "redundancy" if "redundancy" in spec.kinds else spec.kinds[0]
    materialize_ratio = max(ratios)
    prompts, mmlu_notes = _maybe_mmlu_prompts(spec, protocol)
    notes.extend(mmlu_notes)

    if spec.progress:
        print(f"[pipeline] materializing {ranking}@{materialize_ratio} (balanced router)", flush=True)
    eff_ratio = _ratio_for(spec, ranking, materialize_ratio, sim_balanced)
    weight = float(protocol.get("design", {}).get("utilization_weight", 0.25))
    max_steps = int(protocol.get("budget", {}).get("max_greedy_steps", 64))
    keeps = prune.keeps_for(
        model, ranking, eff_ratio,
        per_layer_counts=balance_result.calib_after["per_layer_counts"],
        sim=sim_balanced, utilization_weight=weight, max_steps=max_steps,
    )
    checkpoint_before = rgeval.checkpoint_bytes(model)["bytes"]
    removal = prune.prune_with_keeps(model, keeps)
    run.save_json("pruned_removal", {**removal, "ranking": ranking, "ratio": materialize_ratio,
                                     "layer_budget": isinstance(eff_ratio, dict)})

    bundle = _eval_bundle(model, tokenizer, eval_ids, prompts, spec, precision, protocol, run)
    run.save_json("pruned_eval", bundle)
    ppl_pruned = bundle["perplexity"]["perplexity"]

    # The ablation's winning cell ran under the fitted bias, so the materialized model
    # must be evaluated under the same router condition: the bias is re-indexed to the
    # surviving experts (gate row j of the pruned model is original expert keep[j]).
    sliced_biases = rgbalance.slice_biases(biases, keeps)
    with rgbalance.apply_bias(model, sliced_biases, device=spec.device):
        ppl_pruned_balanced = _ppl(model, eval_ids, spec)["perplexity"]
    run.save_json("pruned_eval_balanced", {
        "ppl_pruned": ppl_pruned,
        "ppl_pruned_balanced": ppl_pruned_balanced,
        "note": "same pruned weights; selection biased by the pre-pruning fitted bias (re-indexed)",
    })

    ppl_deployed = ppl_pruned_balanced

    # -- 9. extension: refit the biases on the pruned model --------------------------
    refit_summary: dict[str, Any] | None = None
    ppl_refit: float | None = None
    if spec.with_refit:
        try:
            refit = rgbalance.evaluate_balance(
                model, calib_ids, eval_ids, protocol,
                batch_size=spec.stats_batch_size, device=spec.device,
                max_tokens=spec.max_calib_tokens, progress=spec.progress,
                mode=spec.update_mode,
            )
            refit_summary = refit.summary()
            refit_summary["context"] = "second calibration pass on the pruned model"
            run.save_json("balance_after_prune", refit_summary)
            refit.save(run.dir / "balance_biases_pruned.pt")
            with rgbalance.apply_bias(model, refit.biases, device=spec.device):
                ppl_refit = _ppl(model, eval_ids, spec)["perplexity"]
            run.save_json("pruned_refit_ppl", {
                "ppl_pruned": ppl_deployed,
                "ppl_refit": ppl_refit,
                "delta": ppl_refit - ppl_deployed,
                "note": "refit biases fitted on the pruned model; negative delta = recovery "
                        "vs the pre-pruning bias re-applied to the pruned router",
            })
        except Exception as exc:  # noqa: BLE001 - an extension arm must not kill the run
            notes.append(f"refit arm failed: {type(exc).__name__}: {exc}")
            if spec.progress:
                traceback.print_exc()

    saved_bytes = checkpoint_before - rgeval.checkpoint_bytes(model)["bytes"]

    # -- 10. extension: merge instead of remove (fresh weights, one ratio per load) ---
    merge_summary: dict[str, Any] | None = None
    if spec.with_merge:
        merge_summary = _run_merge_arm(
            spec, protocol, run, eval_ids, biases,
            balance_result.calib_after["per_layer_counts"], sim_balanced,
            rows, prompts, precision,
        )

    # -- summary ---------------------------------------------------------------------
    state.update(
        {
            "gini_baseline": gini,
            "balance": balance_result.summary(),
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": ppl_pruned,
            "pruned_ppl_balanced": ppl_pruned_balanced,
            "ppl_delta": ppl_pruned - baseline_ppl,
            "headline_cell": headline,
            "gains_over_frequency": gains,
            "checkpoint_bytes_saved": saved_bytes,
            "refit": refit_summary,
            "refit_ppl": ppl_refit,
            "merge": merge_summary,
            "notes": notes,
            "elapsed_s": round(time.time() - started, 2),
        }
    )
    run.save_json("summary", state)
    if spec.progress:
        print(f"[pipeline] done -> {run.dir}", flush=True)
    return state


def _run_merge_arm(
    spec: PipelineSpec,
    protocol: dict[str, Any],
    run: rgrun.Run,
    eval_ids: torch.Tensor,
    biases: dict[int, torch.Tensor],
    balanced_counts: dict[int, Sequence[int]],
    sim_balanced: dict[int, Any],
    ablation_rows: dict[tuple[str, str, float], float],
    prompts: Any,
    precision: str,
) -> dict[str, Any]:
    """Merge arm: fresh weights (materialized removal above destroyed them), merge, eval.

    Merging changes weights, so there is no MaskSim preview; each ratio needs its
    own reload. The comparison row against deletion is labeled with its provenance
    (mask-sim vs real), never presented as apples-to-apples at unmaterialized ratios.
    """
    loader = spec.loader or default_loader(spec)
    weight = float(protocol.get("design", {}).get("utilization_weight", 0.25))
    max_steps = int(protocol.get("budget", {}).get("max_greedy_steps", 64))
    results: dict[str, Any] = {"ratios": {}}
    try:
        for ratio in spec.resolved_ratios(protocol):
            model, tokenizer, info = loader()
            eff_ratio = _ratio_for(spec, "redundancy", ratio, sim_balanced)
            plans = rgmerge.merge_keeps(
                sim_balanced, eff_ratio, per_layer_counts=balanced_counts,
                utilization_weight=weight, max_steps=max_steps,
            )
            merge_report = rgmerge.apply_merge(model, plans)
            ppl = _ppl(model, eval_ids, spec)["perplexity"]
            # Same router condition as the balanced ablation cells: the fitted bias is
            # folded through each merge (the merged gate row is the pair's average, so
            # the matching selection bias is too) and re-indexed to the survivors.
            folded: dict[int, torch.Tensor] = {}
            for layer, bias in biases.items():
                plan = plans.get(int(layer))
                if plan is None:
                    continue
                b = bias.detach().float().clone()
                for dst, src in plan["pairs"]:
                    b[dst] = (b[dst] + b[src]) / 2.0
                folded[int(layer)] = b
            sliced = rgbalance.slice_biases(folded, {l: p["survivors"] for l, p in plans.items()})
            with rgbalance.apply_bias(model, sliced, device=spec.device):
                ppl_balanced = _ppl(model, eval_ids, spec)["perplexity"]
            params = rgeval.param_report(model, precision)
            row = {
                "ratio": ratio,
                "perplexity": ppl,
                "perplexity_balanced": ppl_balanced,
                "router": "balanced",
                "merged_total": merge_report["merged_total"],
                "removed_total": merge_report["removed_total"],
                "num_experts_after": merge_report["num_experts"],
                "checkpoint_gb": params["checkpoint_gb"],
                "delete_masksim_ppl": ablation_rows.get(("balanced", "redundancy", ratio)),
                "provenance": "merge=real merged weights under the folded fitted bias; "
                              "delete=MaskSim (weights unchanged) under the fitted bias",
            }
            results["ratios"][f"{ratio:.2f}"] = row
            run.save_json(f"merge_eval_{ratio:.2f}", {**row, "survivors": merge_report["survivors"]})
            rgmodels.free(model, tokenizer)
    except Exception as exc:  # noqa: BLE001 - the merge arm must not kill the run
        results["error"] = f"{type(exc).__name__}: {exc}"
        if spec.progress:
            traceback.print_exc()
    return results


# ------------------------------------------------------------------ stage commands


def _baseline_stats(
    spec: PipelineSpec,
    run: rgrun.Run,
    protocol: dict[str, Any],
    model,
    splits: dict[str, rgdata.TokenSplit],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Load statistics on both splits + the gini decision gate; notebook 01/02 prologue."""
    calib_ids = splits["calib"].input_ids
    eval_ids = splits["eval"].input_ids
    counts_calib = rgbalance.measure_model(
        model, calib_ids, None, spec.stats_batch_size, spec.device, spec.max_calib_tokens
    )
    counts_eval = rgbalance.measure_model(
        model, eval_ids, None, spec.stats_batch_size, spec.device, spec.max_eval_tokens
    )
    run.save_json("load_stats_calib", counts_calib)
    run.save_json("load_stats_eval", counts_eval)
    gini = counts_calib["gini_mean"]
    gates = rgrun.summarize_gates(protocol, {"gini_skip_balance": gini})
    run.save_json("gini_gate", {"gini_mean": gini, "gates": gates})
    if spec.progress:
        print(rgrun.format_gates(gates), flush=True)
    return counts_calib, counts_eval, gini


def run_stats_only(spec: PipelineSpec) -> dict[str, Any]:
    """Step 1 standalone: env + model + splits + load statistics + gini decision."""
    protocol = _resolve_protocol(spec)
    run = rgrun.new_run(spec.tag or "stats", protocol=protocol, results_root=spec.results_root)
    model, tokenizer, splits, precision = _bootstrap(spec, run, protocol)
    counts_calib, counts_eval, gini = _baseline_stats(spec, run, protocol, model, splits)
    out = {
        "run_dir": str(run.dir),
        "gini_calib": gini,
        "gini_eval": counts_eval["gini_mean"],
        "skip_threshold": protocol["gates"]["gini_skip_balance"],
    }
    run.save_json("summary", out)
    if spec.progress:
        print(f"[stats] done -> {run.dir}", flush=True)
    return out


def run_balance_only(spec: PipelineSpec) -> dict[str, Any]:
    """Step 3 standalone: load stats, fit biases, both guards, biases saved to disk."""
    protocol = _resolve_protocol(spec)
    run = rgrun.new_run(spec.tag or "balance", protocol=protocol, results_root=spec.results_root)
    model, tokenizer, splits, precision = _bootstrap(spec, run, protocol)
    counts_calib, counts_eval, gini = _baseline_stats(spec, run, protocol, model, splits)
    result = rgbalance.evaluate_balance(
        model,
        splits["calib"].input_ids,
        splits["eval"].input_ids,
        protocol,
        batch_size=spec.stats_batch_size,
        device=spec.device,
        max_tokens=spec.max_calib_tokens,
        progress=spec.progress,
        mode=spec.update_mode,
    )
    run.save_json("balance", result.summary())
    result.save(run.dir / "balance_biases.pt")
    gates = rgrun.summarize_gates(
        protocol,
        {
            "gini_skip_balance": gini,
            "max_kl_nats": result.kl_mean,
            "min_balance_transfer_ratio": result.transfer_ratio,
        },
    )
    run.save_json("gates", {"gates": gates})
    out = {
        "run_dir": str(run.dir),
        "gini_calib": gini,
        "gini_gain_eval": result.gini_gain_eval,
        "kl_mean_nats": result.kl_mean,
        "transfer_ratio": result.transfer_ratio,
        "warnings": result.warnings(),
        "biases_path": str(run.dir / "balance_biases.pt"),
    }
    run.save_json("summary", out)
    if spec.progress:
        print(rgrun.format_gates(gates), flush=True)
        print(f"[balance] done -> {run.dir}", flush=True)
    return out


def run_similarity_only(spec: PipelineSpec) -> dict[str, Any]:
    """Step 4a standalone: fit biases (needed for the under-bias decision), then build
    the per-layer cosine matrices, cached and resumable across calls."""
    protocol = _resolve_protocol(spec)
    run = rgrun.new_run(spec.tag or "similarity", protocol=protocol, results_root=spec.results_root)
    model, tokenizer, splits, precision = _bootstrap(spec, run, protocol)
    counts_calib, counts_eval, gini = _baseline_stats(spec, run, protocol, model, splits)
    result = rgbalance.evaluate_balance(
        model,
        splits["calib"].input_ids,
        splits["eval"].input_ids,
        protocol,
        batch_size=spec.stats_batch_size,
        device=spec.device,
        max_tokens=spec.max_calib_tokens,
        progress=spec.progress,
        mode=spec.update_mode,
    )
    run.save_json("balance", result.summary())
    result.save(run.dir / "balance_biases.pt")
    sim_unbalanced, sim_balanced, decision = _similarity_under_bias(
        model,
        splits["calib"].input_ids,
        run,
        spec,
        counts_calib,
        result.calib_after,
        result.biases,
    )
    run.save_json("similarity", {
        "unbalanced": {
            "layers": sorted(int(k) for k in sim_unbalanced),
            "num_experts_per_layer": {int(k): int(v.shape[0]) for k, v in sim_unbalanced.items()},
            "top_pairs": {int(k): rgsim.redundancy_pairs(v, top=3) for k, v in sim_unbalanced.items()},
        },
        "decision": decision,
        "cache_dir": str(run.dir / "sim_unbalanced"),
    })
    out = {
        "run_dir": str(run.dir),
        "mean_kl": decision["mean_kl"],
        "recomputed_under_bias": decision["recomputed"],
        "sim_dir": str(run.dir / "sim_unbalanced"),
    }
    run.save_json("summary", out)
    if spec.progress:
        print(f"[similarity] done -> {run.dir}", flush=True)
    return out
