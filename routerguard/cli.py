"""Command-line entry point: ``python -m routerguard.cli <command>``.

Commands
--------
  env          version pins + GPU + MoE-structure assertions (no model download)
  stats        Step 1: model, splits, load statistics, gini decision gate
  balance      Step 3: fit router biases, both guards, biases saved to disk
  similarity   Step 4a: per-layer cosine matrices (cached/resumable)
  pipeline     Steps 1-4 end to end in one process (+ the protocol-v2 arms)

Examples (Colab, after the install cell)::

  python -m routerguard.cli env
  python -m routerguard.cli pipeline --tag t4_full
  python -m routerguard.cli pipeline --update-mode proportional --no-merge
  python -m routerguard.cli balance --model qwen --device cuda
  python -m routerguard.cli pipeline --device cpu --fast          # CPU smoke run

Every command writes a ``results/<run_id>/`` directory with ``config.resolved.yaml``,
``env.txt``, ``git_sha.txt`` and ``metrics/*.json`` — same layout as the notebooks.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import env as rgenv
from . import pipeline as rgpipeline
from . import run as rgrun


def _parse_value(text: str) -> Any:
    """YAML-ish scalar parsing so --override can express numbers/bools/null."""
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - fall back to the raw string
        return text


def _parse_override(pairs: list[str]) -> dict[str, Any]:
    """``["balance.default_u=0.005", "frozen=true"]`` -> nested dict."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--override expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return out


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m routerguard.cli",
        description="RouterGuard: post-hoc balancing + redundancy-aware expert pruning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    env_p = sub.add_parser("env", help="version pins, GPU info, dependency-range assertions")
    env_p.add_argument("--strict", action="store_true",
                       help="also require CUDA (the default on GPU runtimes)")

    # shared loader/protocol arguments
    def add_loader_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default="olmoe", choices=["olmoe", "qwen"])
        p.add_argument("--model-id", default=None, help="override the protocol's HF id")
        p.add_argument("--precision", default="auto",
                       choices=["auto", "bf16", "8bit", "4bit"])
        p.add_argument("--device", default="cuda")
        p.add_argument("--tag", default=None, help="run-id tag (default: command name)")
        p.add_argument("--results-dir", default=None, help="results root (default: ./results)")
        p.add_argument("--override", action="append", default=[],
                       metavar="KEY.PATH=VALUE", help="protocol override, repeatable")
        p.add_argument("--fast", action="store_true",
                       help="cap tokens (smoke run; numbers are NOT protocol numbers)")
        p.add_argument("--quiet", action="store_true")

    HELP = {
        "stats": "Step 1: model, splits, load statistics, gini decision gate",
        "balance": "Step 3: fit router biases, both guards, biases saved to disk",
        "similarity": "Step 4a: per-layer cosine matrices (cached/resumable)",
        "pipeline": "Steps 1-4 end to end in one process (+ protocol-v2 arms)",
    }
    for name in ("stats", "balance", "similarity", "pipeline"):
        add_loader_args(sub.add_parser(name, help=HELP[name]))

    p = sub.choices["pipeline"]
    p.add_argument("--ratios", default=None, help="comma-separated, e.g. 0.25,0.5")
    p.add_argument("--seeds", default=None, help="comma-separated seeds for the random arm")
    p.add_argument("--kinds", default="random,frequency,redundancy")
    p.add_argument("--update-mode", default=None, choices=["sign", "proportional"],
                   help="None = protocol balance.update_mode")
    p.add_argument("--no-refit", action="store_true", help="skip the post-prune bias refit arm")
    p.add_argument("--merge", action="store_true",
                   help="run the merge arm (needs unquantized experts; one reload per ratio)")
    p.add_argument("--layer-budget", action="store_true",
                   help="allocate redundancy removals by measured layer redundancy")
    p.add_argument("--no-mmlu", action="store_true")
    p.add_argument("--no-throughput", action="store_true")
    p.add_argument("--max-calib-tokens", type=int, default=None)
    p.add_argument("--max-eval-tokens", type=int, default=None)
    p.add_argument("--max-tokens-per-layer", type=int, default=None)
    return parser


def spec_from_args(args: argparse.Namespace, command: str) -> rgpipeline.PipelineSpec:
    overrides = _parse_override(args.override)
    max_calib = args.max_calib_tokens
    max_eval = args.max_eval_tokens
    max_layer = args.max_tokens_per_layer if hasattr(args, "max_tokens_per_layer") else None
    if getattr(args, "fast", False) and max_calib is None:
        max_calib, max_eval, max_layer = 4096, 2048, 4096
    kwargs: dict[str, Any] = {}
    if command == "pipeline":
        kwargs = dict(
            ratios=_floats(args.ratios) if args.ratios else None,
            seeds=_ints(args.seeds) if args.seeds else None,
            kinds=tuple(k.strip() for k in args.kinds.split(",") if k.strip()),
            update_mode=args.update_mode,
            with_refit=not args.no_refit,
            with_merge=args.merge,
            layer_wise_budget=args.layer_budget,
            do_mmlu=not args.no_mmlu,
            do_throughput=not args.no_throughput,
            max_tokens_per_layer=max_layer,
        )
    return rgpipeline.PipelineSpec(
        model=args.model,
        model_id=args.model_id,
        precision=args.precision,
        device=args.device,
        tag=args.tag or command,
        max_calib_tokens=max_calib,
        max_eval_tokens=max_eval,
        overrides=overrides,
        results_root=args.results_dir,
        progress=not args.quiet,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "env":
        return _cmd_env(args)

    try:
        spec = spec_from_args(args, args.command)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"error building run spec: {exc}", file=sys.stderr)
        return 2

    if args.command == "pipeline":
        state = rgpipeline.run_pipeline(spec)
    elif args.command == "stats":
        state = rgpipeline.run_stats_only(spec)
    elif args.command == "balance":
        state = rgpipeline.run_balance_only(spec)
    elif args.command == "similarity":
        state = rgpipeline.run_similarity_only(spec)
    else:  # pragma: no cover - argparse enforces the choices
        parser.error(f"unknown command {args.command!r}")

    printable = {k: v for k, v in state.items() if isinstance(v, (str, int, float, bool, list))}
    print(json.dumps(printable, indent=2, default=str))
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    import torch

    strict = args.strict or torch.cuda.is_available()
    try:
        report = rgenv.assert_env(strict=strict)
    except Exception as exc:  # noqa: BLE001 - a failed env check IS the answer
        print(f"ENV CHECK FAILED\n{exc}", file=sys.stderr)
        return 1
    print(rgenv.env_text())
    print(json.dumps(report["versions"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
