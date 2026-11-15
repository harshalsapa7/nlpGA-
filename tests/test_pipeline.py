"""Pipeline tests: the whole Steps 1-4 sequence (+ extension arms) on tiny models.

The loader and splits are injected, so nothing is downloaded and nothing needs a
GPU; the point is that the notebook-accurate call sequence is *executed* end to end
and writes a complete, parseable run directory.
"""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from routerguard import cli, eval as rgeval, pipeline, run as rgrun
from routerguard.data import TokenSplit
from routerguard.pipeline import PipelineSpec, run_balance_only, run_pipeline, run_similarity_only, run_stats_only
from tests.conftest import build_model, make_ids
from tests.test_step3 import _CharTokenizer, _letter_prompts


class _Info:
    """Stand-in for models.LoadInfo."""

    def __init__(self):
        self.precision = "bf16"

    def to_dict(self):
        return {"model_id": "tiny-olmoe", "precision": "bf16", "notes": ["test loader"]}

    def __str__(self):
        return "tiny-olmoe [bf16] (test loader)"


def _tiny_loader(seed: int = 0):
    """Deterministic loader: same seed -> identical weights, so a reloaded model
    matches the one the similarity matrices were computed on (what the merge arm
    assumes about a real checkpoint reload)."""
    def loader():
        model = build_model("olmoe", seed=seed)
        tokenizer = SimpleNamespace(name_or_path="tiny")
        return model, tokenizer, _Info()
    return loader


def _tiny_splits():
    return {
        "calib": TokenSplit("calib", make_ids(batch=8, seq=16, seed=3),
                            {"dataset": "tiny", "config": "tiny", "split": "train", "token_offset": 0}),
        "eval": TokenSplit("eval", make_ids(batch=4, seq=16, seed=4),
                           {"dataset": "tiny", "config": "tiny", "split": "test", "token_offset": 0}),
    }


def _spec(tmp_path, **kwargs):
    base = dict(
        model="olmoe",
        device="cpu",
        tag="test",
        seeds=(42,),
        kinds=("random", "frequency", "redundancy"),
        ratios=(0.25, 0.5),
        update_mode="sign",
        with_refit=True,
        with_merge=False,
        layer_wise_budget=False,
        do_mmlu=False,
        do_throughput=False,
        loader=_tiny_loader(),
        splits=_tiny_splits(),
        protocol=rgrun.load_protocol(),
        results_root=tmp_path / "results",
        progress=False,
    )
    base.update(kwargs)
    return PipelineSpec(**base)


# ------------------------------------------------------------------ full pipeline


def test_run_pipeline_end_to_end(tmp_path):
    spec = _spec(tmp_path)
    state = run_pipeline(spec)

    assert state["run_dir"] and (tmp_path / "results").exists()
    for key in ("gini_baseline", "baseline_ppl", "pruned_ppl", "ppl_delta",
                "headline_cell", "gains_over_frequency", "checkpoint_bytes_saved"):
        assert key in state, key
    assert state["checkpoint_bytes_saved"] > 0
    assert state["refit"] is not None          # protocol v2 arm ran by default
    assert state["refit_ppl"] is not None
    assert "balanced@0.50" in state["gains_over_frequency"]

    run_dir = tmp_path / "results" / state["run_id"]
    metrics = sorted(p.name for p in (run_dir / "metrics").glob("*.json"))
    expected = {
        "load.json", "splits.json", "load_stats_calib.json", "load_stats_eval.json",
        "balance.json", "similarity.json", "baseline_ppl.json", "ablation_2x2.json",
        "gates.json", "pruned_removal.json", "pruned_eval.json",
        "balance_after_prune.json", "pruned_refit_ppl.json", "summary.json",
    }
    assert expected.issubset(set(metrics)), (expected - set(metrics))
    for path in (run_dir / "metrics").glob("*.json"):
        json.loads(path.read_text())           # every metrics file is parseable
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "balance_biases.pt").exists()
    assert (run_dir / "balance_biases_pruned.pt").exists()
    assert (run_dir / "env.txt").exists()

    ablation = json.loads((run_dir / "metrics" / "ablation_2x2.json").read_text())
    # 2 router states x (random@2ratios + frequency@2 + redundancy@2) = 12 cells
    assert len(ablation["cells"]) == 12
    assert all(c["perplexity"] > 0 for c in ablation["cells"])


def test_run_pipeline_merge_arm(tmp_path):
    spec = _spec(tmp_path, with_merge=True)
    state = run_pipeline(spec)

    assert state["merge"] is not None
    assert "error" not in state["merge"], state["merge"]
    for ratio_key, row in state["merge"]["ratios"].items():
        assert row["merged_total"] == row["removed_total"] > 0
        assert row["perplexity"] > 0
        assert "provenance" in row          # the honest delete-vs-merge labeling
    run_dir = tmp_path / "results" / state["run_id"]
    assert (run_dir / "metrics" / "merge_eval_0.50.json").exists()


def test_run_pipeline_layer_budget(tmp_path):
    spec = _spec(tmp_path, layer_wise_budget=True, kinds=("redundancy",))
    state = run_pipeline(spec)
    ablation = None
    run_dir = tmp_path / "results" / state["run_id"]
    ablation = json.loads((run_dir / "metrics" / "ablation_2x2.json").read_text())
    budget_cells = [c for c in ablation["cells"] if c["ranking"] == "redundancy"]
    assert budget_cells and all(c["layer_budget"] for c in budget_cells)
    removal = json.loads((run_dir / "metrics" / "pruned_removal.json").read_text())
    assert removal["layer_budget"] is True


def test_run_pipeline_proportional_mode(tmp_path):
    spec = _spec(tmp_path, update_mode="proportional", kinds=("frequency", "redundancy"))
    state = run_pipeline(spec)
    assert state["balance"]["update_mode"] == "proportional"


def test_run_pipeline_refit_can_be_disabled(tmp_path):
    spec = _spec(tmp_path, with_refit=False, kinds=("frequency",))
    state = run_pipeline(spec)
    assert state["refit"] is None and state["refit_ppl"] is None
    run_dir = tmp_path / "results" / state["run_id"]
    assert not (run_dir / "metrics" / "balance_after_prune.json").exists()


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA",
    )),
])
def test_kaggle_smoke_settings_complete_pipeline(tmp_path, device):
    """Actual notebook settings, real tiny OLMoE; no checkpoint/data downloads.

    The injected model is unquantized. Kaggle's preflight separately tests int8.
    """
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks/04_kaggle_safe_run.ipynb").read_text()
    )
    source = next(cell["source"] for cell in notebook["cells"] if cell["id"] == "experiment-helper")
    namespace = {"RESULTS_DIR": tmp_path / "results"}
    exec("".join(source), namespace)
    settings = {**namespace["BASE_SETTINGS"], **namespace["SMOKE_SETTINGS"]}

    def loader():
        model = build_model("olmoe").to(
            device=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        )
        info = _Info()
        info.precision = "fp16" if device.startswith("cuda") else "fp32"
        return model, SimpleNamespace(name_or_path="tiny"), info

    splits = {}
    for name, section, seed in (("calib", "calibration", 3), ("eval", "eval", 4)):
        config = settings["overrides"][section]
        splits[name] = TokenSplit(
            name, make_ids(batch=config["n_seqs"], seq=config["seq_len"], seed=seed),
            {"dataset": "tiny", "config": "tiny", "split": name, "token_offset": 0},
        )
    settings.update(device=device, loader=loader, splits=splits, progress=False)
    state = run_pipeline(PipelineSpec(**settings))
    run_dir = Path(state["run_dir"])
    summary = json.loads((run_dir / "metrics/summary.json").read_text())
    ablation = json.loads((run_dir / "metrics/ablation_2x2.json").read_text())
    assert len(ablation["cells"]) == 6  # 2 router states x 3 pruning methods
    assert summary["baseline_ppl"] > 0
    assert summary["pruned_ppl_balanced"] > 0
    assert summary["checkpoint_bytes_saved"] > 0
    assert summary["refit"] is None
    assert summary["merge"] is None
    assert (run_dir / "balance_biases.pt").is_file()


def _full_eval_spec(tmp_path, monkeypatch, device="cpu"):
    monkeypatch.setattr(pipeline.rgdata, "build_mmlu_prompts", lambda **kwargs: _letter_prompts())

    def loader():
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        return (
            build_model("olmoe").to(device=device, dtype=dtype),
            _CharTokenizer(), SimpleNamespace(precision="fp16" if device.startswith("cuda") else "fp32"),
        )

    return _spec(
        tmp_path, device=device, loader=loader, do_mmlu=True, do_throughput=True,
        protocol=None, overrides={
            "metrics": {"throughput": {"seq_len": 8, "warmup": 1, "runs": 2}},
        },
    )


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA",
    )),
])
def test_full_pipeline_including_mmlu_throughput_and_refit(tmp_path, monkeypatch, device):
    """Don't disable the full-only arms in integration coverage."""
    state = run_pipeline(_full_eval_spec(tmp_path, monkeypatch, device))
    run_dir = Path(state["run_dir"])
    metrics = json.loads((run_dir / "metrics/pruned_eval.json").read_text())
    assert metrics["mmlu"]["n_questions"] == 4
    assert len(metrics["throughput"]["seconds_per_run"]) == 2
    assert metrics["throughput"]["seq_len"] == 8  # resolved overrides reach evaluation
    assert metrics["perplexity"]["perplexity"] > 0
    assert state["refit_ppl"] is not None
    assert not any("failed" in note for note in state["notes"])
    partial = json.loads((run_dir / "metrics/pruned_eval_partial.json").read_text())
    assert partial == metrics


def test_failed_throughput_retains_completed_mmlu_and_perplexity(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("throughput probe failed")

    monkeypatch.setattr(rgeval, "throughput", fail)
    with pytest.raises(RuntimeError, match="throughput probe failed"):
        run_pipeline(_full_eval_spec(tmp_path, monkeypatch))
    run_dir, = (tmp_path / "results").iterdir()
    partial = json.loads((run_dir / "metrics/pruned_eval_partial.json").read_text())
    assert partial["perplexity"]["perplexity"] > 0
    assert partial["mmlu"]["n_questions"] == 4
    assert "throughput" not in partial
    assert not (run_dir / "metrics/pruned_eval.json").exists()
    assert not (run_dir / "metrics/summary.json").exists()  # failure is not success


# ------------------------------------------------------------------ stage commands


def test_run_stats_only(tmp_path):
    spec = _spec(tmp_path)
    out = run_stats_only(spec)
    assert set(out) >= {"run_dir", "gini_calib", "gini_eval", "skip_threshold"}
    run_dir = tmp_path / "results" / out["run_dir"]
    assert (run_dir / "metrics" / "gini_gate.json").exists()
    assert (run_dir / "metrics" / "load_stats_calib.json").exists()


def test_run_balance_only(tmp_path):
    spec = _spec(tmp_path)
    out = run_balance_only(spec)
    assert "biases_path" in out and out["kl_mean_nats"] >= 0
    assert out["warnings"] == [] or isinstance(out["warnings"], list)
    biases_path = tmp_path / "results" / out["run_dir"] / "balance_biases.pt"
    assert biases_path.exists()


def test_run_similarity_only_caches_matrices(tmp_path):
    spec = _spec(tmp_path)
    out = run_similarity_only(spec)
    sim_dir = tmp_path / "results" / out["run_dir"] / "sim_unbalanced"
    assert (sim_dir / "layer_00.npy").exists()
    assert (sim_dir / "layer_01.npy").exists()
    assert isinstance(out["recomputed_under_bias"], bool)


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
@pytest.mark.parametrize("recompute", [False, True])
def test_similarity_passes_device_to_both_collections(tmp_path, monkeypatch, device, recompute):
    """Check GPU call wiring even on CPU-only CI; don't mock the decision logic."""
    calls = []

    def build_similarity(model, ids, directory, **kwargs):
        assert kwargs["device"] == device
        calls.append(directory.name)
        return SimpleNamespace(matrices={0: directory.name})

    monkeypatch.setattr(pipeline.rgsim, "build_similarity", build_similarity)
    monkeypatch.setattr(pipeline.rgbalance, "apply_bias", lambda *args, **kwargs: nullcontext())
    before = {"per_layer_counts": {0: [10, 1]}}
    after = {"per_layer_counts": {0: [1, 10]}} if recompute else before
    _, _, report = pipeline._similarity_under_bias(
        None, make_ids(), SimpleNamespace(dir=tmp_path),
        PipelineSpec(device=device, progress=False), before, after, {},
    )
    assert calls == (["sim_unbalanced", "sim_balanced"] if recompute else ["sim_unbalanced"])
    assert report["recomputed"] is recompute


# ----------------------------------------------------------------------- CLI wiring


def test_cli_parse_overrides_and_flags():
    parser = cli.build_parser()
    args = parser.parse_args([
        "pipeline", "--device", "cpu", "--ratios", "0.25,0.5",
        "--update-mode", "proportional", "--merge", "--layer-budget",
        "--override", "balance.default_u=0.002", "--override", "frozen=true",
        "--fast", "--tag", "smoke",
    ])
    spec = cli.spec_from_args(args, "pipeline")
    assert spec.device == "cpu"
    assert spec.ratios == [0.25, 0.5]
    assert spec.update_mode == "proportional"
    assert spec.with_merge is True and spec.layer_wise_budget is True
    assert spec.overrides == {"balance": {"default_u": 0.002}, "frozen": True}
    assert spec.max_calib_tokens == 4096 and spec.max_eval_tokens == 2048
    assert spec.tag == "smoke"


def test_cli_env_command_runs_in_pinned_env(capsys):
    """In the pinned venv (transformers 4.48.3, hub<1.0, datasets<5) env passes."""
    assert cli.main(["env"]) == 0
    captured = capsys.readouterr()
    assert "transformers" in captured.out
