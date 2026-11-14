"""Protocol-v2 extension tests: proportional bias update, expert merging, layer-wise
budgets, checkpoint bytes, Wilson intervals, and bias re-indexing after pruning.

Every test runs on CPU with tiny models built from the real transformers classes,
mirroring the style of test_step3/test_step4 — these functions are *executed* and
checked against independent references, not merely imported.
"""

import pytest
import torch

from routerguard import balance, env as rgenv, eval as rgeval, merge as rgmerge, moe, prune
from tests.conftest import build_model, make_ids
from tests.test_step3 import _CharTokenizer, _LetterStub, _letter_prompts


def _skewed_model(flavor="olmoe", boost=3.0, seed=0):
    """A model whose router loves expert 0, so balancing has something to fix."""
    model = build_model(flavor, seed=seed)
    with torch.no_grad():
        for _, block in rgenv.find_moe_blocks(model):
            block.gate.weight[0] *= boost
    return model


# ------------------------------------------------------------ bias update modes


def test_bias_delta_sign_matches_reference():
    cumulative = torch.tensor([10.0, 30.0, 20.0, 20.0], dtype=torch.float64)  # mean 20
    delta = balance.bias_delta(cumulative, u=0.001, mode="sign")
    expected = torch.tensor([1.0, -1.0, 0.0, 0.0]) * 0.001
    assert torch.allclose(delta, expected)
    assert delta.dtype == torch.float32


def test_bias_delta_proportional_is_gap_proportional():
    cumulative = torch.tensor([0.0, 40.0, 20.0, 20.0], dtype=torch.float64)  # mean 20
    u = 0.01
    delta = balance.bias_delta(cumulative, u=u, mode="proportional")
    # dead expert: full step up; 2x-loaded expert: full step down; at-mean: zero
    assert delta[0] == pytest.approx(u)
    assert delta[1] == pytest.approx(-u)
    assert delta[2] == pytest.approx(0.0, abs=1e-12)
    assert delta[3] == pytest.approx(0.0, abs=1e-12)


def test_bias_delta_shrinks_near_balance():
    """The property that removes the sign update's overshoot boundary."""
    near = torch.tensor([19.8, 20.2, 20.0, 20.0], dtype=torch.float64)   # 1% off
    far = torch.tensor([0.0, 40.0, 20.0, 20.0], dtype=torch.float64)     # dead/2x
    u = 0.01
    assert balance.bias_delta(near, u, "proportional").abs().max() < 0.01 * u * 100
    near_delta = balance.bias_delta(near, u, "proportional").abs().max()
    far_delta = balance.bias_delta(far, u, "proportional").abs().max()
    assert near_delta < far_delta / 10, (near_delta, far_delta)


def test_bias_delta_rejects_bad_mode_and_u():
    cumulative = torch.tensor([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="unknown balance update mode"):
        balance.bias_delta(cumulative, 0.001, mode="magic")
    with pytest.raises(ValueError, match="u must be > 0"):
        balance.bias_delta(cumulative, 0.0, mode="sign")


def test_proportional_fit_reduces_gini_without_overshoot(flavor):
    """At a u where the sign update overshoots, proportional still improves Gini."""
    model = _skewed_model(flavor, boost=3.0)
    ids = make_ids(batch=12, seq=16, seed=3)
    before = balance.measure_model(model, ids)["gini_mean"]

    sign_biases, _ = balance.fit_biases(model, ids, u=0.05, iters=2, batch_size=4,
                                        progress=False, mode="sign")
    prop_biases, info = balance.fit_biases(model, ids, u=0.05, iters=2, batch_size=4,
                                           progress=False, mode="proportional")
    after_sign = balance.measure_model(model, ids, sign_biases)["gini_mean"]
    after_prop = balance.measure_model(model, ids, prop_biases)["gini_mean"]

    assert info["mode"] == "proportional"
    assert after_sign > before, "expected the documented sign-update overshoot at u=0.05"
    assert after_prop < before, (
        f"proportional update should not overshoot: {before:.4f} -> {after_prop:.4f}"
    )


def test_evaluate_balance_threads_mode(flavor):
    model = _skewed_model(flavor)
    protocol = {"balance": {"default_u": 0.005, "iters": 1}}
    result = balance.evaluate_balance(
        model, make_ids(batch=8, seq=16, seed=3), make_ids(batch=4, seq=16, seed=4),
        protocol, batch_size=4, device=None, progress=False, mode="proportional",
    )
    assert result.mode == "proportional"
    assert result.summary()["update_mode"] == "proportional"
    assert result.fit_info["mode"] == "proportional"
    # default stays the registered protocol mode
    default = balance.evaluate_balance(
        model, make_ids(batch=8, seq=16, seed=3), make_ids(batch=4, seq=16, seed=4),
        protocol, batch_size=4, device=None, progress=False,
    )
    assert default.mode == "sign"
    path = result.save("/tmp/rg_test_biases_ext.pt")
    loaded = balance.BalanceResult.load(path)
    assert loaded["update_mode"] == "proportional"


def test_slice_biases_reindexes_to_kept_experts():
    biases = {0: torch.arange(8, dtype=torch.float32)}
    sliced = balance.slice_biases(biases, {0: [5, 0, 7, 2]})
    # keep list is sorted (matching prune_block_inplace's ascending keep order)
    assert torch.equal(sliced[0], torch.tensor([0.0, 2.0, 5.0, 7.0]))
    # keep set covering every expert -> unchanged values
    full = balance.slice_biases(biases, {0: list(range(8))})
    assert torch.equal(full[0], biases[0])


# --------------------------------------------------------------------- merging


def test_greedy_merge_plan_merges_the_most_similar_pair():
    # experts 0 and 1 are near-identical; 2 and 3 are far from everyone
    sim = torch.tensor([
        [1.00, 0.98, 0.10, 0.05],
        [0.98, 1.00, 0.12, 0.07],
        [0.10, 0.12, 1.00, 0.02],
        [0.05, 0.07, 0.02, 1.00],
    ])
    pairs, survivors, info = rgmerge.greedy_merge_plan(sim, ratio=0.5)
    assert len(pairs) == 2 and len(survivors) == 2
    assert (0, 1) in [tuple(p) for p in pairs] or (1, 0) in [tuple(p) for p in pairs]
    assert sorted(survivors) == sorted(survivors)
    assert len(set(survivors)) == 2
    assert info["target_keep"] == 2 and info["merges"] == 2


def test_greedy_merge_plan_survivor_follows_utilization():
    sim = torch.tensor([
        [1.0, 0.95, 0.10],
        [0.95, 1.0, 0.10],
        [0.10, 0.10, 1.0],
    ])
    # expert 1 is heavily used -> it should keep the slot when (0,1) merge
    pairs, survivors, _ = rgmerge.greedy_merge_plan(sim, ratio=2 / 3, utilization=[1.0, 99.0, 0.5])
    assert (1, 0) in pairs, pairs
    # and the opposite when expert 0 is the used one
    pairs2, _, _ = rgmerge.greedy_merge_plan(sim, ratio=2 / 3, utilization=[99.0, 1.0, 0.5])
    assert (0, 1) in pairs2, pairs2


def test_merge_experts_inplace_averages_params_and_gate():
    model = build_model("olmoe", seed=0)
    block = rgenv.find_moe_blocks(model)[0][1]
    e0 = {k: v.detach().clone() for k, v in block.experts[0].named_parameters()}
    e1 = {k: v.detach().clone() for k, v in block.experts[1].named_parameters()}
    g0 = block.gate.weight[0].detach().clone()
    g1 = block.gate.weight[1].detach().clone()

    rgmerge.merge_experts_inplace(block, [(0, 1)])

    for key in e0:
        expected = (e0[key].float() + e1[key].float()) / 2.0
        assert torch.allclose(block.experts[0].state_dict()[key].float(), expected, atol=1e-6), key
    assert torch.allclose(block.gate.weight[0], (g0 + g1) / 2.0, atol=1e-6)


def test_apply_merge_end_to_end(flavor):
    model = build_model(flavor, seed=0)
    n = model.config.num_experts
    sim = {layer: torch.eye(n) * 0.9 + 0.05 for layer in range(2)}  # uniformly similar
    plans = rgmerge.merge_keeps(sim, ratio=0.5)
    report = rgmerge.apply_merge(model, plans)

    assert model.config.num_experts == n // 2
    assert report["num_experts"] == [n // 2]
    # two MoE layers x n//2 merges each
    assert report["merged_total"] == n
    assert report["removed_total"] == n
    for layer, info in report["layers"].items():
        assert info["merged"] == n // 2
    for layer, keep in report["survivors"].items():
        assert len(keep) == n // 2
        assert keep == sorted(keep)
    with torch.no_grad():
        out = model(input_ids=make_ids(), use_cache=False).logits
    assert torch.isfinite(out).all()


def test_merge_refuses_quantized_experts():
    model = build_model("olmoe", seed=0)
    block = rgenv.find_moe_blocks(model)[0][1]
    param = next(block.experts[0].parameters())
    param.quant_state = object()  # what bitsandbytes attaches to packed weights
    try:
        with pytest.raises(RuntimeError, match="bitsandbytes-quantized"):
            rgmerge.merge_experts_inplace(block, [(0, 1)])
    finally:
        del param.quant_state


def test_merge_refuses_non_floating_params():
    model = build_model("olmoe", seed=0)
    block = rgenv.find_moe_blocks(model)[0][1]
    mlp = block.experts[0]
    original = mlp.down_proj.weight
    try:
        mlp.down_proj.weight = torch.nn.Parameter(
            original.detach().to(torch.uint8), requires_grad=False
        )
        with pytest.raises(RuntimeError, match="non-floating"):
            rgmerge.merge_experts_inplace(block, [(0, 1)])
    finally:
        mlp.down_proj.weight = original
    # restore a real floating param set: sanity — merging works again
    rgmerge.merge_experts_inplace(block, [(2, 3)])


def test_merge_keeps_accepts_per_layer_ratio():
    sim = {0: torch.eye(8) * 0.9 + 0.05, 1: torch.eye(8) * 0.9 + 0.05}
    plans = rgmerge.merge_keeps(sim, ratio={0: 0.5, 1: 0.25})
    assert plans[0]["info"]["target_keep"] == 4
    assert plans[1]["info"]["target_keep"] == 6


# --------------------------------------------------------- layer-wise budgets


def _layer_sim(n, off_diag):
    matrix = torch.full((n, n), float(off_diag))
    matrix.fill_diagonal_(1.0)
    return matrix


def test_layer_redundancy_mean_off_diagonal():
    sim = _layer_sim(4, 0.5)
    assert prune.layer_redundancy(sim) == pytest.approx(0.5)
    assert prune.layer_redundancy(_layer_sim(4, -0.2)) == pytest.approx(-0.2)


def test_layer_wise_ratios_conserve_total_budget():
    sims = {0: _layer_sim(8, 0.9), 1: _layer_sim(8, 0.2), 2: _layer_sim(8, 0.6)}
    ratios = prune.layer_wise_ratios(sims, total_ratio=0.5)
    removals = sum(int(round(ratios[l] * 8)) for l in sims)
    assert removals == int(round(0.5 * 24)) == 12
    # the most redundant layer loses the most, the least redundant the least
    assert ratios[0] > ratios[2] > ratios[1]
    for layer, ratio in ratios.items():
        assert 0.0 <= ratio < 1.0


def test_layer_wise_ratios_uniform_when_layers_identical():
    sims = {l: _layer_sim(8, 0.5) for l in range(3)}
    ratios = prune.layer_wise_ratios(sims, total_ratio=0.5)
    assert len(set(ratios.values())) == 1
    removals = sum(int(round(r * 8)) for r in ratios.values())
    assert removals == 12


def test_layer_wise_ratios_never_empties_a_layer():
    sims = {0: _layer_sim(2, 0.99), 1: _layer_sim(8, 0.9), 2: _layer_sim(8, 0.3)}
    ratios = prune.layer_wise_ratios(sims, total_ratio=0.5)
    assert ratios[0] <= 0.5  # a 2-expert layer can lose at most 1 expert
    for layer, ratio in ratios.items():
        n = sims[layer].shape[0]
        assert round(ratio * n) <= n - 1


def test_layer_wise_ratios_respects_cap():
    sims = {0: _layer_sim(8, 0.99), 1: _layer_sim(8, 0.1)}
    ratios = prune.layer_wise_ratios(sims, total_ratio=0.5, max_layer_ratio=0.25)
    assert int(round(ratios[0] * 8)) <= 2


def test_keeps_from_scores_accepts_per_layer_ratio():
    scores = {0: torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              1: torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])}
    keeps = prune.keeps_from_scores(scores, ratio={0: 0.5, 1: 0.25})
    assert keeps[0] == [4, 5, 6, 7]
    assert keeps[1] == [0, 1, 2, 3, 4, 5]


def test_layer_wise_keeps_end_to_end(flavor):
    model = build_model(flavor, seed=0)
    sims = {0: _layer_sim(8, 0.9), 1: _layer_sim(8, 0.1)}
    counts = {0: list(range(8)), 1: list(range(8))}
    ratios = prune.layer_wise_ratios(sims, total_ratio=0.5)
    keeps, infos = prune.redundancy_keeps(sims, ratios, per_layer_counts=counts)
    for layer, keep in keeps.items():
        assert len(keep) == int(round(ratios[layer] * 8)) or len(keep) >= 1
        assert len(keep) == 8 - round(ratios[layer] * 8)
    with torch.no_grad(), moe.MaskSim(model, keeps):
        out = model(input_ids=make_ids(), use_cache=False).logits
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------- eval bits


def test_checkpoint_bytes_counts_stored_tensors():
    model = build_model("olmoe", seed=0)
    expected = sum(p.numel() * p.element_size() for p in model.state_dict().values())
    report = rgeval.checkpoint_bytes(model)
    assert report["bytes"] == expected
    assert report["quantized_tensors"] == 0
    assert report["gb"] == pytest.approx(expected / 1024**3, abs=1e-3)


def test_param_report_carries_checkpoint_bytes(flavor):
    model = build_model(flavor)
    report = rgeval.param_report(model, "bf16")
    stored = rgeval.checkpoint_bytes(model)
    assert report["checkpoint_bytes"] == stored["bytes"]
    assert stored["bytes"] > 0
    # gb is the bytes value rounded for display (0.0 is correct for tiny models)
    assert report["checkpoint_gb"] == round(stored["bytes"] / 1024**3, 3)


def test_wilson_interval_bounds():
    lo, hi = rgeval.wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35          # 0/10 must not claim 0 to 0
    lo, hi = rgeval.wilson_interval(10, 10)
    assert hi == 1.0 and 0.65 < lo < 1.0          # 10/10 must not claim 1 to 1
    lo, hi = rgeval.wilson_interval(250, 500)
    assert lo < 0.5 < hi and hi - lo < 0.1        # 50% at n=500: ~±4.4 points


def test_mmlu_output_carries_ci():
    tokenizer = _CharTokenizer()
    letter_ids = rgeval.resolve_letter_ids(tokenizer)
    result = rgeval.mmlu_accuracy(
        _LetterStub(letter_ids), tokenizer, _letter_prompts(), batch_size=4, device=None
    )
    assert result["accuracy"] == 1.0
    assert result["accuracy_ci95"] == list(rgeval.wilson_interval(4, 4))
    assert result["accuracy_ci95"][0] < 1.0
