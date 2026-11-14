# RouterGuard

Post-training compression for sparse Mixture-of-Experts LLMs, in two coupled parts:

1. **Post-hoc load balancing** — fit a per-expert router bias on calibration data so routed tokens spread
   more evenly across experts, then freeze it. No weight updates, no retraining. Adapted from
   DeepSeek's Loss-Free Balancing (arXiv:2408.15664) from *training-time* to a single post-hoc pass.
2. **Redundancy-aware expert pruning** — rank routed experts by how *unique* their outputs are (primary)
   and how balanced their utilization is (secondary), then remove greedily with recomputation.

The claimed gap: prior expert-pruning work ranks experts by utilization or reconstruction error, never
under a *rebalanced* router and never using an explicit redundancy signal. Phrased as "to our
knowledge" — **this claim is not yet verified against the literature** (see [References](#references)).

Target model: Qwen1.5-MoE-A2.7B (14.3B total / 2.7B active, 60 routed + 1 shared expert, top-4).
Development model: OLMoE-1B-7B (6.9B total / 1.3B active, 64 experts, top-8).

Protocol **v2** adds three opt-in arms around the frozen v1 core (all registered before any real-model
Step 3/4 result existed): a gap-proportional bias update, a post-pruning bias refit, and merge-instead-of-remove.
The v2 arms run automatically in the CLI pipeline; the registered 2×2 ablation and every v1 gate are unchanged.

---

## Status

Code for Steps 1–4 **and the v2 extension arms** is complete and verified **on CPU with tiny models built
from the real `transformers` classes**, plus an end-to-end pipeline test. Only Step 1 has been run on a real
checkpoint so far.

| Step | Deliverable | Code | Verified how | Run on real model |
|---|---|---|---|---|
| 1 | env pinning, loaders, splits, load stats | ✅ | 12 preflight checks, 1 notebook | ✅ OLMoE, 8-bit, T4 |
| 2 | patched forward, mask sim, removal | ✅ | **bitwise no-op test, 8/8 configs** | ❌ |
| 3 | balancing, baselines, evaluation | ✅ | 78 tests + notebook-02 preflight | ❌ |
| 4 | similarity, greedy pruning, 2×2 | ✅ | 108 tests + notebook-03 preflight | ❌ |
| v2 | proportional update, post-prune refit, merge, layer budget, CLI pipeline | ✅ | 40 extension/pipeline tests + preflights 1–4 | ❌ |
| 5 | Qwen port + write-up | ❌ | — | — |

Test suite: **148 passed, 3 skipped** (the skips are opt-in real-checkpoint tests).
```bash
python -m pytest tests/ -q                        # 148 passed, 3 skipped
python tools/preflight.py        ~/routerguard    # structure, protocol, notebooks compile
python tools/selfcheck.py                         # stats/run bookkeeping, executed
python tools/preflight_step3.py                   # executes notebook 02's call sequence
python tools/preflight_step4.py                   # executes notebook 03's call sequence
```

### The only real-model number so far (Step 1, OLMoE, 8-bit, T4)

| Quantity | Value |
|---|---|
| calib `gini_mean` (200×1024, wikitext train) | **0.2710** |
| eval `gini_mean` (100×1024, wikitext test) | **0.2833** |
| mean KL(calib ‖ eval) per layer | **0.00816** nats (worst layer 15: 0.0160) |
| gate `gini_skip_balance = 0.20` | above it → balancing has headroom |

The small KL is the load-bearing result for the balancing half: the imbalance is a property of the
*model*, not of the sample, so a calibration-fitted bias has a reason to transfer. It also means the
headroom is modest — see [Interpretation limits](#interpretation-limits).

---

## What still needs doing

1. **Run the pipeline on a real checkpoint** (~35–55 min on a T4, one process instead of three notebooks):
   `python -m routerguard.cli pipeline --tag t4_full`. This executes the same sequence as notebooks
   02+03 — load stats, bias fit, both Step-3 guards, similarity (with the KL<1e-2 under-bias rule),
   the 2×N MaskSim ablation, the `min_gain_over_frequency_pts` gate, then materializes the headline arm
   and evaluates it (ppl/MMLU/throughput/memory/checkpoint bytes), then runs the post-prune refit arm.
   **Gate:** does the bias transfer, does it cost accuracy, does redundancy beat frequency?
2. **Step 5 — notebook 04 (Qwen port).** Only after 1, because it needs the winning configuration.
   Timing probe first (2 layers, project total, abort over `abort_if_projected_over_seconds`).
   Re-measure everything at 4-bit and never compare against published bf16 numbers.
3. **`README` results section** with the final table: memory saved (headline, from
   `checkpoint_bytes`, not allocator counters), ppl delta, MMLU delta **with its Wilson interval**,
   tok/s with variance.
4. **Verify the citations** below against the actual PDFs before any of them enter a write-up.
5. **Decide the paper's framing** before seeing Step 4: with baseline Gini 0.27 the load-balancing win
   will be small, so the defensible claim is about the *interaction* (does a rebalanced router change
   which experts are redundant, and therefore which are prunable), not about large balancing gains.

---

## Layout

```
configs/protocol.yaml          frozen protocol (v2): splits, ratios, seeds, gates, budget caps,
                               update_mode, extension arms
routerguard/
  env.py                       version pinning, MoE-structure assertions, env dump
  models.py                    OLMoE (bf16/8-bit/4-bit), Qwen (NF4), gate-precision asserts
  data.py                      deterministic wikitext splits, MMLU few-shot prompts
  stats.py                     load counts, Gini, max-violation, KL, hook collector
  moe.py                       patched forward (selection-only bias), mask sim, real removal
  balance.py                   LFB loop (sign + proportional update), KL guard, transfer ratio,
                               bias re-indexing after pruning
  similarity.py                mean-centered cosine per layer, cached/resumable
  prune.py                     baselines + greedy redundancy removal + layer-wise budgets
  merge.py                     merge redundant expert pairs instead of deleting them (v2)
  eval.py                      perplexity, MMLU (+ Wilson CI), memory, checkpoint bytes, throughput
  pipeline.py                  notebook-accurate Steps 1-4 (+ v2 arms) in one process
  cli.py                       python -m routerguard.cli {env,stats,balance,similarity,pipeline}
  run.py                       run ids, resolved config, git sha, metrics files
notebooks/01..03               the original runbook (Step 4's notebook is 03; 04 = Qwen, not written)
tests/                         148 tests, tiny models from real transformers classes
tools/                         preflight + self-check scripts
```

---

## Method, precisely

**Selection-only bias.** Selection uses `topk(s + b)`; the mixing weights still come from the *unbiased*
`s`. A plain logit hook cannot express that — it would bias the mixing weights too — so `moe.py` replaces
each block's `forward` with a mirror of `transformers==4.48.3`'s own loop and changes only the selection
line. This is the single most important implementation detail in the repo.

**Bias update** (per layer, per calibration batch, `c` = cumulative load, `c̄` = its mean):

```
b_i ← b_i + u · sign(c̄ − c_i)
```

frozen afterwards; weights never change.

**Removal semantics** (`protocol.yaml: pruning.removal_semantics: reslice`). The pruned model is the
native one: router rows are sliced, so softmax runs over kept experts only and `top_k` clamps to the kept
count. Mask simulation selects the kept router columns with `index_select` so it sees exactly the same
tensor — which is why mask-sim ≡ real removal.

**Redundancy.** Per layer, every expert runs on the *same* tokens (the hidden states entering that block),
outputs are mean-centered across experts per token, and compared by cosine:

```
z_{i,t} = f_i(x_t) − mean_j f_j(x_t)     S_ij = ⟨z_i, z_j⟩ / (‖z_i‖‖z_j‖)
```

**Removal is greedy**: score each surviving expert by mean similarity to the others (higher = more redundant),
plus a 0.25-weighted utilization term, remove the worst, **recompute**.

---

## Extension arms (protocol v2)

Three opt-in arms wrap the frozen core. Each is registered in `protocol.yaml` (`extensions:`) and runs
from the CLI pipeline; none changes the v1 gates or the 2×2 ablation.

**Gap-proportional bias update** (`balance.update_mode: proportional`). The sign update steps at full
magnitude `u` every batch no matter how close the load is to uniform, which is exactly why it has the
overshoot boundary documented below. The proportional variant steps by the *relative* gap:

```
b_i ← b_i + u · (c̄ − c_i) / (c̄ + eps)
```

A dead expert still gets the full `u`, an expert 2× over-loaded gets `−u`, and one 1% off target gets
≈0.01·u — so the update vanishes as balance converges instead of oscillating at full magnitude. Same
`u` meaning (per-batch step cap), same selection-only semantics, same guards. The test
`test_proportional_fit_reduces_gini_without_overshoot` pins the contrast: at `u = 0.05` (where the sign
update measurably *raises* Gini on a skewed tiny model), proportional still lowers it.

**Post-pruning bias refit** (`extensions.post_prune_refit`, on by default). Removing experts changes the
load distribution, so the bias fitted before pruning no longer matches the pruned router. The pipeline
therefore runs a second LFB calibration pass on the pruned model — nearly free, same loop, same guards —
and reports the ppl delta against the pre-pruning bias *re-applied* to the pruned router (it re-indexes
that bias to the surviving experts via `balance.slice_biases`: the pruned gate's row `j` is original
expert `keep[j]`). This directly combines the two halves of the method.

**Merge instead of remove** (`merge.py`, `--merge`). The similarity matrix already identifies redundant
*pairs*; deletion throws the pair's combined capacity away, so `merge.py` averages it: pick the most
similar surviving pair (same `utilization_weight` semantics as the greedy pruner), average the two
experts' parameters and gate rows, hand the slot to the more-used expert, and slice the loser out with
the same native-removal machinery. The fitted selection bias is folded through each merge (the merged
gate row is the pair's average, so its bias is too) and re-indexed to the survivors, keeping the merged
run under the same balanced-router condition as the deletion cells it is compared against. Two honest
limitations, both enforced or labeled in code: merging changes weights, so **MaskSim cannot preview it**
(its ppl is measured on real merged weights, and every merge-vs-delete comparison in the metrics records
its provenance); and averaging packed bitsandbytes weights is meaningless, so **merging refuses
quantized experts** and needs a bf16/fp16 load.

**Layer-wise removal budgets** (`prune.layer_wise_ratios`, `--layer-budget`). Redundancy is not uniform
across depth, so instead of a uniform 25%/50% everywhere, the total removal budget is distributed across
layers in proportion to each layer's mean off-diagonal cosine (largest-remainder allocation, so the total
is conserved exactly; every layer keeps ≥1 expert; optional per-layer cap). Applies to the redundancy
ranking only — random and frequency stay uniform so the baseline comparison remains apples-to-apples.

---

## Verification, and what it actually guarantees

| Claim | Evidence |
|---|---|
| No-op patch is bitwise identical to the unpatched model | `torch.equal` in 8/8 configs (fp32/bf16 × `norm_topk_prob` on/off × OLMoE/Qwen) |
| Mask simulation ≡ real removal | selection identical exactly; outputs **bitwise in bf16**, ≤3e-08 abs in fp32 (pruning changes the router GEMM shape) |
| Bias is selection-only, not a weighting change | test also proves the naive "bias everything" variant differs |
| Fused v5 expert layout is rejected with an actionable error | `env.assert_moe_structure` on a synthetic fused model |
| A broken dependency environment fails loudly | `env.assert_env` rejects `huggingface-hub>=1.0`, `datasets>=5`, `transformers 5.x` |
| Proportional update does not overshoot where sign does | at `u=0.05` on a skewed tiny model: sign raises Gini, proportional lowers it (`test_proportional_fit_reduces_gini_without_overshoot`) |
| Merging averages exactly, and refuses what it cannot average | params/gate rows = mean of the pair (`test_merge_experts_inplace_averages_params_and_gate`); quantized/non-floating experts raise (`test_merge_refuses_*`) |
| Layer budgets conserve the total removal count | largest-remainder allocation asserted in 3 layouts (`test_layer_wise_ratios_*`) |
| The pipeline writes a complete run directory | end-to-end tiny-model run: 14 metrics JSONs, biases, env dump (`test_run_pipeline_end_to_end`) |

---

## Interpretation limits (read before writing anything up)

- **Modest headroom.** Baseline Gini 0.27 vs the 0.20 skip threshold. If LFB's published
  0.707 → 0.035 holds, our starting point is far more balanced, so absolute balancing gains here will be
  small. Do not present this as a large load-balancing win.
- **Sign-update overshoot.** The sign update has a real instability boundary: measured on a skewed model,
  `u ≤ 0.01` reduces Gini (0.098 → 0.042) while `u ≥ 0.02` *increases* it (0.098 → 0.18), because
  `u × n_updates` grows past the router logit scale. Protocol `u = 0.001` is safely inside; a run that
  lands outside emits a warning (`BalanceResult.warnings()`). The v2 proportional update removes this
  failure mode (its step shrinks with the gap); `sign` remains the registered default.
- **Layer 0 similarity is router-invariant** (its inputs are the embeddings), so recomputing similarity
  under a balanced router only matters for deeper layers; notebook 03 skips it when routing KL < 1e-2.
- **Precision.** Qwen is 4-bit only on a T4, so all Qwen numbers are pruned-vs-unpruned *at 4-bit*. Never
  compare them to published bf16 figures.
- **MMLU here is not lm-eval-harness.** Letters are scored as single tokens from the final position with
  a custom 3-shot prompt template, on a 500-question subset with a fixed seed. Comparable internally,
  not comparable to published MMLU numbers.
- **Two seeds** (42, 123) is thin for variance claims; MMLU at 500 questions has roughly ±2 points of noise.
- **Protocol corrections made before any Step 3/4 results existed** (both recorded in the YAML):
  `budget.max_greedy_steps` 16 → 64 (16 cannot express a 50% removal: 30 removals of 60 experts), and
  `design.utilization_weight: 0.25` added because that method parameter was not pinned at all.

---

## How to run

### Preferred: the CLI (one process, model loaded once)

Colab (GPU) is the intended environment; a T4 is enough. After the install cell below
(or the notebook-01 install cell), from the repo root:

```bash
python -m routerguard.cli env                          # pins + GPU + structure checks, no download
python -m routerguard.cli pipeline --tag t4_full       # Steps 1-4 + post-prune refit, one process
python -m routerguard.cli pipeline --update-mode proportional   # adaptive bias update
python -m routerguard.cli pipeline --merge             # merge arm (needs bf16 experts, reloads per ratio)
python -m routerguard.cli pipeline --layer-budget      # redundancy removals allocated by layer
python -m routerguard.cli balance --model qwen         # Step 3 only
python -m routerguard.cli similarity --fast            # Step 4a only, capped tokens
python -m routerguard.cli pipeline --fast              # capped smoke run (NOT protocol numbers)
```

Useful flags: `--model {olmoe,qwen}`, `--precision {auto,bf16,8bit,4bit}`, `--device {cuda,cpu}`,
`--ratios 0.25,0.5`, `--seeds 42,123`, `--no-mmlu`, `--no-throughput`, `--results-dir DIR`,
`--override balance.default_u=0.002` (repeatable dotted protocol overrides).

Every command writes `results/<timestamp>_<sha>_<tag>/` with `config.resolved.yaml`, `git_sha.txt`,
`env.txt`, and `metrics/*.json` — the `summary.json` carries the headline numbers (memory saved from
checkpoint bytes, ppl deltas, gate table). `--fast` caps tokens so a configuration mistake costs
minutes, not an hour; its numbers must never be reported.

On Colab, point results at Drive so a runtime disconnect doesn't wipe them:

```bash
!mkdir -p /content/drive/MyDrive/routerguard-results
!ln -sfn /content/drive/MyDrive/routerguard-results results
```

### Notebook path (unchanged)

Clone, link Drive, then run `notebooks/01 → 02 → 03` top to bottom:

```python
%cd /content
!git clone https://github.com/ishwar170695/routerguard.git
%cd /content/routerguard
!ln -sfn /content/drive/MyDrive/routerguard-results results    # /content is wiped on disconnect
```

### The install cell (mandatory once per fresh runtime, both paths)

```bash
pip install 'transformers==4.48.3' 'huggingface-hub>=0.24,<1.0' 'datasets>=2.19,<5' \
            accelerate 'bitsandbytes>=0.43.3' pyyaml matplotlib sentencepiece
```

The `huggingface-hub` and `datasets` bounds are **not optional**: a newer `datasets` upgrades
`huggingface-hub` past 1.0 as a side effect, after which `transformers==4.48.3` cannot be imported at
all. `env.assert_env()` enforces the same range.

`transformers` must stay on 4.48.3 (or a logged fallback): v5 fuses experts into 3D tensors, which breaks
per-expert hooks and bitsandbytes expert quantization (transformers issue #1849).

### Local CPU tests (no GPU, no downloads)

The full test suite runs on CPU against tiny models built from the real `transformers` classes:

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install 'transformers==4.48.3' 'huggingface-hub>=0.24,<1.0' \
                     'datasets>=2.19,<5' numpy pyyaml pytest
.venv/bin/python -m pytest tests/ -q            # 148 passed, 3 skipped
```

`tools/preflight_step3.py` / `preflight_step4.py` expect the repo at `~/routerguard` (clone it there,
or run with `HOME` pointed at a directory containing a symlink).

---

## References

⚠️ All of these are cited **second-hand from the project handoff and have not been checked against the
PDFs**. Verify titles, venues, and every number quoted from them before they enter a document.

| Paper | arXiv | Role in this project | Verified |
|---|---|---|---|
| Loss-Free Balancing (DeepSeek) | 2408.15664 | source of the bias update in `balance.py` | ❌ |
| Expert pruning / trimming (TMLR 2025 per handoff) | 2402.14800 | the prior work whose gap this project claims | ❌ |
| DeepSeekMoE | 2401.06066 | background: fine-grained routed experts + always-on shared expert | ❌ |
| OLMoE | 2409.02060 | development model | ❌ |
| Qwen2 technical report | 2407.10671 | target model family | ❌ |
| Switch Transformer | 2101.03961 | background: auxiliary-loss balancing that LFB replaces | ❌ |

Numbers still needing verification: LFB's reported Gini 0.707 → 0.035, and the TMLR paper's claimed 98%
retention.

---

## Possible improvements

**Done in protocol v2** (see [Extension arms](#extension-arms-protocol-v2) for details and limits):
- ✅ Replace the sign update with a proportional/adaptive one — `balance.update_mode: proportional`.
- ✅ Refit the biases after pruning — the default refit arm; reports the ppl delta of the second pass.
- ✅ Merge instead of remove — `merge.py`; opt-in (`--merge`), unquantized experts only.
- ✅ Layer-wise removal budgets — `prune.layer_wise_ratios`; opt-in (`--layer-budget`).
- ✅ Measure **saved checkpoint bytes on disk**, not only allocator counters — `eval.checkpoint_bytes`,
      headline in `summary.json`.
- ✅ Report MMLU noise explicitly — Wilson 95% interval on every MMLU number.
- ✅ A small CLI (`python -m routerguard.cli`) — runs scriptable, single process, diffable in git.

**Method**
- **Alternative redundancy metrics**: linear CKA, or reconstruction error of one expert from the others,
  which is closer to what the trimming literature optimizes and would make the comparison fairer.
- **Calibration-set domain transfer**: does a bias fitted on code transfer to prose?

**Evaluation**
- More seeds and confidence intervals; two seeds (42, 123) is still thin for variance claims beyond the
  Wilson band MMLU now reports.
- Task breadth: GSM8K, HellaSwag.
- Throughput under batching, not just batch-1 tokens/sec — the practical win may differ.
- A bf16 confirmation run for Qwen on a larger GPU, to separate quantization noise from pruning damage.

**Engineering**
- Real-checkpoint versions of the two bitwise tests are written but opt-in
  (`ROUTERGUARD_REAL_MODELS=1`); they should be part of the default GPU run — `cli env` runs the cheap
  parts of this (structure + gate precision) on every real run already.
