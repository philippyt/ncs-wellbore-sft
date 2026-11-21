# NCS Wellbore SFT and RLVR

Long context structured extraction SFT, RLVR post training, and deterministic FactPages evaluation built directly on the NCS wellbore completion Parquet files from https://huggingface.co/datasets/philippyt/ncs-wellbore-completion.

## Research Focus: Recurrent Core Depth, OOD Basin Transfer, and Grounded RLVR (`thinkingmachines/Inkling-2-Mini`)

This repository studies how **latent recurrent depth** (`loop_num_passes = 1, 2, 3, 4` through the looped core of `Inkling-2-Mini`, 26B total / 5.4B active, with `reasoning_effort="none"` to isolate latent recurrence from explicit chain of thought) interacts with **SFT** and **document grounded RLVR** under real physical distribution shift:

1. **Out of distribution geological shift:** Train strictly on North Sea wellbores (`wlbMainArea == "NORTH SEA"`, `693` train / `106` licence isolated val wells) and evaluate on held out North Sea wells (`108` in `eval_exploration_recent`) vs. zero shot OOD Norwegian Sea and Barents Sea wells (`72` wells, where `69.7%` of formation tops never appear in North Sea training).
2. **Document grounded vs. noisy RLVR verifier:** FactPages ground truth disagrees with single document OCR on ~18% of formation tops due to revisions and sibling reports. We compare standard FactPages RLVR against document grounded RLVR (masking ungrounded ground truth rows and penalizing hallucinated depths absent from the OCR pages).
3. **LoRA adaptation of attention and recurrent loop gates:** Adapts `q_proj.linear`, `k_proj.linear`, `v_proj.linear`, `out_proj`, `forget_gate_proj`, and `message_gate_proj` inside `Inkling2LoopTextModel` using chunked exact attention with sinks and completion only loss.

## Reference Baselines on `eval_exploration_recent` (n = 180)

All reference runs evaluated on the 180 task `eval_exploration_recent.jsonl` split with deterministic unit, datum, and tolerance scoring.

| Model | Mode | Header Acc | Casing F1 | LOT Recall (n=150) | Formation Tops F1 | DST F1 (n=24) | RLVR Reward | Valid Schema | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen/qwen3-235b-a22b-2507` | Closed book | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.102 | 179/180 | 0.075 |
| `qwen/qwen3.5-9b` | Open routed | 0.296 | 0.300 | 0.224 | 0.138 | 0.354 | 0.351 | 73/180 | 0.759 |
| Regex baseline | Open book | 0.374 | 0.278 | 0.091 | 0.184 | 0.000 | 0.363 | 180/180 | 0.000 |
| `qwen/qwen3-30b-a3b-instruct-2507` | Open routed | 0.881 | 0.758 | 0.651 | 0.347 | 0.611 | 0.668 | 179/180 | 0.376 |
| `qwen/qwen3.7-flash` | Open full (600k chars) | 0.868 | 0.742 | **0.694** | 0.406 | 0.594 | 0.716 | 170/180 | 3.758 |
| `qwen/qwen3-235b-a22b-2507` | Open routed | **0.917** | **0.772** | 0.678 | 0.353 | **0.677** | 0.718 | 180/180 | 0.566 |
| `deepseek/deepseek-v3.2` | Open routed | 0.853 | 0.769 | 0.680 | **0.432** | 0.635 | **0.727** | 172/180 | 0.926 |

Key baseline findings:

1. **Zero FactPages memorization:** `qwen/qwen3-235b-a22b-2507` closed book scores `0.007` header accuracy and `0.000` on all tables, confirming that open book accuracy reflects genuine document reading.
2. **Routed packing matches full context at 7.5x lower token volume:** Deterministic page routing (`3.5M` prompt tokens) matches or outperforms 600k char full document packing (`26.5M` prompt tokens) on header, casing, DST, and overall RLVR reward at a fraction of the cost.
3. **Formation tops error analysis:** Even on `deepseek/deepseek-v3.2`, `GROUP` recall is only `23.3%` vs. `46.7%` on `FORMATION`, and `646` predictions hit the exact right depth (within `2 m`) with a non canonical name.

## Setup

Core data preparation, evaluation, and scoring:

```bash
uv sync --dev
```

GPU training extras (`torch`, `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes`):

```bash
uv sync --extra train
```

Point to your local Parquet release directory via `WELLBORE_DATA_DIR` or `--data-dir`.

## Commands

| Command | Purpose |
| --- | --- |
| `uv run wellbore-sft build-data --basin-filter "NORTH SEA"` | Build licence isolated SFT and RLVR JSONL datasets from Parquet |
| `uv run wellbore-sft reward-check` | Verify oracle and regex baseline RLVR rewards on `eval_exploration_recent` |
| `uv run wellbore-sft inkling-eval --passes 1,2,3,4` | Sweep `Inkling-2-Mini` recurrent core depth on In Basin vs. OOD Basin splits |
| `uv run wellbore-sft inkling-sft` | Run `Inkling-2-Mini` LoRA SFT on attention and recurrent loop gates |
| `uv run wellbore-sft inkling-rlvr --grounded` | Run document grounded on policy RLVR on `Inkling-2-Mini` |
| `uv run wellbore-sft run-eval --mode open_book --model <id>` | Evaluate any OpenAI compatible endpoint on `eval_exploration_recent` |
| `uv run wellbore-sft score <tasks.jsonl> <preds.jsonl>` | Score any predictions file with 95% bootstrap and Wilson CIs |
| `uv run wellbore-sft train-sft` | Generic TRL `SFTTrainer` LoRA warm start |
| `uv run wellbore-sft train-grpo` | Generic TRL `GRPOTrainer` RLVR post training |
| `uv run wellbore-sft merge-lora` | Merge a saved LoRA adapter into its base model |

## Verification

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
```
