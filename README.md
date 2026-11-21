# Wellbore SFT and RLVR

Long context structured extraction SFT, RLVR GRPO training, and deterministic FactPages evaluation built directly on the NCS wellbore completion Parquet files with zero RAG.

## Benchmark Results on `eval_exploration_recent` (n = 180)

All runs evaluated on the exact 180 task `eval_exploration_recent.jsonl` split with deterministic unit, datum, and tolerance scoring.

| Model | Mode | Header Acc | Casing F1 | LOT Recall (n=150) | Formation Tops F1 | DST F1 (n=24) | RLVR Reward | Valid Schema | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen/qwen3-235b-a22b-2507` | Closed book | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.102 | 179/180 | 0.075 |
| Regex baseline | Open book | 0.374 | 0.278 | 0.091 | 0.184 | 0.000 | 0.363 | 180/180 | 0.000 |
| `qwen/qwen3-30b-a3b-instruct-2507` | Open routed | 0.881 | 0.758 | 0.651 | 0.347 | 0.611 | 0.668 | 179/180 | 0.376 |
| `qwen/qwen3.7-flash` | Open full (600k chars) | 0.868 | 0.742 | **0.694** | 0.406 | 0.594 | 0.716 | 170/180 | 3.758 |
| `qwen/qwen3-235b-a22b-2507` | Open routed | **0.917** | **0.772** | 0.678 | 0.353 | **0.677** | 0.718 | 180/180 | 0.566 |
| `deepseek/deepseek-v3.2` | Open routed | 0.853 | 0.769 | 0.680 | **0.432** | 0.635 | **0.727** | 172/180 | 0.926 |

Key findings from the baseline suite:

1. **Zero FactPages memorization**: `qwen/qwen3-235b-a22b-2507` closed book scores `0.007` header accuracy and `0.000` on all tables, confirming that open book accuracy reflects genuine document reading.
2. **Routed packing matches full context at 7.5x lower token volume**: Deterministic 28 page packing (`3.5M` prompt tokens) matches or outperforms 600k char full document packing (`26.5M` prompt tokens) on header, casing, DST, and overall RLVR reward at a fraction of the cost.
3. **Clear SFT and RLVR headroom**: Formation tops F1 plateaus at `0.35` to `0.43` even on frontier open weight models and full document context, leaving roughly `+0.35` to `+0.40` F1 headroom for SFT and GRPO post training.

## Workflow Overview

1. **Parquet dataset preparation (`build-data`)**: Reads `documents.parquet`, `pages/*.parquet`, `page_flags.parquet`, `casing_and_lot.parquet`, `formation_tops.parquet`, and `dst.parquet` from the `wellbore-completion` release directory. Filters strictly to `split_eval == "train"`, `is_canonical == True`, and `factpages_join_type == "exact"`, drops `is_ocr_loop` pages, packs candidate pages deterministically, and writes licence grouped `sft_train.jsonl`, `sft_val.jsonl`, `rlvr_train.jsonl`, and `rlvr_val.jsonl`.
2. **Open book and closed book LLM evaluation (`run-eval`)**: Evaluates any OpenAI compatible endpoint (vLLM, SGLang, OpenRouter, Together, Ollama) on `eval_exploration_recent.jsonl` in both `open_book` (packed OCR Markdown) and `closed_book` (metadata only memorization control) modes, then scores with exact unit, datum, and tolerance checks.
3. **SFT warm start (`train-sft`)**: LoRA or QLoRA supervised fine tuning with TRL `SFTTrainer` on `sft_train.jsonl`.
4. **RLVR post training (`train-grpo`)**: Group Relative Policy Optimization with TRL `GRPOTrainer` using `compute_rlvr_reward` as a deterministic programmatic verifier over JSON validity, schema compliance, header accuracy, casing F1, conditional LOT recall, formation tops F1, DST interval F1, and an anti bloat penalty.

## Setup

Core data preparation, evaluation, and scoring:

```bash
uv sync --dev
```

GPU training extras (`torch`, `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes`):

```bash
uv sync --extra train
```

Point to a custom Parquet release directory via `WELLBORE_DATA_DIR` or `--data-dir`. Default points to `C:\Users\philip\Documents\wellbore-completion\output\release_v2`.

## Commands

| Command | Purpose |
| --- | --- |
| `uv run wellbore-sft build-data` | Build leakage free SFT and RLVR JSONL datasets from Parquet |
| `uv run wellbore-sft reward-check` | Verify oracle and regex baseline RLVR rewards on `eval_exploration_recent` |
| `uv run wellbore-sft run-eval --mode closed_book --model <id>` | Run closed book memorization baseline |
| `uv run wellbore-sft run-eval --mode open_book --model <id>` | Run open book extraction evaluation and score |
| `uv run wellbore-sft score <tasks.jsonl> <preds.jsonl>` | Score any predictions file with 95% bootstrap and Wilson CIs |
| `uv run wellbore-sft train-sft --model Qwen/Qwen2.5-7B-Instruct` | Run LoRA SFT warm start |
| `uv run wellbore-sft train-grpo --model checkpoints/sft/final` | Run deterministic RLVR GRPO post training |

## Verification

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
```
