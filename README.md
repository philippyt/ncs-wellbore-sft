# Wellbore SFT and RLVR

Long context structured extraction SFT, RLVR GRPO training, and deterministic FactPages evaluation built directly on the NCS wellbore completion Parquet files with zero RAG.

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
