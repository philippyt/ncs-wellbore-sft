# NCS Wellbore SFT and RLVR

Long context structured extraction SFT, RLVR post training, and deterministic FactPages evaluation built directly on the NCS wellbore completion Parquet dataset (`https://huggingface.co/datasets/philippyt/ncs-wellbore-completion`).

## `Inkling-2-Mini` Experiments (`eval_stratified_20`, n = 20)

All `Inkling-2-Mini` (`thinkingmachines/Inkling-2-Mini`, 26.2B total / 3.4B active / 5.4B effective active at 3 recurrent passes, `reasoning_effort="none"`) evaluations use 10 North Sea wells (4 seen licence, 6 unseen licence) and 10 zero shot OOD Norwegian Sea and Barents Sea wells (`max_pages=16`, `max_chars=32,000`, `max_prompt_tokens=6,656`).

### 1. Recurrent Core Depth Lock (`max_new_tokens=1536`, `batch_size=10`)

| Model | Pass | Valid Schema | Real Extractions | Header Acc | Casing F1 | LOT Recall | Tops F1 | DST F1 (n=4) | RLVR Reward | Paired Delta vs. `pass=3` (95% CI) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Base | 1 | 0/20 | 0/20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | +0.4356 [+0.2960, +0.5756] |
| Base | 2 | 0/20 | 0/20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | +0.4356 [+0.2960, +0.5756] |
| Base | 3 | 14/20 | 14/20 | 0.488 | 0.473 | 0.256 | 0.262 | 0.375 | 0.4356 | reference (`0.6195` on valid) |
| Base | 4 | 5/20 | 0/20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0345 | +0.4011 [+0.2544, +0.5518] |
| Arm 2A (`pass=3` SFT) | 2 | 0/20 | 0/20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | depth lock unchanged |
| Arm 2A (`pass=3` SFT) | 4 | 7/20 | 0/20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0500 | empty JSON credit only |

Single well qualitative diagnostic (`2/7-31`, Base):

| Pass | Final Hidden L2 Norm | Observed Failure Mode |
| ---: | ---: | :--- |
| 1 | 197.4 | Echoes input `[PAGE n]` markers |
| 2 | 202.3 | Extracts real water depth (`239 ft` / `72.5 m`), then breaks JSON syntax |
| 3 | 275.2 | Valid JSON extraction |
| 4 | 155.0 | Hidden state shrinks; emits empty `":"` placeholders |

### 2. Prompt Tokenization and Grounding Ceiling (`max_prompt_tokens=6,656`)

OCR markdown tables tokenize at ~2.5 to 3.0 chars/token (`20/20` eval prompts clipped, max `13,461` tokens; `582/671` train prompts clipped, P50 `9,735`, P90 `12,308`). Eval Formation Tops has an `~0.80` recall ceiling under this budget.

| Split | Field | Total | In 32k Chars | In 6,656 Tokens | Absent from 6,656 Prompt |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Eval (`n=20`) | Casing | 84 | 81 (96.4%) | 73 (86.9%) | 11 (13.1%) |
| Eval (`n=20`) | Formation Tops | 400 | 332 (83.0%) | 321 (80.2%) | 79 (19.8%) |
| Eval (`n=20`) | DST | 8 | 8 (100.0%) | 8 (100.0%) | 0 (0.0%) |
| Train (`n=671`) | Casing | 3,015 | n/a | 2,193 (72.7%) | 822 (27.3%) |
| Train (`n=671`) | LOT | 1,069 | n/a | 836 (78.2%) | 233 (21.8%) |
| Train (`n=671`) | Formation Tops | 14,047 | n/a | 8,590 (61.2%) | 5,457 (38.8%) |
| Train (`n=671`) | DST | 613 | n/a | 303 (49.4%) | 310 (50.6%) |

### 3. Matched `pass=3` Base vs. North Sea LoRA SFT (`max_new_tokens=3072`, `batch_size=5`)

Removing the `1536` token cap raised Base `pass=3` from `14/20` (`0.4356`) to `18/20` (`0.5857`); `0/20` outputs hit `3072` in either run. Arm 2A trained `45` steps (`180` North Sea wells, `lora_r=32`, attention + recurrent gates) on unfiltered targets.

| Metric (`pass=3`, `3072` tok, `bs=5`) | Base | Arm 2A (Unfiltered SFT) | Delta (`Arm 2A` vs. `Base`) |
| :--- | ---: | ---: | :--- |
| Valid Schema | 18/20 (90.0%) | 19/20 (95.0%) | +1 well |
| Overall RLVR Reward | 0.5857 | 0.6380 | +0.0523 (95% CI [+0.0181, +0.0871]) |
| Prompt Grounded Reward (6,656 tok) | 0.6090 | 0.6441 | +0.0350 (95% CI [-0.0183, +0.0834]) |
| Conditional on Valid Reward | 0.6486 (n=18) | 0.6716 (n=19) | +0.0230 |
| Header Accuracy | 0.6125 | 0.7625 | +0.1500 |
| Casing F1 (full / prompt grounded) | 0.6733 / 0.7233 | 0.6729 / 0.6729 | -0.0004 / -0.0504 |
| LOT Recall (full / prompt grounded) | 0.4778 / 0.5513 | 0.4333 / 0.5000 | -0.0445 / -0.0513 |
| Formation Tops F1 (full / prompt grounded) | 0.3807 / 0.4040 | 0.3579 / 0.3820 | -0.0228 / -0.0220 |
| DST Non-Empty (exact / mean F1) | 2 of 4 (0.6250) | 1 of 4 (0.5417) | -1 well (-0.0833) |
| Ungrounded Depth Rate (valid) | 0.0041 | 0.0000 | -0.0041 |

Header per field breakdown (`pass=3`, `3072` tok, `bs=5`):

| Header Field | Base | Arm 2A | Delta |
| :--- | ---: | ---: | ---: |
| `kelly_bushing_m` | 2/20 (10.0%) | 13/20 (65.0%) | +11 wells |
| `water_depth_m` | 13/20 (65.0%) | 14/20 (70.0%) | +1 well |
| `total_depth_md_rkb_m` | 18/20 (90.0%) | 19/20 (95.0%) | +1 well |
| `final_vertical_depth_tvd_rkb_m` | 16/20 (80.0%) | 15/20 (75.0%) | -1 well |

3-way basin and licence split (`rlvr_reward`, `pass=3`, `3072` tok, `bs=5`):

| Subset | n | Base | Arm 2A |
| :--- | ---: | ---: | ---: |
| North Sea, seen licence | 4 | 0.3581 (2/4 valid) | 0.4226 (3/4 valid) |
| North Sea, unseen licence | 6 | 0.5971 (6/6 valid) | 0.6426 (6/6 valid) |
| OOD Norwegian and Barents Sea | 10 | 0.6699 (10/10 valid) | 0.7214 (10/10 valid) |

Open next steps: (1) test whether a one line `kelly_bushing_m` (`RKB`/`RT`/`KB`) prompt hint recovers the header gain on Base without fine tuning, (2) rerun Arm 2A and elastic depth Arm 2B (`pass ~ {2,3,4}`) on `6,656` token grounded targets, and (3) check batch size sensitivity (`bs=5` vs. `bs=4`).

## Reference Baselines on `eval_exploration_recent` (n = 180)

| Model | Mode | Header Acc | Casing F1 | LOT Recall (n=150) | Tops F1 | DST F1 (n=24) | RLVR Reward | Valid Schema | Cost (USD) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen/qwen3-235b-a22b-2507` | Closed book | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.102 | 179/180 | 0.075 |
| `qwen/qwen3.5-9b` | Open routed | 0.296 | 0.300 | 0.224 | 0.138 | 0.354 | 0.351 | 73/180 | 0.759 |
| Regex baseline | Open book | 0.374 | 0.278 | 0.091 | 0.184 | 0.000 | 0.363 | 180/180 | 0.000 |
| `qwen/qwen3-30b-a3b-instruct-2507` | Open routed | 0.881 | 0.758 | 0.651 | 0.347 | 0.611 | 0.668 | 179/180 | 0.376 |
| `qwen/qwen3.7-flash` | Open full (600k) | 0.868 | 0.742 | **0.694** | 0.406 | 0.594 | 0.716 | 170/180 | 3.758 |
| `qwen/qwen3-235b-a22b-2507` | Open routed | **0.917** | **0.772** | 0.678 | 0.353 | **0.677** | 0.718 | 180/180 | 0.566 |
| `deepseek/deepseek-v3.2` | Open routed | 0.853 | 0.769 | 0.680 | **0.432** | 0.635 | **0.727** | 172/180 | 0.926 |

## Setup

```bash
uv sync --dev
uv sync --extra train
```

Point to your local Parquet release directory via `WELLBORE_DATA_DIR` or `--data-dir`.

## Commands

| Command | Purpose |
| :--- | :--- |
| `uv run wellbore-sft build-data --basin-filter "NORTH SEA"` | Build licence isolated SFT and RLVR JSONL datasets from Parquet |
| `uv run wellbore-sft reward-check` | Verify oracle and regex baseline RLVR rewards on `eval_exploration_recent` |
| `uv run wellbore-sft inkling-eval --passes 1,2,3,4` | Sweep `Inkling-2-Mini` recurrent core depth on In Basin vs. OOD Basin splits |
| `uv run wellbore-sft inkling-sft` | Run `Inkling-2-Mini` LoRA SFT on attention and recurrent loop gates |
| `uv run wellbore-sft inkling-rlvr --grounded` | Run document grounded on policy RLVR on `Inkling-2-Mini` |
| `uv run wellbore-sft run-eval --mode open_book --model <id>` | Evaluate any OpenAI compatible endpoint on `eval_exploration_recent` |
| `uv run wellbore-sft score <tasks.jsonl> <preds.jsonl>` | Score any predictions file with 95% bootstrap and Wilson CIs |

## Verification

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
```
