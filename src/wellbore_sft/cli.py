import asyncio
from pathlib import Path
from typing import Annotated, Literal

import orjson
import typer

from wellbore_sft.config import InferenceConfig, PackConfig, resolve_release_dir
from wellbore_sft.dataset_prep import build_training_datasets
from wellbore_sft.runner import run_eval_async
from wellbore_sft.schema import ground_truth_to_extraction
from wellbore_sft.scorer import compute_rlvr_reward, score_eval_run
from wellbore_sft.train_grpo import run_grpo_training
from wellbore_sft.train_sft import run_sft_training

app = typer.Typer(no_args_is_help=True)


@app.command("build-data")
def build_data_cmd(
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/data"),
    data_dir: Annotated[Path | None, typer.Option()] = None,
    pack_mode: Annotated[Literal["routed", "full"], typer.Option()] = "routed",
    max_pages: Annotated[int, typer.Option()] = 28,
    max_chars: Annotated[int, typer.Option()] = 120_000,
    val_fraction: Annotated[float, typer.Option()] = 0.10,
    require_casing_and_tops: Annotated[bool, typer.Option()] = True,
    max_docs: Annotated[int | None, typer.Option()] = None,
) -> None:
    release_dir = resolve_release_dir(data_dir)
    pack_cfg = PackConfig(mode=pack_mode, max_pages=max_pages, max_chars=max_chars)
    manifest = build_training_datasets(
        release_dir=release_dir,
        output_dir=output_dir,
        pack_config=pack_cfg,
        val_fraction=val_fraction,
        require_casing_and_tops=require_casing_and_tops,
        max_docs=max_docs,
    )
    typer.echo(orjson.dumps(manifest, option=orjson.OPT_INDENT_2).decode())


@app.command("run-eval")
def run_eval_cmd(
    output_predictions: Annotated[Path, typer.Option()] = Path("outputs/predictions.jsonl"),
    output_score: Annotated[Path, typer.Option()] = Path("outputs/score.json"),
    raw_log: Annotated[Path | None, typer.Option()] = Path("outputs/raw_completions.jsonl"),
    data_dir: Annotated[Path | None, typer.Option()] = None,
    tasks_jsonl: Annotated[Path | None, typer.Option()] = None,
    base_url: Annotated[str, typer.Option()] = "http://127.0.0.1:8000/v1",
    model: Annotated[str, typer.Option()] = "Qwen/Qwen2.5-14B-Instruct",
    api_key: Annotated[str, typer.Option()] = "EMPTY",
    mode: Annotated[Literal["open_book", "closed_book"], typer.Option()] = "open_book",
    pack_mode: Annotated[Literal["routed", "full"], typer.Option()] = "routed",
    max_pages: Annotated[int, typer.Option()] = 28,
    max_chars: Annotated[int, typer.Option()] = 120_000,
    concurrency: Annotated[int, typer.Option()] = 4,
    max_tokens: Annotated[int, typer.Option()] = 4096,
    limit: Annotated[int | None, typer.Option()] = None,
    json_mode: Annotated[bool, typer.Option()] = True,
) -> None:
    release_dir = resolve_release_dir(data_dir)
    tasks_path = tasks_jsonl or (release_dir / "eval" / "eval_exploration_recent.jsonl")
    inf_cfg = InferenceConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        mode=mode,
        concurrency=concurrency,
        max_tokens=max_tokens,
        use_json_mode=json_mode,
    )
    pack_cfg = PackConfig(mode=pack_mode, max_pages=max_pages, max_chars=max_chars)
    report = asyncio.run(
        run_eval_async(
            release_dir=release_dir,
            tasks_jsonl=tasks_path,
            predictions_jsonl=output_predictions,
            inference_cfg=inf_cfg,
            pack_cfg=pack_cfg,
            limit=limit,
            raw_log_jsonl=raw_log,
        )
    )
    output_score.parent.mkdir(parents=True, exist_ok=True)
    output_score.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    summary = {k: v for k, v in report.items() if k != "per_task"}
    typer.echo(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())


@app.command("score")
def score_cmd(
    tasks_jsonl: Annotated[Path, typer.Argument()],
    predictions_jsonl: Annotated[Path, typer.Argument()],
    output_json: Annotated[Path | None, typer.Option()] = None,
) -> None:
    report = score_eval_run(tasks_jsonl, predictions_jsonl)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    summary = {k: v for k, v in report.items() if k != "per_task"}
    typer.echo(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())


@app.command("reward-check")
def reward_check_cmd(
    tasks_jsonl: Annotated[Path | None, typer.Option()] = None,
    predictions_jsonl: Annotated[Path | None, typer.Option()] = None,
    data_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    release_dir = resolve_release_dir(data_dir)
    tasks_path = tasks_jsonl or (release_dir / "eval" / "eval_exploration_recent.jsonl")
    preds_path = predictions_jsonl or (release_dir / "eval" / "baseline_predictions.jsonl")

    tasks: dict[str, dict] = {}
    with tasks_path.open("rb") as handle:
        for line in handle:
            if line.strip():
                row = orjson.loads(line)
                tasks[str(row["task_id"])] = row

    oracle_rewards: list[float] = []
    for task in tasks.values():
        oracle_json = orjson.dumps(
            ground_truth_to_extraction(task["ground_truth"]).model_dump(mode="json")
        ).decode()
        oracle_rewards.append(compute_rlvr_reward(oracle_json, task).reward)

    baseline_rewards: list[float] = []
    if preds_path.exists():
        with preds_path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    pred_row = orjson.loads(line)
                    tid = str(pred_row["task_id"])
                    if tid in tasks:
                        baseline_rewards.append(
                            compute_rlvr_reward(orjson.dumps(pred_row).decode(), tasks[tid]).reward
                        )

    summary = {
        "tasks_checked": len(tasks),
        "oracle_mean_reward": round(sum(oracle_rewards) / max(len(oracle_rewards), 1), 6),
        "oracle_min_reward": round(min(oracle_rewards), 6) if oracle_rewards else 0.0,
        "baseline_predictions_checked": len(baseline_rewards),
        "baseline_mean_reward": round(sum(baseline_rewards) / max(len(baseline_rewards), 1), 6)
        if baseline_rewards
        else 0.0,
    }
    typer.echo(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())


@app.command("train-sft")
def train_sft_cmd(
    train_jsonl: Annotated[Path, typer.Option()] = Path("artifacts/data/sft_train.jsonl"),
    val_jsonl: Annotated[Path, typer.Option()] = Path("artifacts/data/sft_val.jsonl"),
    model: Annotated[str, typer.Option()] = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: Annotated[Path, typer.Option()] = Path("checkpoints/sft"),
    max_seq_length: Annotated[int, typer.Option()] = 32768,
    epochs: Annotated[float, typer.Option()] = 2.0,
    learning_rate: Annotated[float, typer.Option()] = 2e-4,
    batch_size: Annotated[int, typer.Option()] = 1,
    grad_accum: Annotated[int, typer.Option()] = 8,
    lora_r: Annotated[int, typer.Option()] = 32,
    lora_alpha: Annotated[int, typer.Option()] = 64,
    load_in_4bit: Annotated[bool, typer.Option()] = False,
) -> None:
    result = run_sft_training(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        model_name=model,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        load_in_4bit=load_in_4bit,
    )
    typer.echo(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())


@app.command("train-grpo")
def train_grpo_cmd(
    train_jsonl: Annotated[Path, typer.Option()] = Path("artifacts/data/rlvr_train.jsonl"),
    model: Annotated[str, typer.Option()] = "checkpoints/sft/final",
    output_dir: Annotated[Path, typer.Option()] = Path("checkpoints/grpo"),
    max_prompt_length: Annotated[int, typer.Option()] = 24576,
    max_completion_length: Annotated[int, typer.Option()] = 3072,
    num_generations: Annotated[int, typer.Option()] = 4,
    epochs: Annotated[float, typer.Option()] = 1.0,
    learning_rate: Annotated[float, typer.Option()] = 5e-6,
    batch_size: Annotated[int, typer.Option()] = 1,
    grad_accum: Annotated[int, typer.Option()] = 8,
    lora_r: Annotated[int, typer.Option()] = 32,
    lora_alpha: Annotated[int, typer.Option()] = 64,
    beta: Annotated[float, typer.Option()] = 0.04,
) -> None:
    result = run_grpo_training(
        train_jsonl=train_jsonl,
        model_name=model,
        output_dir=output_dir,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        beta=beta,
    )
    typer.echo(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
