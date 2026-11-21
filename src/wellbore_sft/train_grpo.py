from pathlib import Path
from typing import Any

from wellbore_sft.config import RewardWeights
from wellbore_sft.scorer import compute_rlvr_reward


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    return str(completion or "")


def make_grpo_reward_fn(weights: RewardWeights | None = None):
    reward_weights = weights or RewardWeights()

    def wellbore_reward_func(
        completions: list[Any],
        task_id: list[str],
        ground_truth: list[dict[str, Any]],
        is_vertical_approx: list[bool],
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for comp, tid, gt, is_vert in zip(
            completions,
            task_id,
            ground_truth,
            is_vertical_approx,
            strict=True,
        ):
            raw_text = _extract_completion_text(comp)
            task = {
                "task_id": tid,
                "ground_truth": gt,
                "is_vertical_approx": bool(is_vert),
            }
            breakdown = compute_rlvr_reward(raw_text, task, weights=reward_weights)
            rewards.append(breakdown.reward)
        return rewards

    return wellbore_reward_func


def run_grpo_training(
    train_jsonl: Path,
    model_name: str,
    output_dir: Path,
    max_prompt_length: int = 24576,
    max_completion_length: int = 3072,
    num_generations: int = 4,
    epochs: float = 1.0,
    learning_rate: float = 5e-6,
    batch_size: int = 1,
    grad_accum: int = 8,
    lora_r: int = 32,
    lora_alpha: int = 64,
    beta: float = 0.04,
) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: uv sync --extra train"
        ) from exc

    dataset = load_dataset("json", data_files={"train": str(train_jsonl)})["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        beta=beta,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_grpo_reward_fn()],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    return {"output_dir": str(output_dir / "final"), "metrics": dict(train_result.metrics)}
