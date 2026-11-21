from pathlib import Path
from typing import Any


def merge_lora_checkpoint(
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: uv sync --extra train"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return {"merged_dir": str(output_dir)}


def run_sft_training(
    train_jsonl: Path,
    val_jsonl: Path,
    model_name: str,
    output_dir: Path,
    max_seq_length: int = 24576,
    epochs: float = 2.0,
    max_steps: int = -1,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
    lora_r: int = 32,
    lora_alpha: int = 64,
    load_in_4bit: bool = False,
    attn_implementation: str = "sdpa",
    merge_and_save: bool = True,
) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: uv sync --extra train"
        ) from exc

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_jsonl), "validation": str(val_jsonl)},
    )

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        attn_implementation=attn_implementation,
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

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        bf16=True,
        gradient_checkpointing=True,
        max_length=max_seq_length,
        logging_steps=5,
        eval_strategy="epoch" if max_steps <= 0 else "no",
        save_strategy="epoch" if max_steps <= 0 else "steps",
        save_steps=50,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_dir: Path | None = None
    if merge_and_save and not load_in_4bit:
        merged_dir = output_dir / "merged"
        merged_model = trainer.model.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    return {
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir) if merged_dir is not None else None,
        "metrics": dict(train_result.metrics),
    }
