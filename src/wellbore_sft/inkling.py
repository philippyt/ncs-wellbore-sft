import json
import math
import random
import time
from pathlib import Path
from typing import Any

import orjson

from wellbore_sft.config import PackConfig, RewardWeights
from wellbore_sft.dataset_prep import load_pages_for_docs
from wellbore_sft.grounding import (
    compute_grounded_rlvr_reward,
    compute_ungrounded_prediction_rate,
)
from wellbore_sft.page_select import render_pages_markdown, select_pages
from wellbore_sft.prompts import build_extraction_messages
from wellbore_sft.schema import WellborePrediction, parse_prediction_json
from wellbore_sft.scorer import compute_rlvr_reward, score_wellbore_prediction

INKLING_MODEL_ID = "thinkingmachines/Inkling-2-Mini"
INKLING_LORA_REGEX = (
    r".*language_model.*\."
    r"(q_proj\.linear|k_proj\.linear|v_proj\.linear|out_proj"
    r"|forget_gate_proj|message_gate_proj)$"
)


def patch_inkling_chunked_attention(chunk_size: int = 512) -> None:
    import torch
    import torch.nn.functional as F
    import transformers.models.inkling2_loop.modeling_inkling2_loop as mod

    def chunked_eager_attention_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        scaling: float,
        dropout: float = 0.0,
        position_bias: Any = None,
        sinks: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        key_states = mod.repeat_kv(key, module.num_key_value_groups)
        value_states = mod.repeat_kv(value, module.num_key_value_groups)
        q_len = query.shape[2]
        kt = key_states.transpose(2, 3)
        outs: list[torch.Tensor] = []
        for start in range(0, q_len, chunk_size):
            end = min(start + chunk_size, q_len)
            w = torch.matmul(query[:, :, start:end, :], kt) * scaling
            if position_bias is not None:
                w = w + position_bias[:, :, start:end, :]
            if attention_mask is not None:
                m_q = (
                    attention_mask[:, :, start:end, :]
                    if attention_mask.shape[2] == q_len
                    else attention_mask
                )
                w = w + m_q
            if sinks is not None:
                s_c = sinks[:, :, start:end, :] if sinks.shape[2] == q_len else sinks
                w = F.softmax(torch.cat([w, s_c], dim=-1).float(), dim=-1)[..., :-1].to(query.dtype)
            else:
                w = F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
            w = F.dropout(w, p=dropout, training=module.training)
            outs.append(torch.matmul(w, value_states))
        return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None

    mod.eager_attention_forward = chunked_eager_attention_forward


def set_loop_passes(model: Any, num_passes: int) -> None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm = base.model.language_model
    lm.config.loop_num_passes = int(num_passes)


def load_inkling_model(
    model_id: str = INKLING_MODEL_ID,
    adapter_dir: Path | None = None,
    for_training: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

    patch_inkling_chunked_attention(512)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(model.model, attr):
            getattr(model.model, attr).to("cpu")
    torch.cuda.empty_cache()

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=for_training)
    elif for_training:
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=INKLING_LORA_REGEX,
        )
        model = get_peft_model(model, peft_cfg)

    if for_training:
        model.config.use_cache = False
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base.model.language_model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.config.use_cache = True
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base.model.language_model.config.use_cache = True
        model.eval()

    return model, tokenizer


def _format_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_gen: bool,
    reasoning_effort: str = "none",
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            rendered = str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_gen,
                    reasoning_effort=reasoning_effort,
                )
            )
            if add_gen and "<|message_model|>" not in rendered[-40:]:
                rendered += "<|message_model|><|content_text|>"
            return rendered
        except Exception:
            pass
    parts = [f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages]
    if add_gen:
        parts.append("<assistant>\n")
    return "\n".join(parts)


def _encode_prompt_safely(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_prompt_tokens: int,
    reasoning_effort: str = "none",
) -> list[int]:
    prompt_text = _format_chat(tokenizer, messages, add_gen=True, reasoning_effort=reasoning_effort)
    suffix = "<|end_message|><|message_model|><|content_text|>"
    if prompt_text.endswith(suffix):
        body = prompt_text[: -len(suffix)]
        body_ids = tokenizer.encode(body, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        keep = max(1, max_prompt_tokens - len(suffix_ids))
        return body_ids[:keep] + suffix_ids
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return ids[:max_prompt_tokens]


def _compute_completion_only_loss(
    model: Any,
    input_ids: Any,
    prompt_len: int,
) -> Any:
    import torch.nn.functional as F

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    outputs = base.model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    start = max(prompt_len - 1, 0)
    shift_hidden = hidden[:, start:-1, :]
    shift_labels = input_ids[:, start + 1 :]
    logits = base.lm_head(shift_hidden).float()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1),
    )


def run_inkling_sft(
    train_jsonl: Path,
    output_dir: Path,
    model_id: str = INKLING_MODEL_ID,
    loop_passes: list[int] | None = None,
    max_prompt_tokens: int = 6656,
    max_target_tokens: int = 1536,
    epochs: int = 1,
    max_steps: int = -1,
    grad_accum: int = 4,
    learning_rate: float = 1.5e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    import torch

    active_passes = loop_passes or [3]
    rows = [orjson.loads(line) for line in train_jsonl.read_bytes().splitlines() if line.strip()]
    rng = random.Random(seed)
    model, tokenizer = load_inkling_model(
        model_id=model_id,
        for_training=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    set_loop_passes(model, active_passes[0])

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)

    total_micro = len(rows) * epochs
    total_opt_steps = max(1, math.ceil(total_micro / grad_accum))
    if max_steps > 0:
        total_opt_steps = min(total_opt_steps, max_steps)

    def lr_lambda(step: int) -> float:
        warmup = max(1, int(total_opt_steps * 0.05))
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_opt_steps - warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    opt_step = 0
    micro_step = 0
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        epoch_rows = list(rows)
        rng.shuffle(epoch_rows)
        for row in epoch_rows:
            msgs = row["messages"]
            p_ids = _encode_prompt_safely(tokenizer, msgs[:2], max_prompt_tokens)
            target_text = msgs[2]["content"] + "<|end_message|><|content_model_end_sampling|>"
            t_ids = tokenizer.encode(target_text, add_special_tokens=False)
            if len(t_ids) > max_target_tokens:
                t_ids = t_ids[:max_target_tokens]
            if not t_ids:
                continue

            chosen_pass = rng.choice(active_passes)
            set_loop_passes(model, chosen_pass)
            full_ids = torch.tensor([p_ids + t_ids], dtype=torch.long, device="cuda")
            loss = _compute_completion_only_loss(model, full_ids, prompt_len=len(p_ids))
            (loss / grad_accum).backward()
            losses.append(float(loss.item()))
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 10 == 0 or opt_step == 1:
                    recent = sum(losses[-grad_accum * 5 :]) / min(len(losses), grad_accum * 5)
                    elapsed = time.time() - t_start
                    print(
                        f"[inkling_sft] epoch={epoch + 1} step={opt_step}/{total_opt_steps} "
                        f"loss={recent:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                if opt_step % 50 == 0:
                    ckpt = output_dir / f"step_{opt_step}"
                    model.save_pretrained(str(ckpt))
                if max_steps > 0 and opt_step >= max_steps:
                    break
        if max_steps > 0 and opt_step >= max_steps:
            break

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    summary = {
        "model_id": model_id,
        "loop_passes": active_passes,
        "opt_steps": opt_step,
        "lora_r": lora_r,
        "mean_loss_final20": round(sum(losses[-20:]) / max(len(losses[-20:]), 1), 4),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "adapter_dir": str(adapter_dir),
    }
    (output_dir / "train_summary.json").write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2)
    )
    return summary


def run_inkling_rlvr(
    train_jsonl: Path,
    adapter_dir: Path,
    output_dir: Path,
    model_id: str = INKLING_MODEL_ID,
    grounded: bool = True,
    loop_num_passes: int = 3,
    max_prompt_tokens: int = 6144,
    max_new_tokens: int = 1280,
    num_generations: int = 4,
    max_steps: int = 60,
    learning_rate: float = 2e-5,
    temperature: float = 0.7,
    hallucination_penalty_weight: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    import torch

    rows = [orjson.loads(line) for line in train_jsonl.read_bytes().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)

    model, tokenizer = load_inkling_model(
        model_id=model_id,
        adapter_dir=adapter_dir,
        for_training=True,
    )
    set_loop_passes(model, loop_num_passes)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    weights = RewardWeights()

    output_dir.mkdir(parents=True, exist_ok=True)
    reward_hist: list[float] = []
    halluc_hist: list[float] = []
    t_start = time.time()

    end_msg_id = tokenizer.convert_tokens_to_ids("<|end_message|>")
    eos_ids = [int(tokenizer.eos_token_id)]
    if isinstance(end_msg_id, int) and end_msg_id >= 0 and end_msg_id not in eos_ids:
        eos_ids.append(end_msg_id)

    for step in range(1, max_steps + 1):
        row = rows[(step - 1) % len(rows)]
        p_ids = _encode_prompt_safely(tokenizer, row["prompt"], max_prompt_tokens)
        prompt_tensor = torch.tensor([p_ids], dtype=torch.long, device="cuda")

        model.eval()
        model.config.use_cache = True
        base.model.language_model.config.use_cache = True
        completions_ids: list[list[int]] = []
        completions_text: list[str] = []
        with torch.no_grad():
            for _ in range(num_generations):
                gen = model.generate(
                    input_ids=prompt_tensor,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
                new_toks = gen[0, len(p_ids) :].tolist()
                completions_ids.append(new_toks)
                completions_text.append(tokenizer.decode(new_toks, skip_special_tokens=True))

        task_dict = {
            "task_id": row["task_id"],
            "is_vertical_approx": bool(row.get("is_vertical_approx", False)),
            "ground_truth": row["ground_truth"],
        }
        pages_md = str(row.get("pages_markdown") or "")
        step_rewards: list[float] = []
        step_halluc: list[float] = []
        for comp_text in completions_text:
            if grounded and pages_md:
                bd, u_rate = compute_grounded_rlvr_reward(
                    comp_text,
                    task_dict,
                    pages_md,
                    weights=weights,
                    hallucination_penalty_weight=hallucination_penalty_weight,
                )
                step_rewards.append(bd.reward)
                step_halluc.append(u_rate)
            else:
                bd = compute_rlvr_reward(comp_text, task_dict, weights=weights)
                step_rewards.append(bd.reward)
                parsed = parse_prediction_json(comp_text, str(row["task_id"]))
                if parsed.valid_schema and parsed.prediction is not None and pages_md:
                    kb_raw = (row["ground_truth"].get("header") or {}).get("kelly_bushing_m")
                    kb_m = float(kb_raw) if kb_raw is not None else None
                    u_rate = compute_ungrounded_prediction_rate(
                        parsed.prediction,
                        pages_md,
                        kb_m,
                        bool(row.get("is_vertical_approx", False)),
                    )
                    step_halluc.append(u_rate)
                else:
                    step_halluc.append(0.0)

        mean_r = sum(step_rewards) / len(step_rewards)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in step_rewards) / len(step_rewards))
        reward_hist.append(mean_r)
        halluc_hist.append(sum(step_halluc) / len(step_halluc))

        if std_r > 1e-5:
            model.train()
            model.config.use_cache = False
            base.model.language_model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            for new_toks, rew in zip(completions_ids, step_rewards, strict=True):
                if not new_toks:
                    continue
                adv = (rew - mean_r) / (std_r + 1e-6)
                if abs(adv) < 1e-4:
                    continue
                full_ids = torch.tensor([p_ids + new_toks], dtype=torch.long, device="cuda")
                nll = _compute_completion_only_loss(model, full_ids, prompt_len=len(p_ids))
                pg_loss = (adv * nll) / num_generations
                pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if step % 5 == 0 or step == 1:
            print(
                f"[inkling_rlvr grounded={grounded}] step={step}/{max_steps} "
                f"mean_r={mean_r:.4f} std_r={std_r:.4f} "
                f"ungrounded={halluc_hist[-1]:.3f} elapsed={time.time() - t_start:.0f}s",
                flush=True,
            )

    out_adapter = output_dir / "adapter"
    model.save_pretrained(str(out_adapter))
    tokenizer.save_pretrained(str(out_adapter))
    summary = {
        "model_id": model_id,
        "grounded": grounded,
        "loop_num_passes": loop_num_passes,
        "steps": max_steps,
        "mean_reward_first10": round(sum(reward_hist[:10]) / max(len(reward_hist[:10]), 1), 4),
        "mean_reward_last10": round(sum(reward_hist[-10:]) / max(len(reward_hist[-10:]), 1), 4),
        "mean_ungrounded_last10": round(sum(halluc_hist[-10:]) / max(len(halluc_hist[-10:]), 1), 4),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "adapter_dir": str(out_adapter),
    }
    (output_dir / "rlvr_summary.json").write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2)
    )
    return summary


def evaluate_inkling_passes(
    release_dir: Path,
    tasks_jsonl: Path,
    output_dir: Path,
    passes: list[int],
    model_id: str = INKLING_MODEL_ID,
    adapter_dir: Path | None = None,
    pack_cfg: PackConfig | None = None,
    max_prompt_tokens: int = 6656,
    max_new_tokens: int = 1024,
    batch_size: int = 4,
    limit: int | None = None,
    basin_map_path: Path | None = None,
) -> dict[str, Any]:
    import torch

    tasks = [orjson.loads(line) for line in tasks_jsonl.read_bytes().splitlines() if line.strip()]
    if limit is not None and limit > 0:
        tasks = tasks[:limit]

    basin_by_doc: dict[str, str] = {}
    if basin_map_path is not None and basin_map_path.exists():
        basin_by_doc = json.loads(basin_map_path.read_text(encoding="utf-8"))

    cfg = pack_cfg or PackConfig(mode="routed", max_pages=16, max_chars=32_000)
    doc_ids = {str(t["doc_id"]) for t in tasks}
    pages_by_doc = load_pages_for_docs(
        release_dir / "dataset",
        doc_ids,
        drop_ocr_loops=cfg.drop_ocr_loops,
    )
    packed_md_by_doc: dict[str, str] = {}
    for t in tasks:
        did = str(t["doc_id"])
        packed_md_by_doc[did] = render_pages_markdown(select_pages(pages_by_doc.get(did, []), cfg))

    model, tokenizer = load_inkling_model(
        model_id=model_id,
        adapter_dir=adapter_dir,
        for_training=False,
    )
    tokenizer.padding_side = "left"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_reports: dict[str, Any] = {}

    encoded_prompts: list[list[int]] = []
    for task in tasks:
        did = str(task["doc_id"])
        msgs = build_extraction_messages(
            wellbore_name=str(task["wellbore_name"]),
            doc_id=did,
            pages_markdown=packed_md_by_doc[did],
        )
        encoded_prompts.append(_encode_prompt_safely(tokenizer, msgs, max_prompt_tokens))

    pad_id = int(tokenizer.pad_token_id)
    end_msg_id = tokenizer.convert_tokens_to_ids("<|end_message|>")
    eos_ids = [int(tokenizer.eos_token_id)]
    if isinstance(end_msg_id, int) and end_msg_id >= 0 and end_msg_id not in eos_ids:
        eos_ids.append(end_msg_id)

    for npass in passes:
        set_loop_passes(model, npass)
        per_task_rows: list[dict[str, Any]] = []
        t0 = time.time()
        for start in range(0, len(tasks), batch_size):
            end = min(start + batch_size, len(tasks))
            batch_tasks = tasks[start:end]
            batch_ids = encoded_prompts[start:end]
            max_len = max(len(x) for x in batch_ids)
            padded = [[pad_id] * (max_len - len(x)) + x for x in batch_ids]
            attn_mask = [[0] * (max_len - len(x)) + [1] * len(x) for x in batch_ids]
            inp = torch.tensor(padded, dtype=torch.long, device="cuda")
            mask = torch.tensor(attn_mask, dtype=torch.long, device="cuda")

            t_gen = time.time()
            with torch.no_grad():
                gen = model.generate(
                    input_ids=inp,
                    attention_mask=mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_ids,
                )
            gen_sec = (time.time() - t_gen) / len(batch_tasks)

            for bi, task in enumerate(batch_tasks):
                tid = str(task["task_id"])
                did = str(task["doc_id"])
                pages_md = packed_md_by_doc[did]
                new_toks = gen[bi, max_len:].tolist()
                raw_text = tokenizer.decode(new_toks, skip_special_tokens=True)
                parsed = parse_prediction_json(raw_text, tid)
                pred = parsed.prediction or WellborePrediction(task_id=tid)
                scored = score_wellbore_prediction(pred, task)
                raw_reward = compute_rlvr_reward(raw_text, task).reward
                grounded_bd, ungrounded_rate = compute_grounded_rlvr_reward(
                    raw_text, task, pages_md
                )
                basin = basin_by_doc.get(did, "UNKNOWN")
                per_task_rows.append(
                    {
                        "task_id": tid,
                        "doc_id": did,
                        "basin": basin,
                        "is_ood_basin": basin != "NORTH SEA",
                        "valid_json": parsed.valid_json,
                        "valid_schema": parsed.valid_schema,
                        "gen_tokens": len(new_toks),
                        "gen_seconds": round(gen_sec, 3),
                        "header_accuracy": scored["header_accuracy"],
                        "casing_f1": scored["casing_and_lot"]["casing"]["f1"],
                        "lot_recall": scored["casing_and_lot"]["lot_recall"],
                        "formation_tops_f1": scored["formation_tops"]["f1"],
                        "dst_expected": scored["dst"]["expected"],
                        "dst_f1": scored["dst"]["f1"],
                        "rlvr_reward": raw_reward,
                        "grounded_reward": grounded_bd.reward,
                        "ungrounded_rate": ungrounded_rate,
                    }
                )
            print(
                f"[eval pass={npass}] {end}/{len(tasks)} "
                f"schema_ok={sum(1 for r in per_task_rows if r['valid_schema'])}/{end} "
                f"mean_r={sum(r['rlvr_reward'] for r in per_task_rows) / end:.3f} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

        summary = {
            "loop_num_passes": npass,
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
            "all": _aggregate_subset(per_task_rows),
            "in_basin_north_sea": _aggregate_subset(
                [r for r in per_task_rows if not r["is_ood_basin"]]
            ),
            "ood_norwegian_barents": _aggregate_subset(
                [r for r in per_task_rows if r["is_ood_basin"]]
            ),
            "per_task": per_task_rows,
        }
        (output_dir / f"eval_pass_{npass}.json").write_bytes(
            orjson.dumps(summary, option=orjson.OPT_INDENT_2)
        )
        all_reports[f"pass_{npass}"] = {k: v for k, v in summary.items() if k != "per_task"}

    (output_dir / "sweep_summary.json").write_bytes(
        orjson.dumps(all_reports, option=orjson.OPT_INDENT_2)
    )
    return all_reports


def _aggregate_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    lot_vals = [float(r["lot_recall"]) for r in rows if r["lot_recall"] is not None]
    dst_vals = [float(r["dst_f1"]) for r in rows if r["dst_expected"] > 0]
    return {
        "n": n,
        "valid_schema_rate": round(sum(1 for r in rows if r["valid_schema"]) / n, 4),
        "header_accuracy": round(sum(float(r["header_accuracy"]) for r in rows) / n, 4),
        "casing_f1": round(sum(float(r["casing_f1"]) for r in rows) / n, 4),
        "lot_recall": round(sum(lot_vals) / len(lot_vals), 4) if lot_vals else 0.0,
        "formation_tops_f1": round(sum(float(r["formation_tops_f1"]) for r in rows) / n, 4),
        "dst_f1_non_empty": round(sum(dst_vals) / len(dst_vals), 4) if dst_vals else 0.0,
        "rlvr_reward": round(sum(float(r["rlvr_reward"]) for r in rows) / n, 4),
        "grounded_reward": round(sum(float(r["grounded_reward"]) for r in rows) / n, 4),
        "ungrounded_rate": round(sum(float(r["ungrounded_rate"]) for r in rows) / n, 4),
        "mean_gen_seconds": round(sum(float(r["gen_seconds"]) for r in rows) / n, 2),
    }
