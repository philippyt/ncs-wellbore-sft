import asyncio
from pathlib import Path
from typing import Any

import httpx
import orjson

from wellbore_sft.config import InferenceConfig, PackConfig
from wellbore_sft.dataset_prep import load_pages_for_docs
from wellbore_sft.page_select import render_pages_markdown, select_pages
from wellbore_sft.prompts import build_closed_book_messages, build_extraction_messages
from wellbore_sft.schema import WellborePrediction, parse_prediction_json
from wellbore_sft.scorer import compute_rlvr_reward, score_eval_run


def load_tasks(tasks_jsonl: Path, limit: int | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with tasks_jsonl.open("rb") as handle:
        for line in handle:
            if line.strip():
                tasks.append(orjson.loads(line))
                if limit is not None and len(tasks) >= limit:
                    break
    return tasks


def load_existing_predictions(predictions_jsonl: Path) -> set[str]:
    if not predictions_jsonl.exists():
        return set()
    done: set[str] = set()
    with predictions_jsonl.open("rb") as handle:
        for line in handle:
            if line.strip():
                row = orjson.loads(line)
                if "task_id" in row:
                    done.add(str(row["task_id"]))
    return done


async def _call_chat_completion(
    client: httpx.AsyncClient,
    cfg: InferenceConfig,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    use_json = cfg.use_json_mode
    last_error: Exception | None = None

    for attempt in range(cfg.max_retries):
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 400 and use_json:
                use_json = False
                continue
            response.raise_for_status()
            body = response.json()
            text = str(body["choices"][0]["message"].get("content") or "")
            usage = body.get("usage") or {}
            return text, usage
        except Exception as exc:
            last_error = exc
            if attempt + 1 < cfg.max_retries:
                await asyncio.sleep(min(8.0, 2.0**attempt))
    raise RuntimeError(f"Inference failed after {cfg.max_retries} attempts: {last_error}")


async def run_eval_async(
    release_dir: Path,
    tasks_jsonl: Path,
    predictions_jsonl: Path,
    inference_cfg: InferenceConfig,
    pack_cfg: PackConfig | None = None,
    limit: int | None = None,
    raw_log_jsonl: Path | None = None,
) -> dict[str, Any]:
    tasks = load_tasks(tasks_jsonl, limit=limit)
    done_ids = load_existing_predictions(predictions_jsonl)
    pending = [t for t in tasks if str(t["task_id"]) not in done_ids]

    pages_by_doc: dict[str, list[dict[str, Any]]] = {}
    if inference_cfg.mode == "open_book" and pending:
        cfg = pack_cfg or PackConfig()
        doc_ids = {str(t["doc_id"]) for t in pending}
        pages_by_doc = load_pages_for_docs(
            release_dir / "dataset",
            doc_ids,
            drop_ocr_loops=cfg.drop_ocr_loops,
        )

    predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if raw_log_jsonl is not None:
        raw_log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(inference_cfg.concurrency)
    lock = asyncio.Lock()
    pack = pack_cfg or PackConfig()
    stats = {
        "completed": 0,
        "valid_json": 0,
        "valid_schema": 0,
        "errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }
    total_pending = len(pending)

    async with httpx.AsyncClient(timeout=inference_cfg.timeout_seconds) as client:

        async def process_task(task: dict[str, Any]) -> None:
            task_id = str(task["task_id"])
            if inference_cfg.mode == "closed_book":
                messages = build_closed_book_messages(
                    wellbore_name=str(task["wellbore_name"]),
                    production_licence=task.get("production_licence"),
                    completion_year=task.get("completion_year"),
                )
            else:
                packed_pages = select_pages(pages_by_doc.get(str(task["doc_id"]), []), pack)
                pages_md = render_pages_markdown(packed_pages)
                messages = build_extraction_messages(
                    wellbore_name=str(task["wellbore_name"]),
                    doc_id=str(task["doc_id"]),
                    pages_markdown=pages_md,
                )

            raw_text = ""
            usage: dict[str, Any] = {}
            call_error: str | None = None
            async with sem:
                try:
                    raw_text, usage = await _call_chat_completion(client, inference_cfg, messages)
                except Exception as exc:
                    call_error = str(exc)

            parsed = parse_prediction_json(raw_text, task_id)
            pred = parsed.prediction or WellborePrediction(task_id=task_id)
            reward_val = compute_rlvr_reward(raw_text, task).reward if raw_text else 0.0

            async with lock:
                with predictions_jsonl.open("ab") as handle:
                    handle.write(orjson.dumps(pred.model_dump(mode="json")))
                    handle.write(b"\n")
                if raw_log_jsonl is not None:
                    log_row = {
                        "task_id": task_id,
                        "valid_json": parsed.valid_json,
                        "valid_schema": parsed.valid_schema,
                        "reward": reward_val,
                        "error": call_error or parsed.error,
                        "usage": usage,
                        "raw_text": raw_text,
                    }
                    with raw_log_jsonl.open("ab") as log_handle:
                        log_handle.write(orjson.dumps(log_row))
                        log_handle.write(b"\n")

                stats["completed"] += 1
                if parsed.valid_json:
                    stats["valid_json"] += 1
                if parsed.valid_schema:
                    stats["valid_schema"] += 1
                if call_error is not None:
                    stats["errors"] += 1
                stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                stats["cost_usd"] += float(usage.get("cost") or 0.0)
                if stats["completed"] % 20 == 0 or stats["completed"] == total_pending:
                    print(
                        f"[{inference_cfg.model} {inference_cfg.mode}] "
                        f"{stats['completed']}/{total_pending} "
                        f"schema_ok={stats['valid_schema']} "
                        f"errors={stats['errors']} "
                        f"cost_usd={stats['cost_usd']:.4f}",
                        flush=True,
                    )

        await asyncio.gather(*(process_task(task) for task in pending))

    report = score_eval_run(tasks_jsonl, predictions_jsonl)
    rewards: list[float] = []
    if predictions_jsonl.exists():
        tasks_by_id = {str(t["task_id"]): t for t in tasks}
        with predictions_jsonl.open("rb") as handle:
            for line in handle:
                if line.strip():
                    row = orjson.loads(line)
                    tid = str(row["task_id"])
                    if tid in tasks_by_id:
                        rewards.append(
                            compute_rlvr_reward(orjson.dumps(row).decode(), tasks_by_id[tid]).reward
                        )
    report["mean_rlvr_reward"] = round(sum(rewards) / max(len(rewards), 1), 6)
    report["run_metadata"] = {
        "model": inference_cfg.model,
        "mode": inference_cfg.mode,
        "pack_mode": pack.mode if inference_cfg.mode == "open_book" else "none",
        "max_pages": pack.max_pages if inference_cfg.mode == "open_book" else 0,
        "max_chars": pack.max_chars if inference_cfg.mode == "open_book" else 0,
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "cost_usd": round(stats["cost_usd"], 6),
        "valid_json_new": stats["valid_json"],
        "valid_schema_new": stats["valid_schema"],
        "errors_new": stats["errors"],
    }
    return report
