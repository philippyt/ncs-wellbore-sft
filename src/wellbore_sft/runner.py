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
from wellbore_sft.scorer import score_eval_run


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
) -> str:
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
            return str(body["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < cfg.max_retries:
                await asyncio.sleep(2.0**attempt)
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

            async with sem:
                raw_text = await _call_chat_completion(client, inference_cfg, messages)

            parsed = parse_prediction_json(raw_text, task_id)
            pred = parsed.prediction or WellborePrediction(task_id=task_id)

            async with lock:
                with predictions_jsonl.open("ab") as handle:
                    handle.write(orjson.dumps(pred.model_dump(mode="json")))
                    handle.write(b"\n")
                if raw_log_jsonl is not None:
                    log_row = {
                        "task_id": task_id,
                        "valid_json": parsed.valid_json,
                        "valid_schema": parsed.valid_schema,
                        "error": parsed.error,
                        "raw_text": raw_text,
                    }
                    with raw_log_jsonl.open("ab") as log_handle:
                        log_handle.write(orjson.dumps(log_row))
                        log_handle.write(b"\n")

        await asyncio.gather(*(process_task(task) for task in pending))

    report = score_eval_run(tasks_jsonl, predictions_jsonl)
    return report
