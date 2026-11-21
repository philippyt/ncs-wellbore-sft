import hashlib
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from wellbore_sft.config import PackConfig
from wellbore_sft.page_select import render_pages_markdown, select_pages
from wellbore_sft.prompts import build_extraction_messages
from wellbore_sft.schema import ground_truth_to_extraction


def _group_rows(frame: pl.DataFrame, key_col: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in frame.iter_rows(named=True):
        key = row.get(key_col)
        if key is None:
            continue
        grouped.setdefault(int(key), []).append(row)
    return grouped


def _assign_licence_split(licence: str | None, val_fraction: float) -> str:
    key = (licence or "UNKNOWN").strip().upper()
    digest = int(hashlib.blake2b(key.encode("utf-8"), digest_size=4).hexdigest(), 16)
    bucket = (digest % 10_000) / 10_000.0
    return "val" if bucket < val_fraction else "train"


def load_pages_for_docs(
    dataset_dir: Path,
    doc_ids: set[str],
    drop_ocr_loops: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    pages_glob = str(dataset_dir / "pages" / "*.parquet")
    lazy = (
        pl.scan_parquet(pages_glob)
        .filter(pl.col("doc_id").is_in(list(doc_ids)))
        .select(["doc_id", "page_no", "ocr_markdown", "ai_tags", "has_tables", "page_quality"])
    )
    flags_path = dataset_dir / "page_flags.parquet"
    if drop_ocr_loops and flags_path.exists():
        flags = pl.scan_parquet(flags_path).select(["doc_id", "page_no", "is_ocr_loop"])
        lazy = lazy.join(flags, on=["doc_id", "page_no"], how="left").with_columns(
            pl.col("is_ocr_loop").fill_null(False)
        )
    else:
        lazy = lazy.with_columns(pl.lit(False).alias("is_ocr_loop"))

    frame = lazy.collect()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in frame.iter_rows(named=True):
        grouped.setdefault(str(row["doc_id"]), []).append(row)
    return grouped


def build_training_datasets(
    release_dir: Path,
    output_dir: Path,
    pack_config: PackConfig | None = None,
    val_fraction: float = 0.10,
    require_casing_and_tops: bool = True,
    max_docs: int | None = None,
) -> dict[str, Any]:
    cfg = pack_config or PackConfig()
    dataset_dir = release_dir / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = pl.read_parquet(dataset_dir / "documents.parquet")
    casing = pl.read_parquet(dataset_dir / "casing_and_lot.parquet")
    tops = pl.read_parquet(dataset_dir / "formation_tops.parquet")
    dst = pl.read_parquet(dataset_dir / "dst.parquet")

    casing_ids = set(casing["wlbNpdidWellbore"].drop_nulls().to_list())
    tops_ids = set(tops["wlbNpdidWellbore"].drop_nulls().to_list())

    candidates = docs.filter(
        (pl.col("split_eval") == "train")
        & pl.col("is_canonical")
        & (pl.col("factpages_join_type") == "exact")
        & pl.col("wlbNpdidWellbore").is_not_null()
        & (pl.col("total_pages") >= 5)
    )
    if require_casing_and_tops:
        eligible_ids = list(casing_ids & tops_ids)
        candidates = candidates.filter(pl.col("wlbNpdidWellbore").is_in(eligible_ids))

    selected = (
        candidates.sort(["wlbCompletionYear", "total_pages", "doc_id"], descending=True)
        .unique(subset=["wlbNpdidWellbore"], keep="first", maintain_order=True)
    )
    if max_docs is not None and max_docs > 0:
        selected = selected.head(max_docs)

    doc_ids = set(selected["doc_id"].to_list())
    pages_by_doc = load_pages_for_docs(dataset_dir, doc_ids, drop_ocr_loops=cfg.drop_ocr_loops)

    casing_by_id = _group_rows(casing, "wlbNpdidWellbore")
    tops_by_id = _group_rows(tops, "wlbNpdidWellbore")
    dst_by_id = _group_rows(dst, "wlbNpdidWellbore")

    sft_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    rlvr_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    char_lengths: list[int] = []
    page_counts: list[int] = []
    licences_by_split: dict[str, set[str]] = {"train": set(), "val": set()}

    for row in selected.iter_rows(named=True):
        doc_id = str(row["doc_id"])
        raw_pages = pages_by_doc.get(doc_id, [])
        packed_pages = select_pages(raw_pages, cfg)
        if not packed_pages:
            continue
        pages_md = render_pages_markdown(packed_pages)
        npdid = int(row["wlbNpdidWellbore"])
        task_id = f"ncs_train_{npdid}"
        ground_truth = {
            "header": {
                "wellbore_name": row["wellbore_name"],
                "production_licence": row["wlbProductionLicence"],
                "drilling_operator": row["wlbDrillingOperator"],
                "purpose": row["wlbPurpose"],
                "status": row["wlbStatus"],
                "content": row["wlbContent"],
                "kelly_bushing_m": row["wlbKellyBushElevation"],
                "water_depth_m": row["wlbWaterDepth"],
                "total_depth_md_rkb_m": row["wlbTotalDepth"],
                "final_vertical_depth_tvd_rkb_m": row["wlbFinalVerticalDepth"],
            },
            "casing_and_lot": casing_by_id.get(npdid, []),
            "formation_tops": tops_by_id.get(npdid, []),
            "dst": dst_by_id.get(npdid, []),
        }
        target_extraction = ground_truth_to_extraction(ground_truth)
        sft_messages = build_extraction_messages(
            wellbore_name=str(row["wellbore_name"]),
            doc_id=doc_id,
            pages_markdown=pages_md,
            target=target_extraction,
        )
        rlvr_prompt = sft_messages[:2]

        licence = str(row.get("wlbProductionLicence") or "UNKNOWN")
        split = _assign_licence_split(licence, val_fraction)
        licences_by_split[split].add(licence)
        char_lengths.append(len(pages_md))
        page_counts.append(len(packed_pages))

        sft_rows[split].append(
            {
                "task_id": task_id,
                "doc_id": doc_id,
                "wellbore_name": str(row["wellbore_name"]),
                "production_licence": licence,
                "selected_pages": [int(p["page_no"]) for p in packed_pages],
                "prompt_chars": len(pages_md),
                "messages": sft_messages,
            }
        )
        rlvr_rows[split].append(
            {
                "task_id": task_id,
                "doc_id": doc_id,
                "wellbore_name": str(row["wellbore_name"]),
                "production_licence": licence,
                "is_vertical_approx": bool(row.get("is_vertical_approx", False)),
                "selected_pages": [int(p["page_no"]) for p in packed_pages],
                "prompt_chars": len(pages_md),
                "prompt": rlvr_prompt,
                "ground_truth": ground_truth,
            }
        )

    for split in ("train", "val"):
        _write_jsonl(output_dir / f"sft_{split}.jsonl", sft_rows[split])
        _write_jsonl(output_dir / f"rlvr_{split}.jsonl", rlvr_rows[split])

    sorted_chars = sorted(char_lengths)
    sorted_pages = sorted(page_counts)
    overlap = licences_by_split["train"] & licences_by_split["val"]
    manifest = {
        "release_dir": str(release_dir),
        "pack_config": cfg.model_dump(),
        "sft_train_count": len(sft_rows["train"]),
        "sft_val_count": len(sft_rows["val"]),
        "rlvr_train_count": len(rlvr_rows["train"]),
        "rlvr_val_count": len(rlvr_rows["val"]),
        "licence_overlap_count": len(overlap),
        "selected_pages_p50": _percentile(sorted_pages, 0.50),
        "selected_pages_p90": _percentile(sorted_pages, 0.90),
        "prompt_chars_p50": _percentile(sorted_chars, 0.50),
        "prompt_chars_p90": _percentile(sorted_chars, 0.90),
        "est_tokens_p50": int(_percentile(sorted_chars, 0.50) / 4),
        "est_tokens_p90": int(_percentile(sorted_chars, 0.90) / 4),
    }
    (output_dir / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * q))))
    return sorted_vals[idx]
