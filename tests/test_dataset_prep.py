from pathlib import Path

import orjson
import polars as pl

from wellbore_sft.config import PackConfig
from wellbore_sft.dataset_prep import build_training_datasets


def test_build_training_datasets_no_leakage(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "release" / "dataset"
    pages_dir = dataset_dir / "pages"
    pages_dir.mkdir(parents=True)

    docs = pl.DataFrame(
        [
            {
                "doc_id": "1/1-1::1",
                "wellbore_name": "1/1-1",
                "wlbNpdidWellbore": 101,
                "wlbProductionLicence": "PL001",
                "wlbDrillingOperator": "OpA",
                "wlbPurpose": "WILDCAT",
                "wlbStatus": "P&A",
                "wlbContent": "OIL",
                "wlbCompletionYear": 2010,
                "wlbKellyBushElevation": 25.0,
                "wlbWaterDepth": 100.0,
                "wlbTotalDepth": 3000.0,
                "wlbFinalVerticalDepth": 2999.0,
                "is_vertical_approx": True,
                "total_pages": 10,
                "split_eval": "train",
                "is_canonical": True,
                "factpages_join_type": "exact",
            },
            {
                "doc_id": "2/2-2::2",
                "wellbore_name": "2/2-2",
                "wlbNpdidWellbore": 202,
                "wlbProductionLicence": "PL002",
                "wlbDrillingOperator": "OpB",
                "wlbPurpose": "WILDCAT",
                "wlbStatus": "P&A",
                "wlbContent": "DRY",
                "wlbCompletionYear": 2015,
                "wlbKellyBushElevation": 30.0,
                "wlbWaterDepth": 150.0,
                "wlbTotalDepth": 3500.0,
                "wlbFinalVerticalDepth": 3500.0,
                "is_vertical_approx": True,
                "total_pages": 12,
                "split_eval": "test",
                "is_canonical": True,
                "factpages_join_type": "exact",
            },
        ]
    )
    docs.write_parquet(dataset_dir / "documents.parquet")

    pl.DataFrame(
        [
            {
                "wlbNpdidWellbore": 101,
                "wlbCasingDiameter": "13 3/8",
                "wlbCasingDepth": 1500.0,
                "wlbLotMudDencity": 1.65,
            },
            {
                "wlbNpdidWellbore": 202,
                "wlbCasingDiameter": "13 3/8",
                "wlbCasingDepth": 1600.0,
                "wlbLotMudDencity": 1.70,
            },
        ]
    ).write_parquet(dataset_dir / "casing_and_lot.parquet")

    pl.DataFrame(
        [
            {"wlbNpdidWellbore": 101, "lsuName": "NORDLAND GP", "lsuTopDepth": 125.0},
            {"wlbNpdidWellbore": 202, "lsuName": "NORDLAND GP", "lsuTopDepth": 180.0},
        ]
    ).write_parquet(dataset_dir / "formation_tops.parquet")

    pl.DataFrame(
        schema={
            "wlbNpdidWellbore": pl.Int64,
            "wlbDstFromDepth": pl.Float64,
            "wlbDstToDepth": pl.Float64,
        }
    ).write_parquet(dataset_dir / "dst.parquet")

    pl.DataFrame(
        [
            {
                "doc_id": "1/1-1::1",
                "page_no": 0,
                "ocr_markdown": "Well Data Summary. Kelly Bushing 25 m. Water Depth 100 m.",
                "ai_tags": ["well_data"],
                "has_tables": True,
                "page_quality": "Reliable",
            },
            {
                "doc_id": "2/2-2::2",
                "page_no": 0,
                "ocr_markdown": "Well Data Summary. Kelly Bushing 30 m.",
                "ai_tags": ["well_data"],
                "has_tables": True,
                "page_quality": "Reliable",
            },
        ]
    ).write_parquet(pages_dir / "part-0000.parquet")

    pl.DataFrame(
        [
            {"doc_id": "1/1-1::1", "page_no": 0, "is_ocr_loop": False},
            {"doc_id": "2/2-2::2", "page_no": 0, "is_ocr_loop": False},
        ]
    ).write_parquet(dataset_dir / "page_flags.parquet")

    out_dir = tmp_path / "artifacts"
    manifest = build_training_datasets(
        release_dir=tmp_path / "release",
        output_dir=out_dir,
        pack_config=PackConfig(mode="routed", max_pages=10, max_chars=20_000),
        val_fraction=0.0,
    )
    assert manifest["sft_train_count"] == 1
    assert manifest["rlvr_train_count"] == 1
    assert manifest["licence_overlap_count"] == 0

    row = orjson.loads((out_dir / "sft_train.jsonl").read_bytes().splitlines()[0])
    assert row["doc_id"] == "1/1-1::1"
    assert len(row["messages"]) == 3
