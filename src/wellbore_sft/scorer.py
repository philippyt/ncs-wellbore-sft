import math
import random
import re
from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel, Field

from wellbore_sft.config import RewardWeights
from wellbore_sft.schema import (
    CasingPrediction,
    DepthValue,
    DstPrediction,
    FormationTopPrediction,
    HeaderPrediction,
    WellborePrediction,
    parse_prediction_json,
)

FEET_TO_METERS = 0.3048
PPG_TO_GCM3 = 0.119826427
FRACTION_INCH_RE = re.compile(
    r"^\s*(\d+)\s*[- ]\s*(\d+)\s*/\s*(\d+)\s*(?:\"{1,2}|''|”|in(?:ch(?:es)?)?\.?)?\s*$",
    re.IGNORECASE,
)
DECIMAL_INCH_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:\"{1,2}|''|”|in(?:ch(?:es)?)?\.?)?\s*$",
    re.IGNORECASE,
)
UNICODE_FRACTIONS: dict[str, str] = {
    "¼": " 1/4",
    "½": " 1/2",
    "¾": " 3/4",
    "⅛": " 1/8",
    "⅜": " 3/8",
    "⅝": " 5/8",
    "⅞": " 7/8",
}
LITHO_SUFFIX_RE = re.compile(r"\b(GROUP|GP|FORMATION|FM|MEMBER|MB|SUBGROUP|SG)\b")
NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


class ScoreTolerances(BaseModel):
    depth_m: float = Field(default=2.0, ge=0.0)
    kb_m: float = Field(default=0.5, ge=0.0)
    mud_weight_gcm3: float = Field(default=0.05, ge=0.0)
    casing_in: float = Field(default=0.125, ge=0.0)


class TableMetrics(BaseModel):
    matched: int
    predicted: int
    expected: int
    precision: float
    recall: float
    f1: float


class RewardBreakdown(BaseModel):
    reward: float
    valid_json: bool
    valid_schema: bool
    header_accuracy: float
    casing_f1: float
    lot_score: float
    formation_tops_f1: float
    dst_f1: float
    bloat_penalty: float


def compute_prf(matched: int, predicted: int, expected: int) -> TableMetrics:
    if expected == 0 and predicted == 0:
        return TableMetrics(
            matched=0,
            predicted=0,
            expected=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
        )
    precision = matched / predicted if predicted > 0 else 0.0
    recall = matched / expected if expected > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return TableMetrics(
        matched=matched,
        predicted=predicted,
        expected=expected,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def normalize_formation_name(name: str) -> str:
    upper = (
        name.upper()
        .replace("Æ", "AE")
        .replace("Ø", "O")
        .replace("Å", "A")
        .replace("Ä", "A")
        .replace("Ö", "O")
    )
    cleaned = NON_ALNUM_RE.sub(" ", upper).strip()
    stripped = LITHO_SUFFIX_RE.sub("", cleaned)
    return " ".join(stripped.split())


def parse_casing_inches(raw: str | float | None) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = raw.strip().replace("\u2044", "/")
    for sym, repl in UNICODE_FRACTIONS.items():
        if sym in text:
            text = text.replace(sym, repl)
    if not text:
        return None
    frac = FRACTION_INCH_RE.match(text)
    if frac:
        whole, num, den = int(frac.group(1)), int(frac.group(2)), int(frac.group(3))
        return whole + (num / den)
    dec = DECIMAL_INCH_RE.match(text)
    if dec:
        return float(dec.group(1))
    return None


def to_meters(value: float, unit: Literal["m", "ft"]) -> float:
    return value * FEET_TO_METERS if unit == "ft" else value


def to_gcm3(value: float, unit: Literal["g/cm3", "sg", "ppg"]) -> float:
    return value * PPG_TO_GCM3 if unit == "ppg" else value


def normalize_depth_to_md_rkb(
    depth: DepthValue,
    kb_m: float | None,
    is_vertical: bool,
) -> float | None:
    meters = to_meters(depth.value, depth.unit)
    if depth.datum == "md_rkb":
        return meters
    if not is_vertical:
        return None
    if depth.datum == "tvd_rkb":
        return meters
    if depth.datum == "tvd_msl" and kb_m is not None:
        return meters + kb_m
    return None


def normalize_depth_to_tvd_rkb(
    depth: DepthValue,
    kb_m: float | None,
) -> float | None:
    meters = to_meters(depth.value, depth.unit)
    if depth.datum == "tvd_rkb":
        return meters
    if depth.datum == "tvd_msl" and kb_m is not None:
        return meters + kb_m
    return None


def within_tolerance(predicted: float | None, expected: float | None, tol: float) -> bool:
    if predicted is None or expected is None:
        return False
    return abs(predicted - expected) <= tol


def score_header(
    pred: HeaderPrediction,
    truth_header: dict[str, Any],
    is_vertical: bool,
    tolerances: ScoreTolerances | None = None,
) -> dict[str, bool]:
    tol = tolerances or ScoreTolerances()
    kb_truth = _as_float(truth_header.get("kelly_bushing_m"))
    wd_truth = _as_float(truth_header.get("water_depth_m"))
    td_truth = _as_float(truth_header.get("total_depth_md_rkb_m"))
    tvd_truth = _as_float(truth_header.get("final_vertical_depth_tvd_rkb_m"))

    td_pred_m = (
        normalize_depth_to_md_rkb(pred.total_depth, kb_truth, is_vertical)
        if pred.total_depth is not None
        else None
    )
    tvd_pred_m = (
        normalize_depth_to_tvd_rkb(pred.final_vertical_depth, kb_truth)
        if pred.final_vertical_depth is not None
        else None
    )

    return {
        "kelly_bushing_m": within_tolerance(pred.kelly_bushing_m, kb_truth, tol.kb_m),
        "water_depth_m": within_tolerance(pred.water_depth_m, wd_truth, tol.depth_m),
        "total_depth_md_rkb_m": within_tolerance(td_pred_m, td_truth, tol.depth_m),
        "final_vertical_depth_tvd_rkb_m": within_tolerance(tvd_pred_m, tvd_truth, tol.depth_m),
    }


def score_casing_table(
    preds: list[CasingPrediction],
    truth_rows: list[dict[str, Any]],
    kb_m: float | None,
    is_vertical: bool,
    tolerances: ScoreTolerances | None = None,
) -> dict[str, Any]:
    tol = tolerances or ScoreTolerances()
    truth_items: list[tuple[float, float, float | None]] = []
    for row in truth_rows:
        diam = parse_casing_inches(row.get("wlbCasingDiameter"))
        depth = _as_float(row.get("wlbCasingDepth"))
        lot = _as_float(row.get("wlbLotMudDencity"))
        if lot is not None and lot <= 0.0:
            lot = None
        if diam is not None and depth is not None:
            truth_items.append((diam, depth, lot))

    used_truth: set[int] = set()
    matched_casing = 0
    lot_expected = sum(1 for _, _, lot in truth_items if lot is not None)
    lot_matched = 0

    for pred in preds:
        pred_diam = parse_casing_inches(pred.casing_diameter_in)
        pred_depth = normalize_depth_to_md_rkb(pred.shoe_depth, kb_m, is_vertical)
        if pred_diam is None or pred_depth is None:
            continue
        for idx, (t_diam, t_depth, t_lot) in enumerate(truth_items):
            if idx in used_truth:
                continue
            if within_tolerance(pred_diam, t_diam, tol.casing_in) and within_tolerance(
                pred_depth, t_depth, tol.depth_m
            ):
                used_truth.add(idx)
                matched_casing += 1
                if t_lot is not None and pred.lot_mud_weight is not None:
                    pred_lot_gcm3 = to_gcm3(pred.lot_mud_weight, pred.lot_unit)
                    if within_tolerance(pred_lot_gcm3, t_lot, tol.mud_weight_gcm3):
                        lot_matched += 1
                break

    casing_metrics = compute_prf(matched_casing, len(preds), len(truth_items))
    lot_recall = (lot_matched / lot_expected) if lot_expected > 0 else None
    return {
        "casing": casing_metrics.model_dump(),
        "lot_matched": lot_matched,
        "lot_expected": lot_expected,
        "lot_recall": lot_recall,
    }


def score_formation_tops(
    preds: list[FormationTopPrediction],
    truth_rows: list[dict[str, Any]],
    kb_m: float | None,
    is_vertical: bool,
    tolerances: ScoreTolerances | None = None,
) -> TableMetrics:
    tol = tolerances or ScoreTolerances()
    truth_items: list[tuple[str, float]] = []
    for row in truth_rows:
        name = str(row.get("lsuName") or "").strip()
        depth = _as_float(row.get("lsuTopDepth"))
        if name and depth is not None:
            truth_items.append((normalize_formation_name(name), depth))

    used_truth: set[int] = set()
    matched = 0
    for pred in preds:
        pred_name = normalize_formation_name(pred.surface_name)
        pred_depth = normalize_depth_to_md_rkb(pred.top_depth, kb_m, is_vertical)
        if not pred_name or pred_depth is None:
            continue
        for idx, (t_name, t_depth) in enumerate(truth_items):
            if idx in used_truth:
                continue
            if pred_name == t_name and within_tolerance(pred_depth, t_depth, tol.depth_m):
                used_truth.add(idx)
                matched += 1
                break

    return compute_prf(matched, len(preds), len(truth_items))


def score_dst_table(
    preds: list[DstPrediction],
    truth_rows: list[dict[str, Any]],
    kb_m: float | None,
    is_vertical: bool,
    tolerances: ScoreTolerances | None = None,
) -> TableMetrics:
    tol = tolerances or ScoreTolerances()
    truth_items: list[tuple[float, float]] = []
    for row in truth_rows:
        top = _as_float(row.get("wlbDstFromDepth"))
        bottom = _as_float(row.get("wlbDstToDepth"))
        if top is not None and bottom is not None:
            truth_items.append((min(top, bottom), max(top, bottom)))

    used_truth: set[int] = set()
    matched = 0
    for pred in preds:
        p_from = normalize_depth_to_md_rkb(pred.from_depth, kb_m, is_vertical)
        p_to = normalize_depth_to_md_rkb(pred.to_depth, kb_m, is_vertical)
        if p_from is None or p_to is None:
            continue
        p_low, p_high = min(p_from, p_to), max(p_from, p_to)
        for idx, (t_from, t_to) in enumerate(truth_items):
            if idx in used_truth:
                continue
            if within_tolerance(p_low, t_from, tol.depth_m) and within_tolerance(
                p_high, t_to, tol.depth_m
            ):
                used_truth.add(idx)
                matched += 1
                break

    return compute_prf(matched, len(preds), len(truth_items))


def score_wellbore_prediction(
    pred: WellborePrediction,
    task: dict[str, Any],
    tolerances: ScoreTolerances | None = None,
) -> dict[str, Any]:
    truth = task["ground_truth"]
    header_truth = truth["header"]
    kb_m = _as_float(header_truth.get("kelly_bushing_m"))
    is_vertical = bool(task.get("is_vertical_approx", False))

    header_scores = score_header(pred.header, header_truth, is_vertical, tolerances)
    casing_scores = score_casing_table(
        pred.casing_and_lot,
        truth.get("casing_and_lot", []),
        kb_m,
        is_vertical,
        tolerances,
    )
    tops_scores = score_formation_tops(
        pred.formation_tops,
        truth.get("formation_tops", []),
        kb_m,
        is_vertical,
        tolerances,
    )
    dst_scores = score_dst_table(
        pred.dst,
        truth.get("dst", []),
        kb_m,
        is_vertical,
        tolerances,
    )
    return {
        "task_id": pred.task_id,
        "header": header_scores,
        "header_accuracy": sum(header_scores.values()) / len(header_scores),
        "casing_and_lot": casing_scores,
        "formation_tops": tops_scores.model_dump(),
        "dst": dst_scores.model_dump(),
    }


def compute_rlvr_reward(
    raw_completion: str,
    task: dict[str, Any],
    weights: RewardWeights | None = None,
    tolerances: ScoreTolerances | None = None,
) -> RewardBreakdown:
    w = weights or RewardWeights()
    parsed = parse_prediction_json(raw_completion, str(task["task_id"]))
    if not parsed.valid_schema or parsed.prediction is None:
        base_reward = 0.02 if parsed.valid_json else 0.0
        return RewardBreakdown(
            reward=base_reward,
            valid_json=parsed.valid_json,
            valid_schema=False,
            header_accuracy=0.0,
            casing_f1=0.0,
            lot_score=0.0,
            formation_tops_f1=0.0,
            dst_f1=0.0,
            bloat_penalty=0.0,
        )

    scored = score_wellbore_prediction(parsed.prediction, task, tolerances)
    header_acc = float(scored["header_accuracy"])
    casing_f1 = float(scored["casing_and_lot"]["casing"]["f1"])
    lot_raw = scored["casing_and_lot"]["lot_recall"]
    lot_score = float(lot_raw) if lot_raw is not None else casing_f1
    tops_f1 = float(scored["formation_tops"]["f1"])
    dst_f1 = float(scored["dst"]["f1"])

    predicted_rows = (
        len(parsed.prediction.casing_and_lot)
        + len(parsed.prediction.formation_tops)
        + len(parsed.prediction.dst)
    )
    expected_rows = (
        int(scored["casing_and_lot"]["casing"]["expected"])
        + int(scored["formation_tops"]["expected"])
        + int(scored["dst"]["expected"])
    )
    bloat_penalty = 0.0
    if predicted_rows > max(expected_rows * 2, 40):
        excess_ratio = (predicted_rows - max(expected_rows * 2, 40)) / max(expected_rows, 20)
        bloat_penalty = min(w.bloat_penalty_max, excess_ratio * 0.05)

    weight_sum = w.format_valid + w.header + w.casing + w.lot + w.formation_tops + w.dst
    raw_score = (
        w.format_valid * 1.0
        + w.header * header_acc
        + w.casing * casing_f1
        + w.lot * lot_score
        + w.formation_tops * tops_f1
        + w.dst * dst_f1
    ) / max(weight_sum, 1e-9)
    final_reward = max(0.0, min(1.0, raw_score - bloat_penalty))

    return RewardBreakdown(
        reward=round(final_reward, 6),
        valid_json=True,
        valid_schema=True,
        header_accuracy=header_acc,
        casing_f1=casing_f1,
        lot_score=lot_score,
        formation_tops_f1=tops_f1,
        dst_f1=dst_f1,
        bloat_penalty=round(bloat_penalty, 6),
    )


def _bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    seed: int = 42,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    low = means[int(0.025 * n_resamples)]
    high = means[int(0.975 * n_resamples)]
    return [round(low, 4), round(high, 4)]


def _wilson_ci(successes: int, trials: int, z: float = 1.959964) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    p = successes / trials
    denom = 1.0 + (z * z) / trials
    center = (p + (z * z) / (2.0 * trials)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * trials)) / trials)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def score_eval_run(
    tasks_jsonl: Path,
    predictions_jsonl: Path,
    tolerances: ScoreTolerances | None = None,
) -> dict[str, Any]:
    tasks_by_id: dict[str, dict[str, Any]] = {}
    with tasks_jsonl.open("rb") as handle:
        for line in handle:
            if line.strip():
                task = orjson.loads(line)
                tasks_by_id[str(task["task_id"])] = task

    preds_by_id: dict[str, WellborePrediction] = {}
    with predictions_jsonl.open("rb") as handle:
        for line in handle:
            if line.strip():
                pred = WellborePrediction.model_validate(orjson.loads(line))
                preds_by_id[pred.task_id] = pred

    results: list[dict[str, Any]] = []
    for task_id, task in tasks_by_id.items():
        pred = preds_by_id.get(task_id, WellborePrediction(task_id=task_id))
        results.append(score_wellbore_prediction(pred, task, tolerances))

    count = max(len(results), 1)
    header_vals = [float(r["header_accuracy"]) for r in results]
    casing_vals = [float(r["casing_and_lot"]["casing"]["f1"]) for r in results]
    lot_pos_vals = [
        float(r["casing_and_lot"]["lot_recall"])
        for r in results
        if r["casing_and_lot"]["lot_recall"] is not None
    ]
    tops_vals = [float(r["formation_tops"]["f1"]) for r in results]
    dst_pos_vals = [float(r["dst"]["f1"]) for r in results if r["dst"]["expected"] > 0]
    dst_pos_f1 = sum(dst_pos_vals) / len(dst_pos_vals) if dst_pos_vals else 0.0
    dst_pos_hits = sum(1 for v in dst_pos_vals if v > 0.0)
    dst_pos_ci = (
        _wilson_ci(dst_pos_hits, len(dst_pos_vals))
        if dst_pos_f1 == 0.0
        else _bootstrap_ci(dst_pos_vals)
    )
    lot_mean = sum(lot_pos_vals) / len(lot_pos_vals) if lot_pos_vals else 0.0
    return {
        "tasks_scored": len(results),
        "predictions_provided": len(preds_by_id),
        "mean_header_accuracy": sum(header_vals) / count,
        "header_accuracy_ci95": _bootstrap_ci(header_vals),
        "mean_casing_f1": sum(casing_vals) / count,
        "casing_f1_ci95": _bootstrap_ci(casing_vals),
        "lot_positive_tasks": len(lot_pos_vals),
        "mean_lot_recall": lot_mean,
        "lot_recall_ci95": _bootstrap_ci(lot_pos_vals),
        "mean_formation_tops_f1": sum(tops_vals) / count,
        "formation_tops_f1_ci95": _bootstrap_ci(tops_vals),
        "dst_positive_tasks": len(dst_pos_vals),
        "mean_dst_f1_non_empty": dst_pos_f1,
        "dst_f1_non_empty_ci95": dst_pos_ci,
        "per_task": results,
    }


def _as_float(val: float | str | None) -> float | None:
    if val is None or val == "":
        return None
    return float(val)
