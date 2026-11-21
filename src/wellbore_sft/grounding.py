import copy
from typing import Any

from wellbore_sft.config import RewardWeights
from wellbore_sft.schema import WellborePrediction, parse_prediction_json
from wellbore_sft.scorer import (
    RewardBreakdown,
    ScoreTolerances,
    compute_rlvr_reward,
    normalize_depth_to_md_rkb,
)


def _depth_in_text(depth_m: float | None, text: str) -> bool:
    if depth_m is None:
        return False
    rounded = int(round(depth_m))
    for candidate in (rounded, rounded - 1, rounded + 1):
        if candidate > 0 and str(candidate) in text:
            return True
    return False


def filter_ground_truth_to_document(
    ground_truth: dict[str, Any],
    pages_markdown: str,
) -> dict[str, Any]:
    grounded = copy.deepcopy(ground_truth)
    grounded["casing_and_lot"] = [
        row
        for row in (ground_truth.get("casing_and_lot") or [])
        if row.get("wlbCasingDepth") is not None
        and _depth_in_text(float(row["wlbCasingDepth"]), pages_markdown)
    ]
    grounded["formation_tops"] = [
        row
        for row in (ground_truth.get("formation_tops") or [])
        if row.get("lsuTopDepth") is not None
        and _depth_in_text(float(row["lsuTopDepth"]), pages_markdown)
    ]
    grounded["dst"] = [
        row
        for row in (ground_truth.get("dst") or [])
        if row.get("wlbDstFromDepth") is not None
        and row.get("wlbDstToDepth") is not None
        and (
            _depth_in_text(float(row["wlbDstFromDepth"]), pages_markdown)
            or _depth_in_text(float(row["wlbDstToDepth"]), pages_markdown)
        )
    ]
    return grounded


def compute_ungrounded_prediction_rate(
    pred: WellborePrediction,
    pages_markdown: str,
    kb_m: float | None,
    is_vertical: bool,
) -> float:
    total = 0
    ungrounded = 0
    for item in pred.casing_and_lot:
        d = normalize_depth_to_md_rkb(item.shoe_depth, kb_m, is_vertical)
        if d is not None:
            total += 1
            if not _depth_in_text(d, pages_markdown):
                ungrounded += 1
    for item in pred.formation_tops:
        d = normalize_depth_to_md_rkb(item.top_depth, kb_m, is_vertical)
        if d is not None:
            total += 1
            if not _depth_in_text(d, pages_markdown):
                ungrounded += 1
    for item in pred.dst:
        d1 = normalize_depth_to_md_rkb(item.from_depth, kb_m, is_vertical)
        d2 = normalize_depth_to_md_rkb(item.to_depth, kb_m, is_vertical)
        if d1 is not None and d2 is not None:
            total += 1
            if not (_depth_in_text(d1, pages_markdown) or _depth_in_text(d2, pages_markdown)):
                ungrounded += 1
    return (ungrounded / total) if total > 0 else 0.0


def compute_grounded_rlvr_reward(
    raw_completion: str,
    task: dict[str, Any],
    pages_markdown: str,
    weights: RewardWeights | None = None,
    tolerances: ScoreTolerances | None = None,
    hallucination_penalty_weight: float = 0.20,
) -> tuple[RewardBreakdown, float]:
    grounded_task = dict(task)
    grounded_task["ground_truth"] = filter_ground_truth_to_document(
        task["ground_truth"], pages_markdown
    )
    breakdown = compute_rlvr_reward(raw_completion, grounded_task, weights, tolerances)
    if not breakdown.valid_schema:
        return breakdown, 0.0

    parsed = parse_prediction_json(raw_completion, str(task["task_id"]))
    assert parsed.prediction is not None
    kb_raw = (task["ground_truth"].get("header") or {}).get("kelly_bushing_m")
    kb_m = float(kb_raw) if kb_raw is not None else None
    is_vert = bool(task.get("is_vertical_approx", False))
    ungrounded_rate = compute_ungrounded_prediction_rate(
        parsed.prediction, pages_markdown, kb_m, is_vert
    )
    adjusted = max(0.0, breakdown.reward - hallucination_penalty_weight * ungrounded_rate)
    return breakdown.model_copy(update={"reward": round(adjusted, 6)}), round(ungrounded_rate, 6)
