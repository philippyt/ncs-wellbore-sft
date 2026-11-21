import re
from typing import Any, Literal

import orjson
from pydantic import BaseModel, Field, ValidationError

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class DepthValue(BaseModel):
    value: float
    unit: Literal["m", "ft"] = "m"
    datum: Literal["md_rkb", "tvd_msl", "tvd_rkb"] = "md_rkb"


class HeaderPrediction(BaseModel):
    kelly_bushing_m: float | None = None
    water_depth_m: float | None = None
    total_depth: DepthValue | None = None
    final_vertical_depth: DepthValue | None = None


class CasingPrediction(BaseModel):
    casing_diameter_in: str | float
    shoe_depth: DepthValue
    lot_mud_weight: float | None = None
    lot_unit: Literal["g/cm3", "sg", "ppg"] = "g/cm3"


class FormationTopPrediction(BaseModel):
    surface_name: str
    top_depth: DepthValue


class DstPrediction(BaseModel):
    from_depth: DepthValue
    to_depth: DepthValue


class WellboreExtraction(BaseModel):
    header: HeaderPrediction = Field(default_factory=HeaderPrediction)
    casing_and_lot: list[CasingPrediction] = Field(default_factory=list)
    formation_tops: list[FormationTopPrediction] = Field(default_factory=list)
    dst: list[DstPrediction] = Field(default_factory=list)


class WellborePrediction(WellboreExtraction):
    task_id: str


class ParsedCompletion(BaseModel):
    prediction: WellborePrediction | None = None
    valid_json: bool = False
    valid_schema: bool = False
    error: str | None = None


def _as_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def ground_truth_to_extraction(ground_truth: dict[str, Any]) -> WellboreExtraction:
    raw_header = ground_truth.get("header") or {}
    kb = _as_float(raw_header.get("kelly_bushing_m"))
    wd = _as_float(raw_header.get("water_depth_m"))
    td = _as_float(raw_header.get("total_depth_md_rkb_m"))
    tvd = _as_float(raw_header.get("final_vertical_depth_tvd_rkb_m"))

    header = HeaderPrediction(
        kelly_bushing_m=kb,
        water_depth_m=wd,
        total_depth=DepthValue(value=td, unit="m", datum="md_rkb") if td is not None else None,
        final_vertical_depth=(
            DepthValue(value=tvd, unit="m", datum="tvd_rkb") if tvd is not None else None
        ),
    )

    casing_items: list[CasingPrediction] = []
    for row in ground_truth.get("casing_and_lot") or []:
        diam = row.get("wlbCasingDiameter")
        depth = _as_float(row.get("wlbCasingDepth"))
        if diam is None or str(diam).strip() == "" or depth is None:
            continue
        lot = _as_float(row.get("wlbLotMudDencity"))
        if lot is not None and lot <= 0.0:
            lot = None
        casing_items.append(
            CasingPrediction(
                casing_diameter_in=str(diam).strip(),
                shoe_depth=DepthValue(value=depth, unit="m", datum="md_rkb"),
                lot_mud_weight=lot,
                lot_unit="g/cm3",
            )
        )
    casing_items.sort(key=lambda item: item.shoe_depth.value)

    tops_items: list[FormationTopPrediction] = []
    for row in ground_truth.get("formation_tops") or []:
        name = str(row.get("lsuName") or "").strip()
        depth = _as_float(row.get("lsuTopDepth"))
        if not name or depth is None:
            continue
        tops_items.append(
            FormationTopPrediction(
                surface_name=name,
                top_depth=DepthValue(value=depth, unit="m", datum="md_rkb"),
            )
        )
    tops_items.sort(key=lambda item: (item.top_depth.value, item.surface_name))

    dst_items: list[DstPrediction] = []
    for row in ground_truth.get("dst") or []:
        top = _as_float(row.get("wlbDstFromDepth"))
        bottom = _as_float(row.get("wlbDstToDepth"))
        if top is None or bottom is None:
            continue
        low, high = min(top, bottom), max(top, bottom)
        dst_items.append(
            DstPrediction(
                from_depth=DepthValue(value=low, unit="m", datum="md_rkb"),
                to_depth=DepthValue(value=high, unit="m", datum="md_rkb"),
            )
        )
    dst_items.sort(key=lambda item: (item.from_depth.value, item.to_depth.value))

    return WellboreExtraction(
        header=header,
        casing_and_lot=casing_items,
        formation_tops=tops_items,
        dst=dst_items,
    )


def strip_completion_wrapper(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", text).strip()
    fenced = FENCED_JSON_RE.search(cleaned)
    if fenced:
        return fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_prediction_json(raw_text: str, task_id: str) -> ParsedCompletion:
    candidate = strip_completion_wrapper(raw_text)
    if not candidate:
        return ParsedCompletion(error="empty_completion")
    try:
        payload = orjson.loads(candidate)
    except orjson.JSONDecodeError as exc:
        return ParsedCompletion(error=f"json_decode_error: {exc}")
    if not isinstance(payload, dict):
        return ParsedCompletion(valid_json=True, error="json_root_not_object")
    payload = dict(payload)
    payload["task_id"] = task_id
    try:
        pred = WellborePrediction.model_validate(payload)
    except ValidationError as exc:
        return ParsedCompletion(valid_json=True, error=f"schema_validation_error: {exc}")
    return ParsedCompletion(prediction=pred, valid_json=True, valid_schema=True)
