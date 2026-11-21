import orjson

from wellbore_sft.schema import (
    ground_truth_to_extraction,
    parse_prediction_json,
)
from wellbore_sft.scorer import (
    compute_rlvr_reward,
    normalize_formation_name,
    parse_casing_inches,
)
from wellbore_sft.train_grpo import make_grpo_reward_fn


def _sample_task() -> dict:
    return {
        "task_id": "ncs_test_1",
        "is_vertical_approx": True,
        "ground_truth": {
            "header": {
                "kelly_bushing_m": 25.0,
                "water_depth_m": 120.0,
                "total_depth_md_rkb_m": 3200.0,
                "final_vertical_depth_tvd_rkb_m": 3198.0,
            },
            "casing_and_lot": [
                {
                    "wlbCasingDiameter": "13 3/8",
                    "wlbCasingDepth": 1500.0,
                    "wlbLotMudDencity": 1.65,
                },
                {
                    "wlbCasingDiameter": "9 5/8",
                    "wlbCasingDepth": 2800.0,
                    "wlbLotMudDencity": 1.82,
                },
            ],
            "formation_tops": [
                {"lsuName": "NORDLAND GP", "lsuTopDepth": 145.0},
                {"lsuName": "BALDER FM", "lsuTopDepth": 1820.0},
            ],
            "dst": [
                {"wlbDstFromDepth": 2950.0, "wlbDstToDepth": 2980.0},
            ],
        },
    }


def test_parse_casing_and_formation_normalization() -> None:
    assert abs((parse_casing_inches('13 3/8"') or 0.0) - 13.375) < 1e-6
    assert abs((parse_casing_inches("9⅝") or 0.0) - 9.625) < 1e-6
    assert normalize_formation_name("Åre Formation") == "ARE"
    assert normalize_formation_name("Nordland Gp") == "NORDLAND"


def test_parse_prediction_handles_think_and_fences() -> None:
    raw = '<think>checking tables</think>\n```json\n{"header": {"kelly_bushing_m": 25.0}}\n```'
    parsed = parse_prediction_json(raw, "ncs_test_1")
    assert parsed.valid_json
    assert parsed.valid_schema
    assert parsed.prediction is not None
    assert parsed.prediction.header.kelly_bushing_m == 25.0


def test_parse_prediction_coerces_null_lot_unit_and_lists() -> None:
    raw = (
        '{"header": null, "casing_and_lot": [{"casing_diameter_in": "30", '
        '"shoe_depth": {"value": 450.0, "unit": "m", "datum": "md_rkb"}, '
        '"lot_mud_weight": null, "lot_unit": null}], "formation_tops": null, "dst": null}'
    )
    parsed = parse_prediction_json(raw, "ncs_test_1")
    assert parsed.valid_schema
    assert parsed.prediction is not None
    assert parsed.prediction.casing_and_lot[0].lot_unit == "g/cm3"
    assert parsed.prediction.formation_tops == []
    assert parsed.prediction.dst == []


def test_oracle_gets_perfect_rlvr_reward() -> None:
    task = _sample_task()
    extraction = ground_truth_to_extraction(task["ground_truth"])
    raw = orjson.dumps(extraction.model_dump(mode="json")).decode()
    breakdown = compute_rlvr_reward(raw, task)
    assert breakdown.valid_json
    assert breakdown.valid_schema
    assert breakdown.reward == 1.0
    assert breakdown.bloat_penalty == 0.0


def test_invalid_json_gets_zero_reward() -> None:
    task = _sample_task()
    breakdown = compute_rlvr_reward("not json at all", task)
    assert not breakdown.valid_json
    assert breakdown.reward == 0.0


def test_grpo_reward_function_batch() -> None:
    task = _sample_task()
    extraction = ground_truth_to_extraction(task["ground_truth"])
    good = orjson.dumps(extraction.model_dump(mode="json")).decode()
    reward_fn = make_grpo_reward_fn()
    scores = reward_fn(
        completions=[[{"role": "assistant", "content": good}], "bad output"],
        task_id=[task["task_id"], task["task_id"]],
        ground_truth=[task["ground_truth"], task["ground_truth"]],
        is_vertical_approx=[True, True],
    )
    assert scores[0] == 1.0
    assert scores[1] == 0.0
