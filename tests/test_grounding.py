import orjson

from wellbore_sft.grounding import (
    compute_grounded_rlvr_reward,
    filter_ground_truth_to_document,
)


def test_filter_ground_truth_and_hallucination_penalty() -> None:
    task = {
        "task_id": "ncs_g_1",
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
                }
            ],
            "formation_tops": [
                {"lsuName": "NORDLAND GP", "lsuTopDepth": 145.0},
                {"lsuName": "BALDER FM", "lsuTopDepth": 1820.0},
            ],
            "dst": [],
        },
    }
    doc_text = "Casing 13 3/8 shoe at 1500 m. Nordland Group top at 145 m."
    grounded = filter_ground_truth_to_document(task["ground_truth"], doc_text)
    assert len(grounded["formation_tops"]) == 1
    assert grounded["formation_tops"][0]["lsuName"] == "NORDLAND GP"

    pred_clean = orjson.dumps(
        {
            "header": {
                "kelly_bushing_m": 25.0,
                "water_depth_m": 120.0,
                "total_depth": {"value": 3200.0, "unit": "m", "datum": "md_rkb"},
                "final_vertical_depth": {"value": 3198.0, "unit": "m", "datum": "tvd_rkb"},
            },
            "casing_and_lot": [
                {
                    "casing_diameter_in": "13 3/8",
                    "shoe_depth": {"value": 1500.0, "unit": "m", "datum": "md_rkb"},
                    "lot_mud_weight": 1.65,
                    "lot_unit": "g/cm3",
                }
            ],
            "formation_tops": [
                {
                    "surface_name": "NORDLAND GP",
                    "top_depth": {"value": 145.0, "unit": "m", "datum": "md_rkb"},
                }
            ],
            "dst": [],
        }
    ).decode()
    bd_clean, u_clean = compute_grounded_rlvr_reward(pred_clean, task, doc_text)
    assert u_clean == 0.0
    assert bd_clean.reward == 1.0

    pred_halluc = orjson.dumps(
        {
            "header": {
                "kelly_bushing_m": 25.0,
                "water_depth_m": 120.0,
                "total_depth": {"value": 3200.0, "unit": "m", "datum": "md_rkb"},
                "final_vertical_depth": {"value": 3198.0, "unit": "m", "datum": "tvd_rkb"},
            },
            "casing_and_lot": [
                {
                    "casing_diameter_in": "13 3/8",
                    "shoe_depth": {"value": 1500.0, "unit": "m", "datum": "md_rkb"},
                    "lot_mud_weight": 1.65,
                    "lot_unit": "g/cm3",
                }
            ],
            "formation_tops": [
                {
                    "surface_name": "NORDLAND GP",
                    "top_depth": {"value": 145.0, "unit": "m", "datum": "md_rkb"},
                },
                {
                    "surface_name": "FAKE FM",
                    "top_depth": {"value": 2499.0, "unit": "m", "datum": "md_rkb"},
                },
            ],
            "dst": [],
        }
    ).decode()
    bd_halluc, u_halluc = compute_grounded_rlvr_reward(pred_halluc, task, doc_text)
    assert u_halluc > 0.0
    assert bd_halluc.reward < bd_clean.reward
