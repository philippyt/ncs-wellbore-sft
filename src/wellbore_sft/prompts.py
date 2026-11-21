from typing import Any

import orjson

from wellbore_sft.schema import WellboreExtraction

SYSTEM_PROMPT_OPEN_BOOK = (
    "You are an expert Norwegian Continental Shelf drilling and subsurface data extractor. "
    "Read the OCR Markdown pages from a single wellbore completion report and return a single valid JSON object "
    "with keys header, casing_and_lot, formation_tops, and dst. "
    "Extract actual as drilled values rather than prognosed or planned values. "
    "Use units m or ft and datums md_rkb, tvd_rkb, or tvd_msl. "
    "Use lot_unit g/cm3, sg, or ppg when leak off or formation integrity test values are present, else null. "
    "Return only the JSON object with no surrounding prose."
)

SYSTEM_PROMPT_CLOSED_BOOK = (
    "You are evaluating prior memorization of Norwegian Continental Shelf SODIR FactPages records. "
    "Given only the wellbore identifier and basic metadata without any report text, "
    "return a single valid JSON object with keys header, casing_and_lot, formation_tops, and dst. "
    "If you do not know a field, leave it null or an empty list. "
    "Return only the JSON object with no surrounding prose."
)

SCHEMA_TEMPLATE = {
    "header": {
        "kelly_bushing_m": "float or null",
        "water_depth_m": "float or null",
        "total_depth": {"value": "float", "unit": "m|ft", "datum": "md_rkb|tvd_rkb|tvd_msl"},
        "final_vertical_depth": {
            "value": "float",
            "unit": "m|ft",
            "datum": "tvd_rkb|tvd_msl",
        },
    },
    "casing_and_lot": [
        {
            "casing_diameter_in": "string or float",
            "shoe_depth": {"value": "float", "unit": "m|ft", "datum": "md_rkb"},
            "lot_mud_weight": "float or null",
            "lot_unit": "g/cm3|sg|ppg",
        }
    ],
    "formation_tops": [
        {
            "surface_name": "string",
            "top_depth": {"value": "float", "unit": "m|ft", "datum": "md_rkb"},
        }
    ],
    "dst": [
        {
            "from_depth": {"value": "float", "unit": "m|ft", "datum": "md_rkb"},
            "to_depth": {"value": "float", "unit": "m|ft", "datum": "md_rkb"},
        }
    ],
}

SCHEMA_TEXT = orjson.dumps(SCHEMA_TEMPLATE).decode()


def build_extraction_messages(
    wellbore_name: str,
    doc_id: str,
    pages_markdown: str,
    target: WellboreExtraction | None = None,
) -> list[dict[str, str]]:
    user_content = (
        f"Wellbore: {wellbore_name}\n"
        f"Document ID: {doc_id}\n"
        f"Target JSON schema: {SCHEMA_TEXT}\n\n"
        f"Report OCR pages:\n{pages_markdown}"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_OPEN_BOOK},
        {"role": "user", "content": user_content},
    ]
    if target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": orjson.dumps(target.model_dump(mode="json")).decode(),
            }
        )
    return messages


def build_closed_book_messages(
    wellbore_name: str,
    production_licence: Any = None,
    completion_year: Any = None,
) -> list[dict[str, str]]:
    user_content = (
        f"Wellbore: {wellbore_name}\n"
        f"Production licence: {production_licence}\n"
        f"Completion year: {completion_year}\n"
        f"Target JSON schema: {SCHEMA_TEXT}\n\n"
        "Return the JSON object from memory only."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_CLOSED_BOOK},
        {"role": "user", "content": user_content},
    ]
