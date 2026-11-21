import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from wellbore_sft.config import PackConfig

TargetCategory = Literal["header", "casing_and_lot", "formation_tops", "dst"]

FIGURE_DESC_RE = re.compile(
    r"\b(?:this\s+(?:figure|chart|graph|plot|diagram)\s+(?:shows|illustrates|depicts|presents)"
    r"|the\s+(?:x|y)[- ]?axis\s+(?:represents|shows|displays))\b",
    re.IGNORECASE,
)

LOOKALIKE_MASKS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:prognos(?:ed|is)|planned|proposed)\s+(?:td|total\s+depth|casing|tops?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:kick[- ]?off|kop|fluid\s+loss|lost\s+time|casing\s+tally)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:offset\s+well|reference\s+well|neighbour(?:ing)?\s+well)\b",
        re.IGNORECASE,
    ),
)

CATEGORY_PATTERNS: dict[TargetCategory, tuple[tuple[re.Pattern[str], float], ...]] = {
    "header": (
        (re.compile(r"\bwell\s+data\s+(?:record|summary|sheet)\b", re.IGNORECASE), 3.0),
        (re.compile(r"\b(?:kelly\s+bushing|\bKBE\b|RKB\s*[-/]\s*MSL)\b", re.IGNORECASE), 2.5),
        (re.compile(r"\bwater\s+depth\b", re.IGNORECASE), 2.0),
        (re.compile(r"\b(?:total\s+depth|final\s+vertical\s+depth)\b", re.IGNORECASE), 2.0),
        (
            re.compile(
                r"\b(?:spud\s+date|rig\s+released|production\s+licen[cs]e)\b", re.IGNORECASE
            ),
            1.5,
        ),
    ),
    "casing_and_lot": (
        (
            re.compile(
                r"\b(?:casing\s+(?:summary|details|program|strings?)|casing\s+and\s+cement)",
                re.IGNORECASE,
            ),
            3.0,
        ),
        (
            re.compile(
                r"\b(?:leak[- ]?off\s+test|formation\s+integrity\s+test|\bLOT\b|\bFIT\b)\b",
                re.IGNORECASE,
            ),
            2.5,
        ),
        (
            re.compile(
                r"\b(?:13\s*3/8|9\s*5/8|20|30)\s*\"?\s*(?:casing|csg|shoe)\b", re.IGNORECASE
            ),
            2.5,
        ),
        (
            re.compile(r"\b(?:shoe\s+at|casing\s+shoe|equivalent\s+mud\s+weight)\b", re.IGNORECASE),
            2.0,
        ),
    ),
    "formation_tops": (
        (
            re.compile(
                r"\b(?:table\s+of\s+lithostratigraphy|lithostratigraph(?:y|ic)\s+tops?)\b",
                re.IGNORECASE,
            ),
            3.5,
        ),
        (
            re.compile(
                r"\b(?:formation\s+tops?|stratigraphic\s+summary|geological\s+markers?)\b",
                re.IGNORECASE,
            ),
            3.0,
        ),
        (
            re.compile(
                r"\b(?:nordland|hordaland|rogaland|shetland|viking|cromer\s+knoll)\s+(?:gp|group)\b",
                re.IGNORECASE,
            ),
            2.0,
        ),
        (
            re.compile(
                r"\b(?:balder|sele|lista|ekofisk|tor|draupne|kobbe|kolje|knurr)\s+(?:fm|formation)\b",
                re.IGNORECASE,
            ),
            1.5,
        ),
    ),
    "dst": (
        (
            re.compile(
                r"\b(?:drill\s+stem\s+test|production\s+test\s+summary|\bDST\s*#?\s*\d)\b",
                re.IGNORECASE,
            ),
            3.5,
        ),
        (
            re.compile(
                r"\b(?:perforated\s+interval|test\s+interval|choke\s+size|flow\s+period)\b",
                re.IGNORECASE,
            ),
            2.0,
        ),
        (
            re.compile(
                r"\b(?:oil\s+rate|gas\s+rate|gor|downhole\s+temperature|shut[- ]?in\s+pressure)\b",
                re.IGNORECASE,
            ),
            1.5,
        ),
    ),
}

TAG_BOOSTS: dict[TargetCategory, frozenset[str]] = {
    "header": frozenset({"well_data", "well_summary", "well_history"}),
    "casing_and_lot": frozenset({"casing", "cementing", "formation_integrity", "leak_off"}),
    "formation_tops": frozenset({"lithostratigraphy", "formation_tops", "stratigraphy"}),
    "dst": frozenset({"dst", "well_testing", "production_test"}),
}


class PageRoute(BaseModel):
    page_no: int
    scores: dict[TargetCategory, float] = Field(default_factory=dict)
    max_score: float = 0.0
    matched_categories: list[TargetCategory] = Field(default_factory=list)


def mask_lookalikes(text: str) -> str:
    masked = text
    for pattern in LOOKALIKE_MASKS:
        masked = pattern.sub(" ", masked)
    return masked


def score_page(
    page_no: int,
    ocr_markdown: str,
    ai_tags: list[str] | None = None,
    has_tables: bool = False,
    threshold: float = 3.0,
) -> PageRoute:
    masked = mask_lookalikes(ocr_markdown or "")
    has_fig_desc = bool(FIGURE_DESC_RE.search(ocr_markdown or ""))
    tag_set = {t.lower().strip() for t in (ai_tags or [])}
    scores: dict[TargetCategory, float] = {}
    matched: list[TargetCategory] = []

    for category, patterns in CATEGORY_PATTERNS.items():
        total = 0.0
        for pat, weight in patterns:
            hits = min(len(pat.findall(masked)), 2)
            total += hits * weight
        if total > 0.0:
            if tag_set & TAG_BOOSTS[category]:
                total += 1.0
            if has_tables:
                total += 0.5
            if has_fig_desc:
                total = max(0.0, total - 1.0)
        scores[category] = round(total, 2)
        if total >= threshold:
            matched.append(category)

    return PageRoute(
        page_no=page_no,
        scores=scores,
        max_score=max(scores.values(), default=0.0),
        matched_categories=matched,
    )


def select_pages(
    pages: list[dict[str, Any]], config: PackConfig | None = None
) -> list[dict[str, Any]]:
    cfg = config or PackConfig()
    filtered = [
        p
        for p in pages
        if (not cfg.drop_ocr_loops or not bool(p.get("is_ocr_loop", False)))
        and str(p.get("ocr_markdown") or "").strip()
    ]
    filtered.sort(key=lambda item: int(item["page_no"]))

    if cfg.mode == "full":
        selected_rows = filtered[: cfg.max_pages] if cfg.max_pages > 0 else filtered
    else:
        by_no = {int(p["page_no"]): p for p in filtered}
        routes = [
            score_page(
                page_no=int(p["page_no"]),
                ocr_markdown=str(p.get("ocr_markdown") or ""),
                ai_tags=[str(t) for t in (p.get("ai_tags") or [])],
                has_tables=bool(p.get("has_tables", False)),
                threshold=cfg.threshold,
            )
            for p in filtered
        ]
        chosen: set[int] = {r.page_no for r in routes if r.page_no < cfg.always_include_first_n}
        for category in CATEGORY_PATTERNS:
            ranked = sorted(
                (r for r in routes if r.scores.get(category, 0.0) >= cfg.threshold),
                key=lambda r: (r.scores.get(category, 0.0), -r.page_no),
                reverse=True,
            )
            for r in ranked[: cfg.max_pages_per_category]:
                chosen.add(r.page_no)

        if len(chosen) > cfg.max_pages:
            priority_order = sorted(
                (r for r in routes if r.page_no in chosen),
                key=lambda r: (r.page_no < cfg.always_include_first_n, r.max_score, -r.page_no),
                reverse=True,
            )
            chosen = {r.page_no for r in priority_order[: cfg.max_pages]}

        selected_rows = [by_no[no] for no in sorted(chosen) if no in by_no]

    packed: list[dict[str, Any]] = []
    used_chars = 0
    for row in selected_rows:
        text = str(row.get("ocr_markdown") or "").strip()
        if used_chars + len(text) > cfg.max_chars and packed:
            break
        packed.append(row)
        used_chars += len(text)
    return packed


def render_pages_markdown(pages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for page in pages:
        page_no = int(page["page_no"])
        text = str(page.get("ocr_markdown") or "").strip()
        blocks.append(f"[PAGE {page_no}]\n{text}")
    return "\n\n".join(blocks)
