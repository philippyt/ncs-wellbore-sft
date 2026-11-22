from wellbore_sft.config import PackConfig
from wellbore_sft.page_select import render_pages_markdown, score_page, select_pages
from wellbore_sft.prompts import build_closed_book_messages, build_extraction_messages


def test_score_page_and_select_drops_ocr_loops() -> None:
    pages = [
        {
            "page_no": 0,
            "ocr_markdown": "Well Data Summary. Kelly Bushing 25 m. Water Depth 120 m.",
            "ai_tags": ["well_data"],
            "has_tables": True,
            "is_ocr_loop": False,
        },
        {
            "page_no": 1,
            "ocr_markdown": (
                "Casing Summary 13 3/8 casing shoe at 1500 m. Leak off test 1.65 g/cm3."
            ),
            "ai_tags": ["casing"],
            "has_tables": True,
            "is_ocr_loop": False,
        },
        {
            "page_no": 2,
            "ocr_markdown": (
                "Table of Lithostratigraphy. Nordland Group 145 m. Balder Formation 1820 m."
            ),
            "ai_tags": ["lithostratigraphy"],
            "has_tables": True,
            "is_ocr_loop": True,
        },
        {
            "page_no": 5,
            "ocr_markdown": (
                "Formation tops summary. Nordland Group 145 m. Balder Formation 1820 m."
            ),
            "ai_tags": ["formation_tops"],
            "has_tables": True,
            "is_ocr_loop": False,
        },
    ]
    route = score_page(1, pages[1]["ocr_markdown"], pages[1]["ai_tags"], True)
    assert "casing_and_lot" in route.matched_categories

    selected = select_pages(pages, PackConfig(mode="routed", max_pages=10, max_chars=50_000))
    selected_nos = [p["page_no"] for p in selected]
    assert 2 not in selected_nos
    assert 0 in selected_nos
    assert 1 in selected_nos
    assert 5 in selected_nos

    rendered = render_pages_markdown(selected)
    assert "[PAGE 0]" in rendered
    assert "[PAGE 5]" in rendered


def test_figure_description_drop_and_tag() -> None:
    pages = [
        {
            "page_no": 0,
            "ocr_markdown": "Well Data Summary. Kelly Bushing 25 m.",
            "ai_tags": ["well_data"],
            "has_tables": True,
            "is_ocr_loop": False,
            "has_figure_description": False,
        },
        {
            "page_no": 3,
            "ocr_markdown": "This figure shows casing shoe depths and leak off test 1.5 1.6 1.7.",
            "ai_tags": ["casing"],
            "has_tables": False,
            "is_ocr_loop": False,
            "has_figure_description": True,
        },
        {
            "page_no": 4,
            "ocr_markdown": "Casing Summary 13 3/8 shoe at 1500 m. This chart shows pressure.",
            "ai_tags": ["casing"],
            "has_tables": True,
            "is_ocr_loop": False,
            "has_figure_description": True,
        },
    ]
    selected = select_pages(pages, PackConfig(mode="routed", max_pages=10, max_chars=50_000))
    selected_nos = [p["page_no"] for p in selected]
    assert 3 not in selected_nos
    assert 4 in selected_nos
    rendered = render_pages_markdown(selected)
    assert "[PAGE 4 | FIGURE_DESCRIPTION]" in rendered
    assert "[PAGE 0]\n" in rendered


def test_prompt_builders_have_expected_roles() -> None:
    open_msgs = build_extraction_messages("31/2-1", "31/2-1::100", "[PAGE 0]\nhello")
    assert len(open_msgs) == 2
    assert open_msgs[0]["role"] == "system"
    assert "31/2-1" in open_msgs[1]["content"]

    closed_msgs = build_closed_book_messages("31/2-1", "PL090", 2012)
    assert len(closed_msgs) == 2
    assert "PL090" in closed_msgs[1]["content"]
