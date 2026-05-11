"""Unit tests for nutrition parsing from Markdown parents."""

from nutriai.chunking.markdown_converter import recipe_to_markdown
from nutriai.data.sample import SAMPLE_RECIPES
from nutriai.macro_parse import parse_nutrients_from_markdown


def test_parse_shepherds_pie_nutrients():
    md = recipe_to_markdown(SAMPLE_RECIPES[0])
    n = parse_nutrients_from_markdown(md)
    assert n["calories"] == 520.0
    assert n["protein_g"] == 28.0
    assert n["fat_g"] == 27.0
    assert n["carbs_g"] == 38.0


def test_parse_empty_without_nutrition_section():
    assert parse_nutrients_from_markdown("# Title\n\n## Ingredients\n- x") == {}
