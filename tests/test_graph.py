"""Offline graph checks: no API calls or credits required."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from frisco_agent.graph import consolidate_ingredients, graph, packs_needed
from frisco_agent.integrations.frisco import read_package_size, read_pack_price
from frisco_agent.models import Ingredient, ShoppingList


def test_graph_returns_serializable_ingredients():
    parser = AsyncMock()
    parser.ainvoke.return_value = ShoppingList(ingredients=[
        Ingredient(name="Greek yogurt", search_query="jogurt grecki", quantity=600, unit="g"),
        Ingredient(name="salt", search_query="sól", quantity=None, unit=None),
    ])
    # Exercise the real graph and response parsing. Only the model and the
    # external MCP tool are stand-ins, so the test needs no account or network.
    tool = AsyncMock()
    tool.name = "products_search"
    rows = [{"productId": 17, "product": {"name": {"pl": "Jogurt naturalny"},
                                         "grammage": 0.4, "unitOfMeasure": "Kilogram",
                                         "price": {"price": 4.35}}}]
    tool.ainvoke.side_effect = [
        SimpleNamespace(status="success", artifact={"structured_content": {
            "ok": True, "data": {"api_response": {"products": products}}
        }}) for products in [rows, []]
    ]
    with (
        patch("frisco_agent.graph.create_parser", return_value=parser),
        patch("frisco_agent.graph.open_product_search") as connection,
    ):
        connection.return_value.__aenter__.return_value = tool
        result = asyncio.run(graph.ainvoke({
            "ingredients_text": "600 g Greek yogurt\nsalt to taste"
        }))
    assert result["ingredients"] == [
        {"name": "Greek yogurt", "quantity": 600.0, "unit": "g", "search_query": "jogurt grecki"},
        {"name": "salt", "quantity": None, "unit": None, "search_query": "sól"},
    ]
    assert [call.args[0]["args"]["search"] for call in tool.ainvoke.await_args_list] == [
        "jogurt grecki", "sól",
    ]
    assert result["product_searches"] == [
        {"ingredient": "Greek yogurt", "query": "jogurt grecki", "quantity": 600, "unit": "g",
         "candidates": [{"product_id": "17", "name": "Jogurt naturalny",
                         "package_size": {"quantity": 400, "unit": "g"}, "pack_count": 2,
                         "pack_price_grosze": 435, "total_grosze": 870}]},
        {"ingredient": "salt", "query": "sól", "quantity": None, "unit": None, "candidates": []},
    ]
    json.dumps(result)
    parser.ainvoke.assert_awaited_once()


def test_explicit_pack_count_survives_unknown_size():
    assert packs_needed(4, "pack", None) == 4
    assert packs_needed(400, "g", None) is None


def test_two_recipes_search_once_for_five_eggs():
    parser = AsyncMock()
    parser.ainvoke.return_value = ShoppingList(ingredients=[
        Ingredient(name="eggs", search_query=query, quantity=count, unit="piece")
        for query, count in [("jajka", 2), ("  Jajka  ", 3)]
    ])
    with (
        patch("frisco_agent.graph.create_parser", return_value=parser),
        patch("frisco_agent.graph.open_product_search") as connection,
        patch("frisco_agent.graph.find_candidates", new_callable=AsyncMock, return_value=[]) as search,
    ):
        result = asyncio.run(graph.ainvoke({"ingredients_text": "Omelette: 2 eggs. Cake: 3 eggs."}))
    assert [item["quantity"] for item in result["ingredients"]] == [2, 3]
    assert result["shopping_list"] == [{**result["ingredients"][0], "quantity": 5}]
    assert result["product_searches"][0]["quantity"] == 5
    search.assert_awaited_once_with(connection.return_value.__aenter__.return_value, "jajka")


def test_only_compatible_known_requirements_merge():
    entries = [
        ("jogurt naturalny", 500, "g"), ("jogurt naturalny", 0.5, "kg"),
        ("jajka", 2, "piece"), ("jajka", 1, "pack"),
        ("tofu naturalne", 200, "g"), ("tofu wędzone", 200, "g"),
        ("mąka", 100, "g"), ("mąka", None, "g"),
        ("mleko", 0.25, "l"), ("mleko", 150, "ml"), ("mleko", 50, "g"),
    ]
    ingredients = [dict(name=q, search_query=q, quantity=n, unit=u) for q, n, u in entries]
    items = consolidate_ingredients({"ingredients": ingredients})["shopping_list"]
    assert [(i["name"], i["quantity"], i["unit"]) for i in items] == [
        ("jogurt naturalny", 1000, "g"), *entries[2:8], ("mleko", 400, "ml"), entries[-1],
    ]


def test_missing_price_is_distinct_from_a_zero_price():
    assert read_pack_price({}) is None
    assert read_pack_price({"price": {"price": -1}}) is None
    assert read_pack_price({"price": {"price": 0}}) == 0


def test_package_volume_and_incompatible_units():
    size = read_package_size({"grammage": 0.25, "unitOfMeasure": "Litre"})
    assert size == {"quantity": 250, "unit": "ml"}
    assert packs_needed(0.75, "l", size) == 3
    assert packs_needed(400, "g", size) is None
    assert read_package_size({"grammage": 1, "unitOfMeasure": "Piece"}) is None
    assert read_package_size({}) is None


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_input_does_not_call_model(text):
    with patch("frisco_agent.graph.create_parser") as factory:
        with pytest.raises(ValueError, match="non-empty ingredients_text"):
            asyncio.run(graph.ainvoke({"ingredients_text": text}))
        factory.assert_not_called()


def test_refusal_is_not_treated_as_empty_list():
    parser = AsyncMock()
    parser.ainvoke.return_value = None
    with patch("frisco_agent.graph.create_parser", return_value=parser):
        with pytest.raises(ValueError, match="did not return"):
            asyncio.run(graph.ainvoke({"ingredients_text": "600 g yogurt"}))


def test_server_configuration_loads_graph_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "langgraph.json").read_text())
    filename, attribute = config["graphs"]["frisco"].split(":")
    spec = importlib.util.spec_from_file_location("configured_graph", root / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = getattr(module, attribute)
    assert "parse_ingredients" in loaded.get_graph().nodes
