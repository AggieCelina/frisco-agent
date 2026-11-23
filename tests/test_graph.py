"""Offline graph checks: no API calls or credits required."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from frisco_agent.graph import graph
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
    rows = [{"productId": 17, "product": {"name": {"pl": "Jogurt naturalny"}}}]
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
        {"ingredient": "Greek yogurt", "query": "jogurt grecki",
         "candidates": [{"product_id": "17", "name": "Jogurt naturalny"}]},
        {"ingredient": "salt", "query": "sól", "candidates": []},
    ]
    json.dumps(result)
    parser.ainvoke.assert_awaited_once()


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
