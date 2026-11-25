"""Exercise the Studio pause/resume path without modifying a Frisco account."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from frisco_agent.basket import add_to_basket, ingredient_labels, selected_products
from frisco_agent.graph import builder
from frisco_agent.models import Ingredient, ShoppingList


def test_studio_review_validates_then_updates_existing_basket():
    async def run():
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "basket-test"}}
        parser = AsyncMock()
        parser.ainvoke.return_value = ShoppingList(ingredients=[
            Ingredient(name="tofu", search_query="tofu", quantity=4, unit="pack")])
        offers = [{"product_id": "X", "name": "Plain tofu"},
                  {"product_id": "Y", "name": "Smoked tofu", "pack_price_grosze": 579}]
        basket = {"milk": 1, "Y": 1}

        async def set_counts(targets):
            basket.update(targets)
            # Simulate Frisco applying the PUT before the CLI loses its response.
            raise TimeoutError

        with (
            patch("frisco_agent.graph.create_parser", return_value=parser),
            patch("frisco_agent.graph.open_product_search"),
            patch("frisco_agent.graph.find_candidates", new_callable=AsyncMock, return_value=offers),
            patch("frisco_agent.basket.open_frisco_tool"),
            patch("frisco_agent.basket.read_basket", new_callable=AsyncMock,
                  side_effect=lambda tool: basket.copy()) as read,
            patch("frisco_agent.basket.set_basket_quantities", side_effect=set_counts) as write,
        ):
            result = await graph.ainvoke({"ingredients_text": "4 x tofu"}, config)
            prompt = result["__interrupt__"][0].value
            assert prompt["approve_format"]["items"][0]["ingredient"] == "tofu"
            assert prompt["rows"][0]["ingredient"] == "tofu"
            options = prompt["rows"][0]["options"]
            assert options[1]["pack_count"] == 4
            assert options[1]["pack_price"] == "5,79 zł"
            assert options[1]["total"] == "23,16 zł"
            assert "total_grosze" not in options[1]
            assert options[0]["total"] is None
            assert result["product_searches"][0]["candidates"][1]["total_grosze"] == 2316
            read.assert_not_awaited()
            write.assert_not_awaited()

            def approve(option):
                return Command(resume={"action": "approve", "items": [{"ingredient": "Tofu", "option": option, "packs": 2}]})

            invalid = await graph.ainvoke(approve(99), config)
            assert "error" in invalid["__interrupt__"][0].value
            read.assert_not_awaited()
            result = await graph.ainvoke(approve(2), config)
            assert result["basket_result"] == {"status": "verified", "lines": [
                {"product_id": "Y", "name": "Smoked tofu", "packs_to_add": 2, "before": 1, "target": 3}]}
            assert basket == {"milk": 1, "Y": 3}
            write.assert_awaited_once_with({"Y": 3})

            state = await graph.aget_state(config)
            assert await add_to_basket(state.values) == {"basket_result": result["basket_result"]}
            write.assert_awaited_once()
            for reply in [{"action": "cancel"}, {"action": "approve", "items": []}]:
                paused = await graph.ainvoke({"ingredients_text": "4 x tofu"}, config)
                assert paused["basket_result"] is None
                result = await graph.ainvoke(Command(resume=reply), config)
                assert result["basket_result"] == {"status": "skipped", "lines": []}
            write.assert_awaited_once()

    asyncio.run(run())


def test_selection_merges_same_product_from_two_rows():
    row = {"candidates": [{"product_id": "x", "name": "Tofu"}]}
    searches = [{**row, "ingredient": "plain tofu"}, {**row, "ingredient": "tofu"}]
    # Older Studio interrupts used row numbers.
    items = [{"row": i, "option": 1, "packs": i} for i in (1, 2)]
    assert selected_products({"action": "approve", "items": items}, searches) == {"x": ("Tofu", 3)}


def test_repeated_names_use_the_displayed_labels():
    searches = [
        {"ingredient": "eggs", "quantity": qty, "unit": unit,
         "candidates": [{"product_id": unit, "name": "Eggs"}]}
        for qty, unit in [(2, "pack"), (5, "piece")]
    ]
    assert ingredient_labels(searches) == ["eggs (2 pack)", "eggs (5 piece)"]

    def choose(*names):
        items = [{"ingredient": name, "option": 1, "packs": 1} for name in names]
        return selected_products({"action": "approve", "items": items}, searches)

    assert choose(" Eggs   (5 piece) ") == {"piece": ("Eggs", 1)}
    with pytest.raises(ValueError, match="label"):
        choose("eggs")
    with pytest.raises(ValueError, match="only once"):
        choose("eggs (2 pack)", "EGGS (2 pack)")


@pytest.mark.parametrize("change", [
    {"row": 0}, {"option": -1}, {"packs": 0}, {"packs": True}, {"packs": 1.5}, {"product_id": "other"},
])
def test_bad_review_cannot_write_unlisted_products_or_invalid_counts(change):
    item = {"ingredient": "tofu", "option": 1, "packs": 2, **change}
    with pytest.raises(ValueError):
        selected_products({"action": "approve", "items": [item]},
                          [{"ingredient": "tofu", "candidates": [{"product_id": "x", "name": "Tofu"}]}])


@pytest.mark.parametrize(("reads", "expected_error"), [
    ([{}, {"x": 2}], None),  # An initially empty basket.
    ([{"x": 1}], "changed"),  # A different quantity appeared after approval.
    ([{}, {"x": 1}], "different quantities"),  # Frisco limited the requested quantity.
    ([{"milk": 1}, {"x": 2}], "Another basket line changed"),
])
def test_batch_read_back(reads, expected_error):
    state = {"basket_plan": [{"product_id": "x", "name": "Tofu", "before": 0,
                              "packs_to_add": 2, "target": 2}]}
    with (
        patch("frisco_agent.basket.open_frisco_tool"),
        patch("frisco_agent.basket.read_basket", side_effect=reads),
        patch("frisco_agent.basket.set_basket_quantities", return_value=True) as write,
    ):
        if expected_error:
            with pytest.raises(RuntimeError, match=expected_error):
                asyncio.run(add_to_basket(state))
        else:
            assert asyncio.run(add_to_basket(state))["basket_result"]["status"] == "verified"
        if len(reads) == 1:
            write.assert_not_awaited()
        else:
            write.assert_awaited_once_with({"x": 2})
