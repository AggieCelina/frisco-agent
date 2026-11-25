import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from frisco_agent.graph import search_products
from frisco_agent.integrations.frisco import read_package_size


@pytest.mark.parametrize(("pl", "en", "category", "count"), [
    ("Jaja kurze (rozmiar L) - 10 szt.", "Eggs size L", "19118", 10),
    ("Jaja ekologiczne", "Organic eggs – 6 pieces", "19118", 6),
    ("Jaja 8 szt.", "Eggs 6 pcs", "19118", None),
    ("Jaja, rozmiar L", "Eggs", "19118", None),
    ("Jaja 2 x 6 szt.", "Eggs", "19118", None),
    ("Jaja 6–8 szt.", "Eggs", "19118", None),
    ("Czekoladowe jajka 6 szt.", "Chocolate eggs", "chocolate", None),
])
def test_carton_needs_an_unambiguous_contents_label(pl, en, category, count):
    size = read_package_size({"name": {"pl": pl, "en": en},
                              "primaryCategory": {"parentId": category},
                              "unitOfMeasure": "Piece", "grammage": 1, "packSize": 1})
    assert size == ({"unit": "piece", "quantity": count} if count else None)


@pytest.mark.parametrize(("needed", "price_for_eight", "winner", "six_count", "eight_count"), [
    (5, 800, "eight", 1, 1), (5, 960, "six", 1, 1), (5, 840, "six", 1, 1),
    (7, 960, "eight", 2, 1), (14, 960, "eight", 3, 2),
])
def test_compare_whole_cartons(needed, price_for_eight, winner, six_count, eight_count):
    offers = [{"product_id": name, "name": name,
               "package_size": {"quantity": size, "unit": "piece"} if size else None,
               "pack_price_grosze": price}
              for name, size, price in [("unknown", None, 1), ("six", 6, 840), ("eight", 8, price_for_eight)]]
    with (patch("frisco_agent.graph.open_product_search"),
          patch("frisco_agent.graph.find_candidates", new_callable=AsyncMock, return_value=offers)):
        state = asyncio.run(search_products({"shopping_list": [
            {"name": "eggs", "search_query": "jajka", "unit": "piece", "quantity": needed}
        ]}))
    search = state["product_searches"][0]
    assert search["lowest_cost_carton_id"] == winner
    unknown, six, eight = search["candidates"]
    assert unknown["total_grosze"] is None
    for candidate, count, size, price in [(six, six_count, 6, 840), (eight, eight_count, 8, price_for_eight)]:
        assert candidate["pack_count"] == count
        assert candidate["total_grosze"] == count * price
        assert candidate["leftover_pieces"] == count * size - needed
    assert six["price_per_piece_grosze"] == 140
