import pytest

from frisco_agent.rules import choose_yogurt


@pytest.mark.parametrize(("price", "protein", "enabled", "expected"), [
    (1100, 7, True, "B"),     # Exactly 1 zł extra is within the allowance.
    (1101, 7, True, "A"),
    (1080, 7, False, "A"),
    (1080, 4, True, "A"),     # Same protein: prefer the cheaper offer.
    (1080, None, True, None),
    (1200, None, True, "A"),  # An out-of-budget offer needs no nutrition label.
    (None, 7, True, None),
])
def test_choose_yogurt(price, protein, enabled, expected):
    offers = [
        {"product_id": "B", "name": "B", "total_grosze": price, "protein_g_per_100g": protein},
        {"product_id": "A", "name": "A", "total_grosze": 1000, "protein_g_per_100g": 4},
    ]
    winner = choose_yogurt(offers, prefer_more_protein=enabled)
    assert (winner["product_id"] if winner else None) == expected
