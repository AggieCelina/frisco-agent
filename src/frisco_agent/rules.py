"""Shopping preferences evaluated against comparable, already suitable products."""

from math import isfinite

from frisco_agent.state import ProductCandidate


class YogurtOffer(ProductCandidate):
    # Total price of enough whole packages for the recipe, in hundredths of a złoty.
    total_grosze: int | None
    protein_g_per_100g: float | None


def choose_yogurt(offers: list[YogurtOffer], *, prefer_more_protein: bool) -> YogurtOffer | None:
    """Return a suggestion, or None if the available facts cannot settle the choice."""
    if not offers or any(o["total_grosze"] is None or o["total_grosze"] < 0 for o in offers):
        return None

    # Sorting by total cost also breaks protein ties in favor of the cheaper offer.
    by_price = sorted(offers, key=lambda o: o["total_grosze"])
    cheapest = by_price[0]
    if not prefer_more_protein:
        return cheapest

    ceiling = cheapest["total_grosze"] + 100
    eligible = [offer for offer in by_price if offer["total_grosze"] <= ceiling]
    if len(eligible) == 1:
        return cheapest

    for offer in eligible:
        protein = offer["protein_g_per_100g"]
        if protein is None or not isfinite(protein) or not 0 <= protein <= 100:
            return None
    return max(eligible, key=lambda o: o["protein_g_per_100g"])
