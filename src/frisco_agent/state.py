"""Data passed into, between, and out of graph steps."""

from typing_extensions import NotRequired, TypedDict


class GraphInput(TypedDict):
    """Fields a caller supplies when starting a run."""

    # TypedDict describes dictionary keys/types for tooling and LangGraph.
    # It is not runtime validation like Pydantic; the node checks this input.
    ingredients_text: str


class IngredientData(TypedDict):
    """An extracted ingredient represented as an ordinary dictionary."""

    name: str
    search_query: NotRequired[str]
    # `| None` permits an unknown value; None becomes null in JSON.
    quantity: float | None
    unit: str | None


class PackageSize(TypedDict):
    quantity: float
    unit: str


class ProductCandidate(TypedDict):
    product_id: str
    name: str
    package_size: NotRequired[PackageSize | None]
    # Money stays in whole grosze (100 grosze = 1 zł).
    pack_price_grosze: NotRequired[int | None]


class ShoppingCandidate(ProductCandidate):
    # Depends on both the catalog product and the user's requested quantity.
    pack_count: int | None
    total_grosze: int | None


class ProductSearch(TypedDict):
    ingredient: str
    query: str
    quantity: float | None
    unit: str | None
    candidates: list[ShoppingCandidate]


class GraphOutput(TypedDict):
    """Fields returned to the caller after the graph finishes."""

    ingredients: list[IngredientData]
    product_searches: list[ProductSearch]


class ShoppingState(GraphInput, GraphOutput):
    """Combined internal state: original text plus extracted ingredients.

    Inheriting from both schemas collects their fields. The ingredients field
    is populated by our node during execution; the caller need not supply it.
    `pass` means this class adds no further fields or behavior.
    """
    pass
