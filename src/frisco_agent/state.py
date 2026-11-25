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
    leftover_pieces: NotRequired[int]
    price_per_piece_grosze: NotRequired[int | None]


class ProductSearch(TypedDict):
    ingredient: str
    query: str
    quantity: float | None
    unit: str | None
    candidates: list[ShoppingCandidate]
    lowest_cost_carton_id: NotRequired[str | None]


class BasketLine(TypedDict):
    product_id: str
    name: str
    packs_to_add: int
    before: int
    target: int


class BasketResult(TypedDict):
    status: str
    lines: list[BasketLine]


class GraphOutput(TypedDict):
    """Fields returned to the caller after the graph finishes."""

    ingredients: list[IngredientData]
    shopping_list: list[IngredientData]
    product_searches: list[ProductSearch]
    basket_result: NotRequired[BasketResult | None]


class ShoppingState(GraphInput, GraphOutput):
    """The reviewed targets are checkpointed before the node that writes to Frisco."""

    basket_plan: list[BasketLine]
