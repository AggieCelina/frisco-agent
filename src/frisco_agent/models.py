"""Initial extraction schema; product matching will be added later."""

from pydantic import BaseModel, ConfigDict


class Ingredient(BaseModel):
    """Pydantic schema used to describe and validate a model's ingredient output."""

    # Silently discard unexpected fields when validating data locally.
    # LangChain's strict=True separately enforces the OpenAI response schema.
    model_config = ConfigDict(extra="ignore")

    name: str
    search_query: str  # Polish catalog phrase, separate from the original name.
    # Fields are required, but their values can be null when unspecified.
    # For example, "salt to taste" has no numeric quantity. There is no default
    # value here: the model must explicitly supply a number or null.
    quantity: float | None
    unit: str | None


class ShoppingList(BaseModel):
    """Top-level response object containing zero or more ingredients."""

    # Apply the same rule to extra fields on the whole shopping list.
    model_config = ConfigDict(extra="ignore")

    # Pydantic validates each list item using the Ingredient schema above.
    ingredients: list[Ingredient]
