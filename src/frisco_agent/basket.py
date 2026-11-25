"""Review catalog matches in Studio and apply confirmed pack counts to the basket."""

import asyncio
from collections import Counter
from typing import Literal

from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

from frisco_agent.integrations.frisco import open_frisco_tool, read_basket, set_basket_quantities
from frisco_agent.state import BasketLine, ProductSearch, ShoppingState


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    option: PositiveInt
    packs: PositiveInt


class IngredientChoice(Choice):
    ingredient: str


class NumberedChoice(Choice):
    """Allow threads that already showed the old numbered prompt to resume."""

    row: PositiveInt


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["approve"]
    items: list[IngredientChoice | NumberedChoice]


def format_price(grosze: int | None) -> str | None:
    """Display a nonnegative catalog price using Polish currency formatting."""
    if grosze is None:
        return None
    zlote, remainder = divmod(grosze, 100)
    return f"{zlote},{remainder:02d} zł"


def name_key(name: str) -> str:
    return " ".join(name.casefold().split())


def ingredient_labels(searches: list[ProductSearch]) -> list[str]:
    """Give repeated names distinct labels while keeping ordinary names simple."""
    names = [search["ingredient"] for search in searches]
    counts = Counter(name_key(name) for name in names)
    reserved = set(counts)
    labels = []
    for name, search in zip(names, searches):
        label = name
        if counts[name_key(name)] > 1:
            quantity = search.get("quantity")
            amount = f"{quantity:g} {search.get('unit') or ''}".strip() if quantity is not None else ""
            base = f"{name} ({amount})" if amount else name
            label = base
            number = 1
            while name_key(label) in reserved:
                label = f"{base} #{number}"
                number += 1
            reserved.add(name_key(label))
        labels.append(label)
    return labels


def selected_products(reply: object, searches: list[ProductSearch]) -> dict[str, tuple[str, int]]:
    """Resolve ingredient labels and their numbered options against the saved offers."""
    approval = Approval.model_validate(reply)
    labels = ingredient_labels(searches)
    by_name = {name_key(label): index for index, label in enumerate(labels)}
    selected = {}
    seen_rows = set()
    for choice in approval.items:
        if isinstance(choice, IngredientChoice):
            index = by_name.get(name_key(choice.ingredient))
            if index is None:
                raise ValueError("Use an ingredient label shown in the review.")
        else:
            index = choice.row - 1
            if index >= len(searches):
                raise ValueError("Choose a listed ingredient.")
        if index in seen_rows:
            raise ValueError(f"Choose {labels[index]} only once.")
        seen_rows.add(index)
        offers = searches[index]["candidates"]
        if choice.option > len(offers):
            raise ValueError(f"Choose a listed option for {labels[index]}.")
        offer = offers[choice.option - 1]
        pid = offer["product_id"]
        # Different shopping-list rows can point to the same catalog product.
        _, previous = selected.get(pid, (offer["name"], 0))
        selected[pid] = (offer["name"], previous + choice.packs)
    return selected


async def review_selection(state: ShoppingState) -> dict:
    """Pause for a choice. Only read the account after the user approves products."""
    searches = state["product_searches"]
    if not any(search["candidates"] for search in searches):
        return {"basket_plan": [], "basket_result": {"status": "no_matches", "lines": []}}

    rows = []
    for label, search in zip(ingredient_labels(searches), searches):
        options = []
        for number, candidate in enumerate(search["candidates"], start=1):
            option = {"option": number, **candidate}
            # Format the display copy. Search results retain integer prices for comparisons.
            for field in ("pack_price_grosze", "total_grosze", "price_per_piece_grosze"):
                if field in option:
                    option[field.removesuffix("_grosze")] = format_price(option.pop(field))
            if candidate["product_id"] == search.get("lowest_cost_carton_id"):
                option["suggestion"] = "Lowest total cost among comparable cartons for this ingredient."
            options.append(option)
        rows.append({"ingredient": label, "quantity": search["quantity"],
                     "unit": search["unit"], "options": options})

    prompt = {
        "instructions": "Use the ingredient name and choose one of its numbered options. "
                        "Omit ingredients to skip them. "
                        "Enter whole packs to ADD.",
        "rows": rows,
        "approve_format": {"action": "approve", "items": [{
            "ingredient": next(row["ingredient"] for row in rows if row["options"]),
            "option": 1, "packs": 1,
        }]},
        "cancel": {"action": "cancel"},
    }
    # LangGraph replays the node on resume. All code before interrupt is read-only.
    while True:
        reply = interrupt(prompt)
        if reply == {"action": "cancel"}:
            selected = {}
            break
        try:
            selected = selected_products(reply, searches)
            break
        except ValidationError:
            prompt = {**prompt, "error": "Use the approve or cancel format shown. "
                                         "Supply an ingredient name and positive whole numbers for option and packs."}
        except ValueError as exc:
            prompt = {**prompt, "error": str(exc)}

    if not selected:
        return {"basket_plan": [], "basket_result": {"status": "skipped", "lines": []}}

    async with open_frisco_tool("cart_show") as tool:
        basket = await read_basket(tool)
    plan = [
        BasketLine(product_id=pid, name=name, packs_to_add=packs,
                   before=basket.get(pid, 0), target=basket.get(pid, 0) + packs)
        for pid, (name, packs) in selected.items()
    ]
    return {"basket_plan": plan}


def after_review(state: ShoppingState) -> Literal["add_to_basket", "__end__"]:
    return "add_to_basket" if state["basket_plan"] else END


# The local Agent Server can run several threads. Serialize our read/write/read
# sequences within this process because the CLI updates a whole account basket.
_basket_lock = asyncio.Lock()


async def add_to_basket(state: ShoppingState) -> dict:
    """Apply saved absolute targets and verify them, including after an uncertain write."""
    plan = state["basket_plan"]
    if not plan:
        raise ValueError("A confirmed basket plan is required.")
    targets = {line["product_id"]: line["target"] for line in plan}
    async with _basket_lock, open_frisco_tool("cart_show") as tool:
        current = await read_basket(tool)
        # A retry after a lost reply may find the original targets already reached.
        if not all(current.get(pid, 0) == qty for pid, qty in targets.items()):
            if any(current.get(line["product_id"], 0) != line["before"] for line in plan):
                raise RuntimeError("Selected basket quantities changed. Review a fresh run in Studio.")
            try:
                reported_success = await set_basket_quantities(targets)
            except TimeoutError:
                reported_success = False
            observed = await read_basket(tool)
            if any(observed.get(pid) != qty for pid, qty in targets.items()):
                if not reported_success:
                    raise RuntimeError("Frisco did not confirm the update. Check the basket before retrying this run.")
                raise RuntimeError("Frisco returned different quantities. Check the basket before retrying this run.")
            if any(observed.get(pid) != qty for pid, qty in current.items() if pid not in targets):
                raise RuntimeError("Another basket line changed during the update. Check your basket in Frisco.")
    return {"basket_result": {"status": "verified", "lines": plan}}
