"""Connect the recipe graph to the local Frisco MCP process."""

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from decimal import Decimal
from importlib import import_module
from math import isfinite
from pathlib import Path

from fastmcp.client.transports import StdioTransport
from langchain.mcp import MCPAdapter
from langsmith import tracing_context

from frisco_agent.state import PackageSize, ProductCandidate


def frisco_binary() -> Path:
    # The default matches our two sibling project folders. Other installations
    # can supply an absolute binary path in .env.
    default = Path(__file__).parents[4] / "frisco-mcp/bin/frisco"
    return Path(os.getenv("FRISCO_MCP_BINARY") or default).expanduser()


@asynccontextmanager
async def open_frisco_tool(name: str):
    """Start the local MCP server and expose only the requested tool."""
    binary = frisco_binary()
    # MCP validates the first tool result with jsonschema. Its first import
    # reads bundled schema files, so load it off the server's event loop.
    await asyncio.to_thread(import_module, "jsonschema")
    transport = StdioTransport(command=str(binary), args=["mcp"], keep_alive=False)
    async with AsyncExitStack() as stack:
        try:
            adapter = await stack.enter_async_context(MCPAdapter(transport))
        except RuntimeError as exc:
            # FastMCP wraps the operating system's startup error as its cause.
            cause = exc.__cause__
            if isinstance(cause, (FileNotFoundError, PermissionError)):
                raise RuntimeError(
                    f"Cannot start Frisco CLI at {binary}: {cause.strerror}. "
                    "Set FRISCO_MCP_BINARY to the path of an executable Frisco CLI."
                ) from exc
            raise
        for tool in await adapter.list_tools():
            if tool.name == name:
                yield tool
                return
        raise RuntimeError(f"The Frisco MCP server did not advertise {name}.")


def open_product_search():
    return open_frisco_tool("products_search")


# The Frisco category tree places egg cartons under 19118 ("Jaja").
EGG_CATEGORY = "19118"
EGG_COUNT = re.compile(r"\b(\d+)\s*(?:szt(?:uk[aiy]?)?|pcs?|pieces?|eggs?|jaj(?:a|ek)?)\b", re.I)


def read_egg_carton(product: dict) -> PackageSize | None:
    category = product.get("primaryCategory") or {}
    path = str(category.get("parentPath", "")).split(",")
    if EGG_CATEGORY not in [str(category.get("categoryId")), str(category.get("parentId")), *path]:
        return None

    # Both translations describe the same pack. An omitted count is fine, but
    # contradictory counts, ranges and multipacks need further product details.
    names = product.get("name")
    labels = names.values() if isinstance(names, dict) else [names]
    counts = set()
    for label in labels:
        if not isinstance(label, str):
            continue
        matches = list(EGG_COUNT.finditer(label))
        if not matches:
            continue
        if re.search(r"\d\s*[-–/+×x.,]\s*\d|\d\s*(?:szt\.?|pcs\.?)\s*[+×x]\s*\d", label, re.I):
            return None
        counts.update(int(match[1]) for match in matches)
    if len(counts) == 1:
        count = counts.pop()
        if count > 0:
            return PackageSize(quantity=count, unit="piece")
    return None


def read_package_size(product: dict) -> PackageSize | None:
    """Read the weight, volume or verified egg count of one sale item."""
    if product.get("unitOfMeasure") == "Piece":
        return read_egg_carton(product)
    amount = product.get("grammage")
    units = {"Kilogram": ("g", 1000), "Gram": ("g", 1),
             "Litre": ("ml", 1000), "Millilitre": ("ml", 1)}
    measure = units.get(product.get("unitOfMeasure"))
    if measure is None or type(amount) not in (float, int) or not isfinite(amount) or amount <= 0:
        return None
    unit, multiplier = measure
    return PackageSize(quantity=float(Decimal(str(amount)) * multiplier), unit=unit)


async def call_frisco(tool, arguments: dict) -> dict:
    """Keep raw Frisco responses (including account metadata) out of graph traces."""
    with tracing_context(enabled=False):
        message = await tool.ainvoke({
            "type": "tool_call", "name": tool.name, "id": tool.name, "args": arguments,
        }, config={"callbacks": []})
    if message.status == "error" or not message.artifact:
        raise RuntimeError(f"Frisco {tool.name} failed. Check the saved Frisco session.")
    payload = message.artifact["structured_content"]
    if not payload.get("ok"):
        raise RuntimeError(f"Frisco reported an unsuccessful {tool.name} call.")
    return payload["data"]["api_response"]


async def read_basket(tool) -> dict[str, int]:
    """Read product IDs and quantities only. An unreadable cart must block a write."""
    response = await call_frisco(tool, {})
    rows = response.get("products")
    if not isinstance(rows, list):
        raise ValueError("Frisco returned a basket without a products list.")
    counts = {}
    for row in rows:
        product_id = row.get("productId") or (row.get("product") or {}).get("productId")
        quantity = row.get("quantity")
        if type(product_id) not in (int, str) or not str(product_id).strip():
            raise ValueError("A basket line has no product ID.")
        if type(quantity) is not int or quantity < 0 or str(product_id) in counts:
            raise ValueError("A basket line has an ambiguous quantity.")
        counts[str(product_id)] = quantity
    return {pid: qty for pid, qty in counts.items() if qty > 0}


async def set_basket_quantities(targets: dict[str, int]) -> bool:
    """Pass final counts to the CLI, which merges them with its own fresh cart read."""
    lines = [{"product_id": pid, "quantity": qty} for pid, qty in targets.items()]
    process = await asyncio.create_subprocess_exec(
        str(frisco_binary()), "cart", "add-batch", "--file", "/dev/stdin", "--format", "json",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(process.communicate(json.dumps(lines).encode()), timeout=60)
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
    return process.returncode == 0


async def find_candidates(tool, query: str) -> list[ProductCandidate]:
    """Invoke the LangChain tool and keep the catalog fields our graph needs."""
    response = await call_frisco(tool, {"search": query, "page_size": 5})
    # An empty products array is a valid search with zero matches. A missing
    # field is a different outcome and should surface as a response format error.
    candidates = []
    for row in response["products"]:
        name = row["product"]["name"]
        if isinstance(name, dict):
            name = name.get("pl") or name.get("en")
        product_id = row["productId"]
        if not product_id or not isinstance(name, str) or not name.strip():
            raise ValueError("Frisco returned a product without an ID or display name.")
        product = row["product"]
        # Only offer products the catalog allows the user to buy.
        size = read_package_size(product)
        unavailable = product.get("isAvailable") is False or product.get("isStocked") is False
        if unavailable:
            continue
        candidates.append(ProductCandidate(product_id=str(product_id), name=name,
                                           package_size=size,
                                           pack_price_grosze=read_pack_price(product)))
    return candidates


def read_pack_price(product: dict) -> int | None:
    """Convert Frisco's nested catalog price from PLN to exact whole grosze."""
    price = product.get("price")
    amount = price.get("price") if isinstance(price, dict) else None
    if type(amount) not in (int, float) or not isfinite(amount) or amount < 0:
        return None
    grosze = Decimal(str(amount)) * 100
    return int(grosze) if grosze == grosze.to_integral_value() else None
