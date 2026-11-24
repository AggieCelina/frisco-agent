"""Connect the recipe graph to the local Frisco MCP process."""

import asyncio
import os
from contextlib import AsyncExitStack, asynccontextmanager
from decimal import Decimal
from importlib import import_module
from math import isfinite
from pathlib import Path

from fastmcp.client.transports import StdioTransport
from langchain.mcp import MCPAdapter
from langsmith import tracing_context

from frisco_agent.state import PackageSize, ProductCandidate


@asynccontextmanager
async def open_product_search():
    """Start one server for this batch of ingredients and release it afterwards."""
    # The default matches our two sibling project folders. Other installations
    # can supply an absolute binary path in .env.
    default = Path(__file__).parents[4] / "frisco-mcp/bin/frisco"
    binary = Path(os.getenv("FRISCO_MCP_BINARY") or default).expanduser()
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
        # The server advertises other tools too. This node only needs search.
        for tool in await adapter.list_tools():
            if tool.name == "products_search":
                yield tool
                return
        raise RuntimeError("The Frisco MCP server did not advertise products_search.")


def read_package_size(product: dict) -> PackageSize | None:
    """Read the weight or volume of one sale item from the catalog."""
    amount = product.get("grammage")
    units = {"Kilogram": ("g", 1000), "Gram": ("g", 1),
             "Litre": ("ml", 1000), "Millilitre": ("ml", 1)}
    measure = units.get(product.get("unitOfMeasure"))
    # A carton of ten eggs may say "1 Piece": that counts cartons, not eggs.
    if measure is None or type(amount) not in (float, int) or not isfinite(amount) or amount <= 0:
        return None
    unit, multiplier = measure
    return PackageSize(quantity=float(Decimal(str(amount)) * multiplier), unit=unit)


async def find_candidates(tool, query: str) -> list[ProductCandidate]:
    """Invoke the LangChain tool and keep the catalog fields our graph needs."""
    # Passing a tool call gives us a ToolMessage with structured MCP data.
    # A raw response includes session-related metadata. Keep it out of traces.
    with tracing_context(enabled=False):
        message = await tool.ainvoke({
            "type": "tool_call", "name": tool.name, "id": "product-search",
            "args": {"search": query, "page_size": 5},
        }, config={"callbacks": []})
    if message.status == "error" or not message.artifact:
        raise RuntimeError("Frisco product search failed. Check the saved Frisco session.")
    payload = message.artifact["structured_content"]
    if not payload.get("ok"):
        raise RuntimeError("Frisco reported an unsuccessful product search.")
    # An empty products array is a valid search with zero matches. A missing
    # field is a different outcome and should surface as a response format error.
    candidates = []
    for row in payload["data"]["api_response"]["products"]:
        name = row["product"]["name"]
        if isinstance(name, dict):
            name = name.get("pl") or name.get("en")
        product_id = row["productId"]
        if not product_id or not isinstance(name, str) or not name.strip():
            raise ValueError("Frisco returned a product without an ID or display name.")
        product = row["product"]
        candidates.append(ProductCandidate(product_id=str(product_id), name=name,
                                           package_size=read_package_size(product),
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
