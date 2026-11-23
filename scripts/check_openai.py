"""Make one small paid request to verify credentials and native structured output."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Ingredient(BaseModel):
    name: str
    quantity: float
    unit: str


def main() -> None:
    # A quoted positional argument becomes one message, even with spaces.
    # nargs="?" makes it optional; --help displays usage without an API call.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "message",
        nargs="?",
        default="Extract this ingredient: 600 g of Greek yogurt.",
        help="Message to send to the model (describe a single ingredient).",
    )
    args = parser.parse_args()
    if not args.message.strip():
        parser.error("message must not be blank")

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY in .env")
    model = os.getenv("OPENAI_MODEL") or "gpt-5.4-mini"
    # Disable tracing for this standalone connection check.
    os.environ["LANGSMITH_TRACING"] = "false"
    llm = ChatOpenAI(
        model=model,
        use_responses_api=True,
        reasoning_effort="low",
        max_tokens=512,
        timeout=60,
        max_retries=0,
    )
    structured = llm.with_structured_output(
        Ingredient, method="json_schema", strict=True, include_raw=True
    )
    try:
        response = structured.invoke(args.message)
    except Exception as exc:
        # Avoid printing provider error messages that might echo credentials.
        raise SystemExit(f"API check failed: {type(exc).__name__}") from None
    ingredient = response["parsed"]
    if response["parsing_error"] or ingredient is None:
        raise SystemExit("No valid structured output returned.")
    print(f"Model: {model}")
    print(ingredient.model_dump_json(indent=2))
    print(f"Token usage: {response['raw'].usage_metadata}")
    print("Connection and native structured-output check passed.")


if __name__ == "__main__":
    main()
