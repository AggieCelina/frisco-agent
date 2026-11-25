# Frisco agent

Paste recipe text, a shopping note such as `4 x tofu and toilet paper`, or both into
`ingredients_text`. The graph extracts shopping items and searches the Frisco catalog.
Recipe links and photo uploads are planned. For now, paste the text itself.
Review your choices in Studio to add products to your signed-in Frisco basket.
Complete checkout yourself on Frisco. The Streamlit interface is planned.

## Environment

```sh
source langchain-env/bin/activate
python -m pip install -e '.[dev]'
```

`pyproject.toml` declares dependencies. `requirements.lock.txt` records the
installed versions from the initial local setup. To recreate those versions:

```sh
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
```

## Structure

- `src/frisco_agent/graph.py`: graph construction, ingredient parsing, and product search.
- `src/frisco_agent/basket.py` handles Studio review, approvals and basket verification.
- `src/frisco_agent/state.py`: graph input, output, and shared state.
- `src/frisco_agent/models.py`: structured ingredient extraction schema.
- `src/frisco_agent/integrations/frisco.py`: connection to the local Frisco MCP server.
- `ui/`: Streamlit interface, calling the Agent Server API.
- `.env.example`: configuration names; copy to `.env` and fill credentials locally.
- `langchain-env/`: local Python environment, excluded from Git.

`langgraph.json` registers the `frisco` graph and loads `.env`. The development
server is supplied by `langgraph-cli[inmem]`; it is not a production deployment.
Studio is a web interface, not a separate Python package. LangSmith's SDK is
installed for tracing and evaluation.

## Configuration

- OpenAI is configured through `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`; the initial model is `gpt-5.4-mini`. Run `python scripts/check_openai.py` for a small paid connection/structured-output check.
- Build the rrudol/frisco CLI in a sibling folder at `../frisco-mcp/bin/frisco`, or set `FRISCO_MCP_BINARY` in `.env` to its absolute path. Log in with `../frisco-mcp/bin/frisco session login` (substitute your binary path if needed).

The MCP integration uses LangChain's `mcp` extra (currently beta). The MCP server
may be a Go or Node process; it need not share this Python environment.

## Minimal graph

`parse_ingredients → consolidate_ingredients → search_products → review_selection`

Approval continues to `add_to_basket`. Skipping every item or cancelling ends the run. The parsing node
makes one OpenAI request to extract ingredients and Polish search phrases.
The consolidation node combines known amounts when their Polish search phrases match
after ignoring case and extra whitespace. It converts kg to g and l to ml, and
keeps individual pieces, packs, and unknown amounts separate. The search node
invokes the Frisco MCP tool once per resulting shopping-list entry.

After activating the environment, start the local development server:

```sh
langgraph dev --no-browser
```

Select graph `frisco` in Studio's **Graph mode** (it is not a chat graph),
or invoke it through the API with this input:

```json
{"ingredients_text": "4 x tofu\n600 g Greek yogurt"}
```

The output contains the original `ingredients`, consolidated `shopping_list`,
and `product_searches`. For example, two recipes calling for 2 eggs and 3 eggs
produce one shopping-list entry for 5 pieces. Each search records
the ingredient, the Polish query, requested quantity and unit, and up to five
candidates with an ID, name, package size and proposed pack count. For
`4 x tofu`, each candidate has `pack_count: 4`. For `600 g Greek yogurt`, a
400 g tub has `pack_count: 2`. Each candidate includes the catalog price for one
pack (`pack_price_grosze`) and the estimated cost of the proposed packs
(`total_grosze`). Both use whole grosze, with 100 grosze to 1 zł. The total is
null if either price or count is unknown. Package sizes use grams, millilitres,
or individual eggs when an egg-carton label explicitly gives its contents.
Unknown or conflicting carton counts stay null. For a request in individual
eggs, comparable cartons show leftovers and price per egg (rounded to the
nearest grosz). For a generic egg search such as `jajka`, `lowest_cost_carton_id`
identifies the cheapest covering option among the returned cartons with known
counts and prices. Ties favor fewer leftovers. Each option uses one carton size.
An empty candidate list means the catalog returned no matches.
A failed search fails the run.

## Review and add in Studio

Start a run with your shopping text. When the graph pauses at `review_selection`,
look under `rows` for each ingredient and its numbered options. The review displays
prices in złoty, for example `5,79 zł` per pack and `23,16 zł` for four packs. Check the name,
package size, price and proposed pack count. A null pack count means you decide
the quantity. A comparable egg carton can carry a lowest-cost suggestion.

In the interrupt's **resume** input on the same thread, enter a JSON object.
For example, to add two packs of the second tofu option, enter:

```json
{"action":"approve","items":[{"ingredient":"tofu","option":2,"packs":2}]}
```

Use the displayed ingredient name. Repeated names have distinct labels, and
capitalization and extra spaces are ignored. Omit ingredients you want to skip. To finish
without adding anything, resume with `{"action":"cancel"}`.
An invalid choice pauses again so you can correct it.

`packs` means packs to **add**. If you already have one of that product and
approve two, the target is three. The review node saves that target. The
`add_to_basket` node sends the targets together through the Frisco CLI's
`cart add-batch` command, which merges them with the current basket.
The node reads back the quantities using the MCP tool `cart_show`.
On success, `basket_result.status` is `verified`. Each receipt line includes
the product name, the previous count and the verified target. Refresh the basket
on Frisco while signed into the account used by the CLI.

If a basket update is interrupted, retry the failed run on the same checkpoint.
A fresh run is a new request to add packs. The saved targets let a retry recognize
quantities already reached. A conflicting quantity requires a fresh review.
Let one basket update finish before editing the same basket elsewhere, since
the CLI reads and writes it in separate requests.

Run offline checks with `python -m pytest -q` from the activated environment.
