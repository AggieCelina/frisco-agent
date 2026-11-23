# Frisco agent

Turn pasted recipe ingredients into a reviewed Frisco shopping basket.
Checkout remains manual. The graph extracts ingredients and searches the Frisco catalog. Product matching, basket operations, and the shopping UI are still pending.

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

Flow: `START → parse_ingredients → search_products → END`. The parsing node
makes one OpenAI request to extract ingredients and Polish search phrases.
The search node invokes the Frisco MCP tool once per ingredient.

After activating the environment, start the local development server:

```sh
langgraph dev --no-browser
```

Select graph `frisco` in Studio's **Graph mode** (it is not a chat graph),
or invoke it through the API with this input:

```json
{"ingredients_text": "600 g Greek yogurt\n2 eggs\nsalt to taste"}
```

The output contains `ingredients` and `product_searches`. Each search records
the ingredient, the Polish query, and up to five candidates with a product ID
and display name. An empty candidate list means the catalog returned no matches.
A failed search fails the run. The graph only reads the catalog.

Run offline checks with `python -m pytest -q` from the activated environment.
