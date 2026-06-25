# LLM Proxy

A small Python proxy for an LLM API.

The proxy forwards requests to the model server and saves logs.

Default values:

- listen address: `localhost:8081`
- upstream base URL: `http://localhost:3000/v1`
- application log: `logs/app.log`
- request/response logs: `communication_logs/`
- HTML dashboard: `http://localhost:8081/_ui`

For each proxied request, the proxy writes two files:

- a readable `.log` file
- a structured `.json` file for the dashboard

## How URL Forwarding Works

The target URL is used as a base URL.

If the client calls:

```text
/chat/completions
```

the proxy forwards it to:

```text
http://localhost:3000/v1/chat/completions
```

If the client calls:

```text
/v1/chat/completions
```

the proxy forwards it to the same upstream URL. It does not add `/v1` twice.

## Start

```bash
python3 app.py
```

With custom options:

```bash
python3 app.py \
  --listen 0.0.0.0:8081 \
  --target-url http://localhost:3000/v1 \
  --logs-dir logs \
  --communication-logs-dir communication_logs
```

## Configuration

Settings come from three places, in order of precedence:

1. command-line options (highest)
2. a `config.toml` file
3. built-in defaults (lowest)

To keep local values (for example an internal upstream hostname) out of
version control, copy the example file and edit it:

```bash
cp config.example.toml config.toml
```

`config.toml` is git-ignored and never committed. Available keys:

```toml
listen = "localhost:8081"
target_url = "http://localhost:3000/v1"
app_logs_dir = "logs"
communication_logs_dir = "communication_logs"
timeout = 120.0
```

Point to a different file with `--config path/to/file.toml`.

## Python Virtual Environment

Create a local Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Run the proxy:

```bash
python app.py
```

You can also install the project in editable mode:

```bash
python -m pip install -e .
llm-proxy
```

Stop using the environment:

```bash
deactivate
```

## Dashboard

Start the proxy and open:

```text
http://localhost:8081/_ui
```

The dashboard shows:

- request list
- status filter
- text search
- request detail
- response detail

Internal API:

```text
GET /_api/exchanges
GET /_api/exchanges/<id>
```

Sensitive headers are saved as `[REDACTED]`:

- `Authorization`
- `Proxy-Authorization`
- `Cookie`
- `Set-Cookie`

## Web UI And Ollama

`http://localhost:3000/v1` is an OpenAI-compatible API base URL.

In Web UI, use an OpenAI-compatible provider. Do not use an Ollama provider for this URL.

OpenAI-compatible tools usually call:

- `GET /v1/models`
- `POST /v1/chat/completions`

Ollama tools usually call:

- `GET /api/tags`
- `POST /api/chat`
- `POST /api/generate`

Quick check:

```bash
curl http://localhost:8081/v1/models
```

## Example Request

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen3-coder-next-fp8",
    "messages": [
      {
        "role": "user",
        "content": "Rename the file graph/MyGraph.grf in sandbox MySandbox to MyGraphV2.grf"
      }
    ],
    "tool_choice": "auto"
  }'
```

## Log Files

Example communication log names:

```text
20260625T061530.123456Z_a1b2c3d4.log
20260625T061530.123456Z_a1b2c3d4.json
```

## Tests

```bash
python3 -m unittest discover -s tests
```
