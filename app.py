#!/usr/bin/env python3
"""Small HTTP proxy for LLM chat completion requests."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit


DEFAULT_LISTEN = "localhost:8081"
DEFAULT_TARGET_URL = "http://localhost:3000/v1"
DEFAULT_APP_LOGS_DIR = "logs"
DEFAULT_COMMUNICATION_LOGS_DIR = "communication_logs"
DEFAULT_TIMEOUT = 120.0
DEFAULT_CONFIG_FILE = "config.toml"
LOGGER_NAME = "llm_proxy"
DEFAULT_EXCHANGE_LIMIT = 200
REDACTED_VALUE = "[REDACTED]"
EXCHANGE_ID_PATTERN = re.compile(r"^[0-9T._A-Za-z-]+$")

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
}


@dataclass(frozen=True)
class ProxyConfig:
    target_url: str
    communication_logs_dir: Path
    timeout: float


@dataclass(frozen=True)
class UpstreamResponse:
    status: int
    reason: str
    headers: list[tuple[str, str]]
    body: bytes


def parse_listen(value: str) -> tuple[str, int]:
    """Parse listen address in host:port or port format."""
    if not value:
        raise argparse.ArgumentTypeError("listen address cannot be empty")

    if ":" not in value:
        return "localhost", parse_port(value)

    host, port_text = value.rsplit(":", 1)
    if not host:
        host = "localhost"
    return host, parse_port(port_text)


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid port: {value}") from exc

    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError(f"port out of range: {port}")
    return port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy POST requests to an LLM endpoint and log request/response pairs."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help=(
            "Path to a TOML config file with default settings (ignored if missing). "
            "Command-line options override config file values. "
            f"Default: {DEFAULT_CONFIG_FILE}"
        ),
    )
    parser.add_argument(
        "--listen",
        default=None,
        help=f"Address to listen on, as host:port or port. Default: {DEFAULT_LISTEN}",
    )
    parser.add_argument(
        "--target-url",
        default=None,
        help=f"Upstream endpoint URL. Default: {DEFAULT_TARGET_URL}",
    )
    parser.add_argument(
        "--app-logs-dir",
        "--logs-dir",
        dest="app_logs_dir",
        default=None,
        help=f"Directory for application runtime logs. Default: {DEFAULT_APP_LOGS_DIR}",
    )
    parser.add_argument(
        "--communication-logs-dir",
        default=None,
        help=(
            "Directory for request/response communication logs. "
            f"Default: {DEFAULT_COMMUNICATION_LOGS_DIR}"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Upstream request timeout in seconds. Default: {DEFAULT_TIMEOUT:g}",
    )
    return parser.parse_args(argv)


def load_config_file(path: Path) -> dict[str, object]:
    """Load settings from a TOML config file; return {} if missing or invalid."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"Warning: ignoring config file {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(
            f"Warning: ignoring config file {path}: top level is not a table",
            file=sys.stderr,
        )
        return {}
    return data


def resolve_setting(
    cli_value: object, config: dict[str, object], key: str, default: object
) -> object:
    """Resolve one setting by precedence: CLI argument > config file > built-in default."""
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def make_handler(config: ProxyConfig) -> type[BaseHTTPRequestHandler]:
    class LlmProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/health":
                self.send_simple_response(HTTPStatus.OK, b"ok\n", "text/plain; charset=utf-8")
                return
            if path in {"/_ui", "/_ui/"}:
                self.send_simple_response(
                    HTTPStatus.OK,
                    dashboard_html().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            if path == "/_api/exchanges":
                self.handle_exchange_index()
                return
            if path.startswith("/_api/exchanges/"):
                self.handle_exchange_detail(path)
                return
            if path.startswith("/_api/"):
                self.send_json_response(
                    HTTPStatus.NOT_FOUND,
                    {"error": "not_found", "message": "Unknown API endpoint."},
                )
                return
            self.proxy_request()

        def do_HEAD(self) -> None:
            if self.path == "/health":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            self.proxy_request()

        def proxy_request(self) -> None:
            started_at = datetime.now(timezone.utc)
            started_perf = time.perf_counter()
            request_body = self.read_request_body()
            request_headers = list(self.headers.items())
            upstream_url = target_url_for_request(config.target_url, self.path)
            logger = get_logger()
            logger.info(
                "Proxying %s %s to %s (%d request bytes)",
                self.command,
                self.path,
                upstream_url,
                len(request_body),
            )

            try:
                upstream = forward_request(
                    upstream_url,
                    self.command,
                    request_headers,
                    request_body,
                    config.timeout,
                )
            except Exception as exc:  # The proxy should return a useful response and still log.
                logger.exception("Upstream request failed for %s %s", self.command, self.path)
                upstream = make_bad_gateway_response(exc)

            duration_ms = round((time.perf_counter() - started_perf) * 1000, 2)
            log_path = self.write_communication_log(
                started_at,
                duration_ms,
                upstream_url,
                request_headers,
                request_body,
                upstream,
            )
            logger.info(
                "Finished %s %s with status %d (%d response bytes); communication_log=%s",
                self.command,
                self.path,
                upstream.status,
                len(upstream.body),
                log_path if log_path else "not_written",
            )
            self.write_upstream_response(upstream)

        def read_request_body(self) -> bytes:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                return b""
            try:
                length = int(content_length)
            except ValueError:
                return b""
            if length <= 0:
                return b""
            return self.rfile.read(length)

        def write_upstream_response(self, upstream: UpstreamResponse) -> None:
            self.send_response(upstream.status, upstream.reason)
            for name, value in upstream.headers:
                if should_forward_response_header(name):
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(upstream.body)))
            self.end_headers()
            self.wfile.write(upstream.body)

        def send_simple_response(
            self, status: HTTPStatus, body: bytes, content_type: str
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def write_communication_log(
            self,
            started_at: datetime,
            duration_ms: float,
            upstream_url: str,
            request_headers: list[tuple[str, str]],
            request_body: bytes,
            upstream: UpstreamResponse,
        ) -> Path | None:
            try:
                config.communication_logs_dir.mkdir(parents=True, exist_ok=True)
                log_stem = timestamped_log_stem(started_at)
                log_path = config.communication_logs_dir / f"{log_stem}.log"
                json_path = config.communication_logs_dir / f"{log_stem}.json"
                exchange = build_exchange_record(
                    exchange_id=log_stem,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    method=self.command,
                    path=self.path,
                    target_url=upstream_url,
                    request_headers=request_headers,
                    request_body=request_body,
                    upstream=upstream,
                )
                log_path.write_text(
                    format_exchange_log(
                        exchange=exchange,
                    ),
                    encoding="utf-8",
                )
                json_path.write_text(
                    json.dumps(exchange, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return log_path
            except OSError as exc:
                get_logger().exception("Failed to write communication log")
                print(f"Failed to write exchange log: {exc}", file=sys.stderr)
                return None

        def handle_exchange_index(self) -> None:
            limit = exchange_limit_from_request(self.path)
            exchanges = list_exchange_summaries(config.communication_logs_dir, limit)
            self.send_json_response(HTTPStatus.OK, {"exchanges": exchanges})

        def handle_exchange_detail(self, path: str) -> None:
            exchange_id = unquote(path.removeprefix("/_api/exchanges/"))
            exchange = load_exchange_detail(config.communication_logs_dir, exchange_id)
            if exchange is None:
                self.send_json_response(
                    HTTPStatus.NOT_FOUND,
                    {"error": "not_found", "message": "Exchange log not found."},
                )
                return
            self.send_json_response(HTTPStatus.OK, exchange)

        def send_json_response(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_simple_response(status, body, "application/json; charset=utf-8")

        def log_message(self, _format: str, *args: object) -> None:
            get_logger().info("%s - %s", self.client_address[0], _format % args)

    return LlmProxyHandler


def forward_request(
    target_url: str,
    method: str,
    request_headers: Iterable[tuple[str, str]],
    request_body: bytes,
    timeout: float,
) -> UpstreamResponse:
    headers = {
        name: value
        for name, value in request_headers
        if should_forward_request_header(name)
    }
    data = request_body
    if method.upper() in {"GET", "HEAD"} and not request_body:
        data = None

    request = urllib.request.Request(
        target_url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return UpstreamResponse(
                status=response.status,
                reason=response.reason,
                headers=list(response.headers.items()),
                body=body,
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return UpstreamResponse(
            status=exc.code,
            reason=exc.reason,
            headers=list(exc.headers.items()),
            body=body,
        )


def target_url_for_request(base_url: str, request_target: str) -> str:
    base = urlsplit(base_url)
    request = urlsplit(request_target)
    base_path = base.path.rstrip("/")
    request_path = request.path or "/"

    if base_path and (
        request_path == base_path or request_path.startswith(f"{base_path}/")
    ):
        path = request_path
    elif request_path == "/":
        path = base_path or "/"
    elif base_path:
        path = f"{base_path}/{request_path.lstrip('/')}"
    else:
        path = request_path

    query = request.query or base.query
    return urlunsplit((base.scheme, base.netloc, path, query, ""))


def make_bad_gateway_response(exc: Exception) -> UpstreamResponse:
    body = json.dumps(
        {
            "error": "upstream_request_failed",
            "message": str(exc),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    return UpstreamResponse(
        status=HTTPStatus.BAD_GATEWAY,
        reason=HTTPStatus.BAD_GATEWAY.phrase,
        headers=[("Content-Type", "application/json; charset=utf-8")],
        body=body,
    )


def should_forward_request_header(name: str) -> bool:
    lowered = name.lower()
    return lowered not in HOP_BY_HOP_HEADERS and lowered != "accept-encoding"


def should_forward_response_header(name: str) -> bool:
    lowered = name.lower()
    return lowered not in HOP_BY_HOP_HEADERS


def timestamped_log_stem(created_at: datetime | None = None) -> str:
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def timestamped_log_name() -> str:
    return f"{timestamped_log_stem()}.log"


def build_exchange_record(
    exchange_id: str,
    started_at: datetime,
    duration_ms: float,
    method: str,
    path: str,
    target_url: str,
    request_headers: list[tuple[str, str]],
    request_body: bytes,
    upstream: UpstreamResponse,
) -> dict[str, object]:
    request_text = decode_body(request_body)
    response_text = decode_body(upstream.body)
    model = extract_model_from_body(request_text)
    return {
        "id": exchange_id,
        "timestamp": isoformat_utc(started_at),
        "method": method,
        "path": path,
        "target_url": target_url,
        "status": upstream.status,
        "reason": upstream.reason,
        "duration_ms": duration_ms,
        "model": model,
        "request_headers": headers_to_json(redact_headers(request_headers)),
        "request_body": request_text,
        "request_bytes": len(request_body),
        "response_headers": headers_to_json(redact_headers(upstream.headers)),
        "response_body": response_text,
        "response_bytes": len(upstream.body),
    }


def format_exchange_log(exchange: dict[str, object]) -> str:
    request_headers = headers_from_json(exchange.get("request_headers"))
    response_headers = headers_from_json(exchange.get("response_headers"))
    parts = [
        "=== REQUEST ===",
        f"{exchange.get('method', '')} {exchange.get('path', '')}",
        f"Timestamp: {exchange.get('timestamp', '')}",
        f"Target-URL: {exchange.get('target_url', '')}",
        f"Duration-ms: {exchange.get('duration_ms', '')}",
        format_headers(request_headers),
        "",
        str(exchange.get("request_body") or ""),
        "",
        "=== RESPONSE ===",
        f"HTTP {exchange.get('status', '')} {exchange.get('reason', '')}",
        format_headers(response_headers),
        "",
        str(exchange.get("response_body") or ""),
        "",
    ]
    return "\n".join(parts)


def format_headers(headers: Iterable[tuple[str, str]]) -> str:
    return "\n".join(f"{name}: {value}" for name, value in headers)


def redact_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    redacted = []
    for name, value in headers:
        if name.lower() in SENSITIVE_HEADERS:
            redacted.append((name, REDACTED_VALUE))
        else:
            redacted.append((name, value))
    return redacted


def headers_to_json(headers: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"name": name, "value": value} for name, value in headers]


def headers_from_json(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        return []
    headers = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        header_value = str(item.get("value", ""))
        if name:
            headers.append((name, header_value))
    return headers


def decode_body(body: bytes) -> str:
    if not body:
        return ""
    return body.decode("utf-8", errors="replace")


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_model_from_body(body: str) -> str | None:
    if not body.strip():
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if isinstance(model, str):
        return model
    return None


def exchange_limit_from_request(request_target: str) -> int:
    query = parse_qs(urlsplit(request_target).query)
    raw_limit = query.get("limit", [""])[0]
    if not raw_limit:
        return DEFAULT_EXCHANGE_LIMIT
    try:
        limit = int(raw_limit)
    except ValueError:
        return DEFAULT_EXCHANGE_LIMIT
    return max(1, min(limit, 1000))


def list_exchange_summaries(logs_dir: Path, limit: int) -> list[dict[str, object]]:
    summaries = []
    for path in sorted(logs_dir.glob("*.json"), reverse=True):
        exchange = read_exchange_json(path)
        if exchange is None:
            continue
        summaries.append(exchange_summary(exchange, path))
        if len(summaries) >= limit:
            break
    return summaries


def exchange_summary(exchange: dict[str, object], path: Path) -> dict[str, object]:
    return {
        "id": str(exchange.get("id") or path.stem),
        "timestamp": exchange.get("timestamp"),
        "method": exchange.get("method"),
        "path": exchange.get("path"),
        "target_url": exchange.get("target_url"),
        "status": exchange.get("status"),
        "duration_ms": exchange.get("duration_ms"),
        "model": exchange.get("model"),
        "request_bytes": exchange.get("request_bytes"),
        "response_bytes": exchange.get("response_bytes"),
    }


def load_exchange_detail(logs_dir: Path, exchange_id: str) -> dict[str, object] | None:
    if not exchange_id or "/" in exchange_id or not EXCHANGE_ID_PATTERN.match(exchange_id):
        return None
    return read_exchange_json(logs_dir / f"{exchange_id}.json")


def read_exchange_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def dashboard_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Proxy Logs</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #f0f4f8;
      --text: #17202a;
      --muted: #5f6b7a;
      --border: #d9e0e7;
      --accent: #2563eb;
      --ok: #15803d;
      --warn: #b45309;
      --bad: #b91c1c;
      --code: #101827;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    button, input, select {
      font: inherit;
      letter-spacing: 0;
    }

    .topbar {
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      padding: 18px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
    }

    .subtitle {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .refresh {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      min-height: 36px;
      padding: 0 14px;
      cursor: pointer;
    }

    .refresh:hover { border-color: var(--accent); }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 180px;
      gap: 12px;
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      background: var(--surface-2);
    }

    .toolbar input, .toolbar select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      padding: 0 12px;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(360px, 44%) minmax(0, 1fr);
      gap: 16px;
      padding: 16px 24px 24px;
      height: calc(100vh - 126px);
      min-height: 520px;
    }

    .panel {
      min-width: 0;
      min-height: 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }

    .rows {
      height: 100%;
      overflow: auto;
    }

    .exchange-row {
      display: grid;
      grid-template-columns: 132px 56px 1fr 70px 82px;
      gap: 10px;
      width: 100%;
      border: 0;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      padding: 11px 12px;
      text-align: left;
      cursor: pointer;
    }

    .exchange-row:hover,
    .exchange-row.selected {
      background: #eef4ff;
    }

    .exchange-row span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .time, .method, .duration, .model, .muted {
      color: var(--muted);
    }

    .method {
      font-weight: 700;
    }

    .status {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 52px;
      height: 24px;
      border-radius: 999px;
      color: #ffffff;
      font-weight: 700;
      font-size: 12px;
    }

    .status.ok { background: var(--ok); }
    .status.warn { background: var(--warn); }
    .status.bad { background: var(--bad); }
    .status.neutral { background: var(--muted); }

    .detail {
      height: 100%;
      overflow: auto;
      padding: 18px;
    }

    .empty, .loading, .error {
      padding: 18px;
      color: var(--muted);
    }

    .error { color: var(--bad); }

    .detail h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }

    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }

    .kv {
      min-width: 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 10px;
      background: #fbfcfe;
    }

    .kv b {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 3px;
    }

    .kv span {
      display: block;
      overflow-wrap: anywhere;
    }

    .section {
      margin-top: 14px;
    }

    .section h3 {
      margin: 0 0 8px;
      font-size: 14px;
    }

    pre {
      margin: 0;
      max-height: 360px;
      overflow: auto;
      border-radius: 8px;
      background: var(--code);
      color: #e5edf7;
      padding: 12px;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .body-toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
      margin-bottom: 8px;
    }

    .body-toolbar button {
      min-height: 28px;
      padding: 0 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--surface);
      color: var(--muted);
      cursor: pointer;
    }

    .body-toolbar button:hover {
      border-color: var(--accent);
      color: var(--accent);
    }

    .json-tree {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 8px 10px;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow: auto;
    }

    .json-tree details {
      margin-left: 14px;
    }

    .json-tree details.root {
      margin-left: 0;
    }

    .json-tree summary {
      cursor: pointer;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .json-tree summary::-webkit-details-marker {
      display: none;
    }

    .json-tree summary::before {
      content: "▸";
      color: var(--accent);
      font-size: 11px;
      width: 10px;
      flex: 0 0 10px;
    }

    .json-tree details[open] > summary::before {
      content: "▾";
    }

    .json-key {
      color: #0f766e;
    }

    .json-string {
      color: #1d4ed8;
      overflow-wrap: anywhere;
    }

    .json-number {
      color: #b45309;
    }

    .json-boolean {
      color: #7c3aed;
    }

    .json-null {
      color: #6b7280;
      font-style: italic;
    }

    .json-size,
    .json-bracket {
      color: var(--muted);
    }

    .json-leaf {
      margin-left: 24px;
      overflow-wrap: anywhere;
    }

    .headers {
      display: grid;
      gap: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }

    .header-line {
      display: grid;
      grid-template-columns: minmax(120px, 220px) minmax(0, 1fr);
      gap: 10px;
      padding: 6px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fbfcfe;
    }

    .header-line span {
      overflow-wrap: anywhere;
    }

    .stream-label {
      margin: 14px 0 6px;
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
    }

    .stream-meta {
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }

    .stream-raw {
      margin-top: 12px;
    }

    .stream-raw > summary {
      cursor: pointer;
      color: var(--accent);
      font-size: 12px;
    }

    @media (max-width: 980px) {
      .layout {
        grid-template-columns: 1fr;
        height: auto;
      }

      .panel {
        min-height: 420px;
      }
    }

    @media (max-width: 680px) {
      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .toolbar {
        grid-template-columns: 1fr;
      }

      .layout {
        padding: 12px;
      }

      .exchange-row {
        grid-template-columns: 1fr 54px 64px;
      }

      .exchange-row .method,
      .exchange-row .duration {
        display: none;
      }

      .meta {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div>
      <h1>LLM Proxy Logs</h1>
      <p id="subtitle" class="subtitle">Loading communication logs...</p>
    </div>
    <button id="refresh" class="refresh" type="button">Refresh</button>
  </header>

  <section class="toolbar" aria-label="Filters">
    <input id="search" type="search" placeholder="Search path, model, target, or status">
    <select id="statusFilter" aria-label="Status filter">
      <option value="">All statuses</option>
      <option value="2">2xx success</option>
      <option value="4">4xx client errors</option>
      <option value="5">5xx server errors</option>
    </select>
  </section>

  <main class="layout">
    <section class="panel" aria-label="Exchange list">
      <div id="rows" class="rows"></div>
    </section>
    <section class="panel" aria-label="Exchange detail">
      <div id="empty" class="empty">Select a request to inspect the request and response.</div>
      <div id="detail" class="detail" hidden></div>
    </section>
  </main>

  <script>
    const state = { items: [], selectedId: null };
    const rows = document.getElementById("rows");
    const detail = document.getElementById("detail");
    const empty = document.getElementById("empty");
    const subtitle = document.getElementById("subtitle");
    const search = document.getElementById("search");
    const statusFilter = document.getElementById("statusFilter");
    const refresh = document.getElementById("refresh");

    function node(tag, className, text) {
      const element = document.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined) element.textContent = text;
      return element;
    }

    function formatTime(value) {
      if (!value) return "";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      return date.toLocaleString();
    }

    function statusClass(status) {
      const value = Number(status);
      if (value >= 200 && value < 300) return "ok";
      if (value >= 400 && value < 500) return "warn";
      if (value >= 500) return "bad";
      return "neutral";
    }

    function statusMatches(item) {
      const filter = statusFilter.value;
      if (!filter) return true;
      return String(item.status || "").startsWith(filter);
    }

    function textMatches(item) {
      const query = search.value.trim().toLowerCase();
      if (!query) return true;
      const haystack = [
        item.path,
        item.target_url,
        item.model,
        item.status,
        item.method
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    }

    function filteredItems() {
      return state.items.filter((item) => statusMatches(item) && textMatches(item));
    }

    function renderRows() {
      rows.replaceChildren();
      const items = filteredItems();
      subtitle.textContent = `${items.length} shown, ${state.items.length} loaded`;
      if (!items.length) {
        rows.append(node("div", "empty", "No matching communication logs."));
        return;
      }
      for (const item of items) {
        const row = node(
          "button",
          `exchange-row${item.id === state.selectedId ? " selected" : ""}`
        );
        row.type = "button";
        row.title = item.path || item.id;
        row.addEventListener("click", () => loadDetail(item.id));

        row.append(node("span", "time", formatTime(item.timestamp)));
        row.append(node("span", "method", item.method || ""));
        row.append(node("span", "path", item.path || ""));
        row.append(node("span", `status ${statusClass(item.status)}`, item.status || ""));
        row.append(node("span", "duration", `${item.duration_ms || 0} ms`));
        rows.append(row);
      }
    }

    function parseBody(value) {
      if (!value) return null;
      try {
        return JSON.parse(value);
      } catch {
        return null;
      }
    }

    function jsonValueNode(value) {
      if (value === null) return node("span", "json-null", "null");
      if (typeof value === "string") return node("span", "json-string", `"${value}"`);
      if (typeof value === "number") return node("span", "json-number", String(value));
      if (typeof value === "boolean") return node("span", "json-boolean", String(value));
      return node("span", "", String(value));
    }

    function containerLabel(value) {
      if (Array.isArray(value)) {
        return value.length ? `[${value.length}]` : "[]";
      }
      const size = Object.keys(value).length;
      return size ? `{${size}}` : "{}";
    }

    function createJsonTree(value, label, depth = 0) {
      if (value === null || typeof value !== "object") {
        const line = node("div", "json-leaf");
        if (label !== null && label !== undefined) {
          line.append(node("span", "json-key", `${label}: `));
        }
        line.append(jsonValueNode(value));
        return line;
      }

      const details = node("details", depth === 0 ? "root" : "");
      if (depth < 2) details.open = true;

      const summary = document.createElement("summary");
      if (label !== null && label !== undefined) {
        summary.append(node("span", "json-key", `${label}:`));
      }
      summary.append(node("span", "json-bracket", Array.isArray(value) ? "[" : "{"));
      summary.append(node("span", "json-size", containerLabel(value)));
      summary.append(node("span", "json-bracket", Array.isArray(value) ? "]" : "}"));
      details.append(summary);

      if (Array.isArray(value)) {
        value.forEach((item, index) => {
          details.append(createJsonTree(item, index, depth + 1));
        });
      } else {
        Object.entries(value).forEach(([key, child]) => {
          details.append(createJsonTree(child, key, depth + 1));
        });
      }
      return details;
    }

    function setAllJsonNodesOpen(container, open) {
      container.querySelectorAll("details").forEach((details) => {
        details.open = open;
      });
    }

    function addMeta(parent, label, value) {
      const item = node("div", "kv");
      item.append(node("b", "", label));
      item.append(node("span", "", value === null || value === undefined ? "" : String(value)));
      parent.append(item);
    }

    function addHeaders(parent, title, headers) {
      const section = node("section", "section");
      section.append(node("h3", "", title));
      const list = node("div", "headers");
      for (const header of headers || []) {
        const line = node("div", "header-line");
        line.append(node("span", "muted", header.name || ""));
        line.append(node("span", "", header.value || ""));
        list.append(line);
      }
      if (!list.children.length) list.append(node("div", "muted", "No headers."));
      section.append(list);
      parent.append(section);
    }

    function appendJsonTree(section, parsed) {
      const toolbar = node("div", "body-toolbar");
      const expandButton = node("button", "", "Expand all");
      expandButton.type = "button";
      const collapseButton = node("button", "", "Collapse all");
      collapseButton.type = "button";
      toolbar.append(expandButton);
      toolbar.append(collapseButton);
      section.append(toolbar);

      const tree = node("div", "json-tree");
      tree.append(createJsonTree(parsed, null));
      expandButton.addEventListener("click", () => setAllJsonNodesOpen(tree, true));
      collapseButton.addEventListener("click", () => setAllJsonNodesOpen(tree, false));
      section.append(tree);
    }

    function parseSse(value) {
      if (!value || typeof value !== "string") return null;
      if (!/(^|\n)\s*data:/.test(value)) return null;
      const events = [];
      let parsedAny = false;
      for (const line of value.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          events.push(JSON.parse(payload));
          parsedAny = true;
        } catch {
          events.push({ _raw: payload });
        }
      }
      return parsedAny ? events : null;
    }

    function reconstructStream(events) {
      let content = "";
      let reasoning = "";
      const toolCalls = [];
      let usage = null;
      let finishReason = null;
      for (const event of events) {
        if (event && event.usage) usage = event.usage;
        const choices = event && event.choices;
        if (!Array.isArray(choices)) continue;
        for (const choice of choices) {
          const delta = (choice && (choice.delta || choice.message)) || {};
          if (typeof delta.content === "string") content += delta.content;
          if (typeof delta.reasoning_content === "string") reasoning += delta.reasoning_content;
          if (choice.finish_reason) finishReason = choice.finish_reason;
          const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
          for (const call of calls) {
            const index = typeof call.index === "number" ? call.index : toolCalls.length;
            if (!toolCalls[index]) {
              toolCalls[index] = { id: call.id || "", name: "", arguments: "" };
            }
            const fn = call.function || {};
            if (call.id) toolCalls[index].id = call.id;
            if (fn.name) toolCalls[index].name += fn.name;
            if (typeof fn.arguments === "string") toolCalls[index].arguments += fn.arguments;
          }
        }
      }
      return { content, reasoning, toolCalls: toolCalls.filter(Boolean), usage, finishReason };
    }

    function appendSseView(section, events, raw) {
      const summary = reconstructStream(events);
      let renderedAnything = false;

      if (summary.reasoning) {
        section.append(node("div", "stream-label", "Reasoning"));
        section.append(node("pre", "", summary.reasoning));
        renderedAnything = true;
      }
      if (summary.content) {
        section.append(node("div", "stream-label", "Message"));
        section.append(node("pre", "", summary.content));
        renderedAnything = true;
      }
      if (summary.toolCalls.length) {
        section.append(node("div", "stream-label", "Tool calls"));
        const tree = node("div", "json-tree");
        tree.append(createJsonTree(summary.toolCalls, null));
        section.append(tree);
        renderedAnything = true;
      }
      if (!renderedAnything) {
        section.append(node("pre", "", raw ? String(raw) : ""));
      }

      const bits = [`${events.length} stream chunks`];
      if (summary.finishReason) bits.push(`finish: ${summary.finishReason}`);
      if (summary.usage) {
        const u = summary.usage;
        const prompt = u.prompt_tokens != null ? u.prompt_tokens : "?";
        const completion = u.completion_tokens != null ? u.completion_tokens : "?";
        const total = u.total_tokens != null ? u.total_tokens : "?";
        bits.push(`tokens: ${prompt} + ${completion} = ${total}`);
      }
      section.append(node("div", "stream-meta", bits.join("  ·  ")));

      const details = node("details", "stream-raw");
      const disclosure = document.createElement("summary");
      disclosure.append(node("span", "", `Raw stream chunks (${events.length})`));
      details.append(disclosure);
      const rawTree = node("div", "json-tree");
      rawTree.append(createJsonTree(events, null));
      details.append(rawTree);
      section.append(details);
    }

    function addBody(parent, title, value) {
      const section = node("section", "section");
      section.append(node("h3", "", title));

      const parsed = parseBody(value);
      if (parsed !== null) {
        appendJsonTree(section, parsed);
        parent.append(section);
        return;
      }

      const events = parseSse(value);
      if (events) {
        appendSseView(section, events, value);
        parent.append(section);
        return;
      }

      section.append(node("pre", "", value ? String(value) : ""));
      parent.append(section);
    }

    function renderDetail(exchange) {
      empty.hidden = true;
      detail.hidden = false;
      detail.replaceChildren();
      detail.append(node("h2", "", `${exchange.method || ""} ${exchange.path || ""}`.trim()));

      const meta = node("div", "meta");
      addMeta(meta, "Timestamp", formatTime(exchange.timestamp));
      addMeta(meta, "Status", `${exchange.status || ""} ${exchange.reason || ""}`.trim());
      addMeta(meta, "Duration", `${exchange.duration_ms || 0} ms`);
      addMeta(meta, "Model", exchange.model || "");
      addMeta(meta, "Target URL", exchange.target_url || "");
      addMeta(meta, "Bytes", `${exchange.request_bytes || 0} request / ${exchange.response_bytes || 0} response`);
      detail.append(meta);

      addHeaders(detail, "Request Headers", exchange.request_headers);
      addBody(detail, "Request Body", exchange.request_body);
      addHeaders(detail, "Response Headers", exchange.response_headers);
      addBody(detail, "Response Body", exchange.response_body);
    }

    async function loadList() {
      rows.replaceChildren(node("div", "loading", "Loading communication logs..."));
      try {
        const response = await fetch("/_api/exchanges");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        state.items = payload.exchanges || [];
        renderRows();
      } catch (error) {
        rows.replaceChildren(node("div", "error", `Failed to load logs: ${error.message}`));
      }
    }

    async function loadDetail(id) {
      state.selectedId = id;
      renderRows();
      detail.hidden = false;
      empty.hidden = true;
      detail.replaceChildren(node("div", "loading", "Loading detail..."));
      try {
        const response = await fetch(`/_api/exchanges/${encodeURIComponent(id)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderDetail(await response.json());
      } catch (error) {
        detail.replaceChildren(node("div", "error", `Failed to load detail: ${error.message}`));
      }
    }

    refresh.addEventListener("click", loadList);
    search.addEventListener("input", renderRows);
    statusFilter.addEventListener("change", renderRows);
    loadList();
  </script>
</body>
</html>
"""


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure_app_logging(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "app.log"

    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return log_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    file_config = load_config_file(Path(args.config))

    listen = resolve_setting(args.listen, file_config, "listen", DEFAULT_LISTEN)
    target_url = resolve_setting(
        args.target_url, file_config, "target_url", DEFAULT_TARGET_URL
    )
    app_logs_dir = resolve_setting(
        args.app_logs_dir, file_config, "app_logs_dir", DEFAULT_APP_LOGS_DIR
    )
    communication_logs_dir = resolve_setting(
        args.communication_logs_dir,
        file_config,
        "communication_logs_dir",
        DEFAULT_COMMUNICATION_LOGS_DIR,
    )
    timeout = float(
        resolve_setting(args.timeout, file_config, "timeout", DEFAULT_TIMEOUT)
    )

    host, port = parse_listen(str(listen))
    app_log_path = configure_app_logging(Path(app_logs_dir))
    config = ProxyConfig(
        target_url=str(target_url),
        communication_logs_dir=Path(communication_logs_dir),
        timeout=timeout,
    )
    logger = get_logger()

    server = ThreadingHTTPServer((host, port), make_handler(config))
    logger.info("Listening on http://%s:%d", host, port)
    logger.info("Forwarding proxied requests to %s", config.target_url)
    logger.info("Writing application logs to %s", app_log_path)
    logger.info("Writing communication logs to %s", config.communication_logs_dir)
    print(f"Listening on http://{host}:{port}", flush=True)
    print(f"Forwarding proxied requests to {config.target_url}", flush=True)
    print(f"Writing application logs to {app_log_path}", flush=True)
    print(f"Writing communication logs to {config.communication_logs_dir}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping proxy after keyboard interrupt")
        print("\nStopping proxy.", flush=True)
    finally:
        server.server_close()
        logger.info("Proxy stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
