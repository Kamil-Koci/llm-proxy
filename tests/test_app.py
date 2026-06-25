from __future__ import annotations

import json
import logging
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app import (
    DEFAULT_TARGET_URL,
    ProxyConfig,
    configure_app_logging,
    load_config_file,
    make_handler,
    parse_listen,
    resolve_setting,
    target_url_for_request,
)


class MockUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    captured_body = b""
    captured_content_type = ""
    captured_path = ""
    captured_method = ""

    def do_GET(self) -> None:
        type(self).captured_method = "GET"
        type(self).captured_path = self.path
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": "/models/qwen3-coder-next-fp8",
                        "object": "model",
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Set-Cookie", "upstream-secret=1")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        type(self).captured_method = "POST"
        length = int(self.headers.get("Content-Length", "0"))
        type(self).captured_body = self.rfile.read(length)
        type(self).captured_content_type = self.headers.get("Content-Type", "")
        type(self).captured_path = self.path
        body = json.dumps({"ok": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Set-Cookie", "upstream-secret=1")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


def start_server(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


class ProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        MockUpstreamHandler.captured_body = b""
        MockUpstreamHandler.captured_content_type = ""
        MockUpstreamHandler.captured_path = ""
        MockUpstreamHandler.captured_method = ""

    def test_parse_listen_accepts_host_port_and_port_only(self) -> None:
        self.assertEqual(parse_listen("0.0.0.0:9090"), ("0.0.0.0", 9090))
        self.assertEqual(parse_listen("9091"), ("localhost", 9091))

    def test_target_url_for_request_uses_base_url_without_duplicate_path(self) -> None:
        self.assertEqual(
            target_url_for_request("http://localhost:3000/v1", "/chat/completions"),
            "http://localhost:3000/v1/chat/completions",
        )
        self.assertEqual(
            target_url_for_request("http://localhost:3000/v1", "/v1/chat/completions"),
            "http://localhost:3000/v1/chat/completions",
        )

    def test_load_config_file_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(load_config_file(Path(tmpdir) / "nope.toml"), {})

    def test_load_config_file_parses_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                'target_url = "http://upstream:9000/v1"\ntimeout = 30.0\n',
                encoding="utf-8",
            )
            config = load_config_file(config_path)
            self.assertEqual(config["target_url"], "http://upstream:9000/v1")
            self.assertEqual(config["timeout"], 30.0)

    def test_load_config_file_invalid_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text("this is = = not valid toml", encoding="utf-8")
            self.assertEqual(load_config_file(config_path), {})

    def test_resolve_setting_precedence(self) -> None:
        config = {"target_url": "http://from-config/v1"}
        # CLI value wins over config and default.
        self.assertEqual(
            resolve_setting("http://from-cli/v1", config, "target_url", DEFAULT_TARGET_URL),
            "http://from-cli/v1",
        )
        # Config wins over default when no CLI value is given.
        self.assertEqual(
            resolve_setting(None, config, "target_url", DEFAULT_TARGET_URL),
            "http://from-config/v1",
        )
        # Falls back to the built-in default when neither is set.
        self.assertEqual(
            resolve_setting(None, {}, "target_url", DEFAULT_TARGET_URL),
            DEFAULT_TARGET_URL,
        )

    def test_configure_app_logging_writes_to_app_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = configure_app_logging(Path(tmpdir))
            logging.getLogger("llm_proxy").info("runtime log test")
            for handler in logging.getLogger("llm_proxy").handlers:
                handler.flush()

            self.assertEqual(log_path, Path(tmpdir) / "app.log")
            self.assertIn("runtime log test", log_path.read_text(encoding="utf-8"))

    def test_ui_route_is_served_without_forwarding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proxy = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    ProxyConfig(
                        target_url="http://127.0.0.1:1/v1",
                        communication_logs_dir=Path(tmpdir),
                        timeout=1,
                    )
                ),
            )
            start_server(proxy)

            with urllib.request.urlopen(
                f"http://127.0.0.1:{proxy.server_port}/_ui", timeout=5
            ) as response:
                self.assertEqual(response.status, 200)
                html = response.read().decode("utf-8")

            self.assertIn("LLM Proxy Logs", html)
            self.assertEqual(MockUpstreamHandler.captured_method, "")
            self.assertEqual(list(Path(tmpdir).glob("*")), [])

            proxy.shutdown()
            proxy.server_close()

    def test_get_models_is_forwarded_and_logged(self) -> None:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstreamHandler)
        start_server(upstream)
        upstream_url = f"http://127.0.0.1:{upstream.server_port}/v1"

        with tempfile.TemporaryDirectory() as tmpdir:
            proxy = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    ProxyConfig(
                        target_url=upstream_url,
                        communication_logs_dir=Path(tmpdir),
                        timeout=5,
                    )
                ),
            )
            start_server(proxy)

            with urllib.request.urlopen(
                f"http://127.0.0.1:{proxy.server_port}/v1/models", timeout=5
            ) as response:
                self.assertEqual(response.status, 200)
                body = json.loads(response.read())

            self.assertEqual(MockUpstreamHandler.captured_method, "GET")
            self.assertEqual(MockUpstreamHandler.captured_path, "/v1/models")
            self.assertEqual(
                body["data"][0]["id"],
                "/models/qwen3-coder-next-fp8",
            )

            log_files = list(Path(tmpdir).glob("*.log"))
            self.assertEqual(len(log_files), 1)
            log_text = log_files[0].read_text(encoding="utf-8")
            self.assertIn("GET /v1/models", log_text)
            self.assertIn('"object": "model"', log_text)

            json_files = list(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(json_files), 1)
            exchange = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(exchange["method"], "GET")
            self.assertEqual(exchange["path"], "/v1/models")
            self.assertEqual(exchange["status"], 200)
            self.assertIn("duration_ms", exchange)

            proxy.shutdown()
            proxy.server_close()

        upstream.shutdown()
        upstream.server_close()

    def test_post_is_forwarded_and_logged(self) -> None:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstreamHandler)
        start_server(upstream)
        upstream_url = f"http://127.0.0.1:{upstream.server_port}/v1"

        with tempfile.TemporaryDirectory() as tmpdir:
            proxy = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    ProxyConfig(
                        target_url=upstream_url,
                        communication_logs_dir=Path(tmpdir),
                        timeout=5,
                    )
                ),
            )
            start_server(proxy)

            payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
            request = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer secret-token",
                    "Cookie": "session=secret",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read()), {"ok": True})

            self.assertEqual(json.loads(MockUpstreamHandler.captured_body), payload)
            self.assertEqual(MockUpstreamHandler.captured_content_type, "application/json")
            self.assertEqual(MockUpstreamHandler.captured_path, "/v1/chat/completions")

            log_files = list(Path(tmpdir).glob("*.log"))
            self.assertEqual(len(log_files), 1)
            self.assertRegex(log_files[0].name, r"^\d{8}T\d{6}\.\d{6}Z_[0-9a-f]{8}\.log$")

            log_text = log_files[0].read_text(encoding="utf-8")
            self.assertIn("=== REQUEST ===", log_text)
            self.assertIn("=== RESPONSE ===", log_text)
            self.assertIn('"model": "test-model"', log_text)
            self.assertIn('"ok": true', log_text)
            self.assertIn("Authorization: [REDACTED]", log_text)
            self.assertIn("Cookie: [REDACTED]", log_text)
            self.assertIn("Set-Cookie: [REDACTED]", log_text)
            self.assertNotIn("Bearer secret-token", log_text)
            self.assertNotIn("session=secret", log_text)
            self.assertNotIn("upstream-secret=1", log_text)

            json_files = list(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(json_files), 1)
            exchange = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(exchange["model"], "test-model")
            self.assertEqual(exchange["request_body"], json.dumps(payload))
            self.assertEqual(exchange["response_body"], json.dumps({"ok": True}))
            request_header_values = {
                header["name"].lower(): header["value"]
                for header in exchange["request_headers"]
            }
            response_header_values = {
                header["name"].lower(): header["value"]
                for header in exchange["response_headers"]
            }
            self.assertEqual(request_header_values["authorization"], "[REDACTED]")
            self.assertEqual(request_header_values["cookie"], "[REDACTED]")
            self.assertEqual(response_header_values["set-cookie"], "[REDACTED]")

            with urllib.request.urlopen(
                f"http://127.0.0.1:{proxy.server_port}/_api/exchanges", timeout=5
            ) as response:
                index = json.loads(response.read())

            self.assertEqual(len(index["exchanges"]), 1)
            self.assertEqual(index["exchanges"][0]["id"], exchange["id"])
            self.assertEqual(index["exchanges"][0]["model"], "test-model")

            with urllib.request.urlopen(
                f"http://127.0.0.1:{proxy.server_port}/_api/exchanges/{exchange['id']}",
                timeout=5,
            ) as response:
                detail = json.loads(response.read())

            self.assertEqual(detail["id"], exchange["id"])
            self.assertEqual(detail["status"], 200)

            with urllib.request.urlopen(
                f"http://127.0.0.1:{proxy.server_port}/_api/exchanges/{exchange['id']}/log",
                timeout=5,
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers.get("Content-Type"), "text/plain; charset=utf-8"
                )
                log_body = response.read().decode("utf-8")
            self.assertIn("=== REQUEST ===", log_body)
            self.assertEqual(log_body, log_files[0].read_text(encoding="utf-8"))

            with urllib.request.urlopen(
                f"http://127.0.0.1:{proxy.server_port}/_api/exchanges/{exchange['id']}/json",
                timeout=5,
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers.get("Content-Type"),
                    "application/json; charset=utf-8",
                )
                raw_json = response.read().decode("utf-8")
            self.assertEqual(raw_json, json_files[0].read_text(encoding="utf-8"))

            missing = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_port}/_api/exchanges/{exchange['id']}/txt"
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(missing, timeout=5)
            self.assertEqual(ctx.exception.code, 404)

            proxy.shutdown()
            proxy.server_close()

        upstream.shutdown()
        upstream.server_close()


if __name__ == "__main__":
    unittest.main()
