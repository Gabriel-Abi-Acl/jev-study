"""Contrato do cliente HTTP: o POST bate com a API e a chave não vaza nem segue redirect."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jev_study.backends.jev import JevHttpBackend
from jev_study.errors import BackendError, ConfigError, SchemaError
from jev_study.schema import BoxQuestion, NoulQuestion, SystemOneRequest
from tests.payloads import NOUL_RESPONSE

PROXY_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)


def _request(question=None) -> SystemOneRequest:
    return SystemOneRequest(
        model="jev-1.14.0",
        state="Quero o estorno.",
        questions={"billing": question or NoulQuestion(instructions="Pede estorno?")},
    )


def test_post_shape_header_and_redaction():
    captured = {}

    def transport(url, headers, body, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["timeout"] = timeout
        return 200, json.dumps(NOUL_RESPONSE).encode()

    client = JevHttpBackend(
        api_key="super-secret",
        model="jev-1.14.0",
        base_url="https://api.typesafe.ai/",
        transport=transport,
        timeout_s=12,
    )
    response = client.decide(_request())
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer super-secret"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert "TypeSafe-Danger-Zone-Images" not in captured["headers"]
    assert captured["timeout"] == 12
    decoded = captured["body"].decode()
    assert "Quero o estorno." in decoded
    assert "super-secret" not in decoded
    assert json.loads(decoded)["model"] == "jev-1.14.0"
    assert response.answers["billing"].noul == 0.98


def test_image_header_for_box_or_assets_only():
    seen = []

    def transport(url, headers, body, timeout):
        seen.append(headers.get("TypeSafe-Danger-Zone-Images"))
        return 200, json.dumps(NOUL_RESPONSE).encode()

    client = JevHttpBackend(api_key="k", model="jev-1.14.0", transport=transport)
    client.decide(
        SystemOneRequest(
            model="jev-1.14.0",
            state="foto",
            questions={"damage": BoxQuestion(instructions="Onde está o dano?")},
        )
    )
    client.decide(
        _request().model_copy(update={"assets": {"foto": {"type": "image", "format": "png"}}})
    )
    assert seen == ["true", "true"]


def test_http_error_and_transport_failure_redact_the_key():
    def forbidden(url, headers, body, timeout):
        return 422, b'{"detail":"bad super-secret"}'

    client = JevHttpBackend(api_key="super-secret", model="jev-1.14.0", transport=forbidden)
    with pytest.raises(BackendError) as exc:
        client.decide(_request())
    assert exc.value.status == 422
    assert "HTTP 422" in str(exc.value)
    assert "super-secret" not in str(exc.value)

    def explode(url, headers, body, timeout):
        raise RuntimeError(f"boom {headers['Authorization']}")

    client = JevHttpBackend(api_key="super-secret", model="jev-1.14.0", transport=explode)
    with pytest.raises(BackendError) as exc:
        client.decide(_request())
    assert "super-secret" not in str(exc.value)
    assert "[redacted]" in str(exc.value)
    assert exc.value.__cause__ is None


def test_success_without_usage_is_a_schema_error():
    def transport(url, headers, body, timeout):
        payload = {"model": "jev-1.14.0", "answers": {"billing": {"type": "noul", "noul": 0.2}}}
        return 200, json.dumps(payload).encode()

    client = JevHttpBackend(api_key="k", model="jev-1.14.0", transport=transport)
    with pytest.raises(SchemaError):
        client.decide(_request())


def test_rejects_bad_key_and_credential_in_url():
    with pytest.raises(ConfigError) as exc:
        JevHttpBackend(api_key="bad key", model="jev-1.14.0")
    assert "bad" not in str(exc.value)
    with pytest.raises(ConfigError) as exc:
        JevHttpBackend(
            api_key="k",
            model="jev-1.14.0",
            base_url="https://user:super-secret@api.typesafe.ai",
        )
    assert "super-secret" not in str(exc.value)


@contextmanager
def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_urllib_transport_posts_json(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            seen.append(
                {
                    "path": self.path,
                    "body": raw,
                    "auth": self.headers.get("Authorization"),
                }
            )
            payload = json.dumps(NOUL_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            return

    with _serve(Handler) as server:
        port = server.server_address[1]
        client = JevHttpBackend(
            api_key="local-test-key",
            model="jev-1.14.0",
            base_url=f"http://127.0.0.1:{port}",
        )
        response = client.decide(_request())
    assert response.usage is not None
    assert response.usage.input_tokens == 120
    assert seen[0]["path"] == "/v1/systemone"
    assert seen[0]["auth"] == "Bearer local-test-key"
    assert json.loads(seen[0]["body"])["state"] == "Quero o estorno."


def test_urllib_transport_does_not_follow_redirects(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            hits.append(("POST", self.path))
            host, port = self.server.server_address[:2]
            self.send_response(302)
            self.send_header("Location", f"http://{host}:{port}/stolen")
            self.end_headers()

        def do_GET(self):
            hits.append(("GET", self.path))
            self.send_response(200)
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    with _serve(Handler) as server:
        port = server.server_address[1]
        client = JevHttpBackend(
            api_key="local-test-key",
            model="jev-1.14.0",
            base_url=f"http://127.0.0.1:{port}",
            timeout_s=2,
        )
        with pytest.raises(BackendError) as exc:
            client.decide(_request())
    assert exc.value.status == 302
    assert hits == [("POST", "/v1/systemone")]
    assert "local-test-key" not in str(exc.value)
