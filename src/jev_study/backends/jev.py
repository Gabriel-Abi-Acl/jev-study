"""Cliente HTTP do System One: POST /v1/systemone.

O contrato público é HTTP, então este módulo não importa SDK da TypeSafe.
A chave só entra em Authorization e é apagada das mensagens de erro.
Redirect é recusado: seguir 3xx reenviaria a chave para outro host.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable

from jev_study.config import validate_http_base_url
from jev_study.errors import BackendError, ConfigError, SchemaError, redact
from jev_study.schema import SystemOneRequest, SystemOneResponse, dump_request, parse_response

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]

DEFAULT_TIMEOUT_S = 60.0


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, _newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


class JevHttpBackend:
    """POST JSON para a origem configurada. Testes injetam transport e não abrem rede."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.typesafe.ai",
        transport: Transport | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        name: str = "jev",
    ) -> None:
        self.name = name
        self.model = model
        self._api_key = _require_api_key(api_key)
        self._base_url = validate_http_base_url(base_url)
        self._transport = transport or urllib_transport
        self._timeout_s = timeout_s
        if not model.strip():
            raise ConfigError("modelo Jev vazio")

    def decide(self, request: SystemOneRequest) -> SystemOneResponse:
        url = f"{self._base_url}/v1/systemone"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if _needs_image_header(request):
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        body = json.dumps(dump_request(request), ensure_ascii=False).encode("utf-8")
        try:
            status, raw = self._transport(url, headers, body, self._timeout_s)
        except Exception as exc:
            # from None: o traceback da causa pode repetir o header Authorization.
            raise BackendError(redact(str(exc), self._api_key)) from None
        if status < 200 or status >= 300:
            detail = redact(raw.decode("utf-8", errors="replace")[:500], self._api_key)
            raise BackendError(f"HTTP {status}: {detail}", status=status)
        response = parse_response(raw)
        if response.usage is None:
            raise SchemaError("usage ausente na resposta do System One")
        return response


def urllib_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float,
) -> tuple[int, bytes]:
    """POST pela stdlib. HTTP 4xx/5xx volta como status, não como exceção."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(_RefuseRedirects)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _require_api_key(api_key: str) -> str:
    if not api_key or any(character.isspace() for character in api_key):
        raise ConfigError("chave API ausente ou inválida")
    return api_key


def _needs_image_header(request: SystemOneRequest) -> bool:
    # assets presente, mesmo vazio, já pede o header de visão da API.
    if request.assets is not None:
        return True
    return any(question.type == "box" for question in request.questions.values())
