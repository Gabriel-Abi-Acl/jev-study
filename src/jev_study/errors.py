"""Erros de configuração, HTTP e schema. A mensagem pública não carrega segredo."""


class ConfigError(Exception):
    """Env ou URL inválidos. A mensagem não repete o valor rejeitado."""


class BackendError(Exception):
    """Falha antes de existir uma decisão tipada (rede, HTTP, chave, SDK)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class SchemaError(Exception):
    """Corpo que não cabe no schema tipado compartilhado pelos backends."""


def redact(text: str, secret: str | None) -> str:
    """Tira a chave de uma mensagem de erro antes de ela ir para o registro."""
    if not secret:
        return text
    return text.replace(secret, "[redacted]")
