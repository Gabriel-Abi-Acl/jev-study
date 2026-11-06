"""Backends do estudo. Importar este pacote não carrega SDK de fornecedor."""

from jev_study.backends.fake import FakeBackend, echo_labels
from jev_study.backends.jev import JevHttpBackend
from jev_study.backends.llm import LlmStructuredBackend

__all__ = [
    "FakeBackend",
    "JevHttpBackend",
    "LlmStructuredBackend",
    "echo_labels",
]
