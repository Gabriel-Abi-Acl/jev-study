"""Interface comum. Jev e o baseline devolvem SystemOneResponse."""

from typing import Protocol

from jev_study.schema import SystemOneRequest, SystemOneResponse


class DecisionBackend(Protocol):
    """Um backend nomeado. model é o id pedido, não necessariamente o id que respondeu."""

    name: str
    model: str

    def decide(self, request: SystemOneRequest) -> SystemOneResponse:
        """Avalia state e questions. Não devolve gabarito."""
