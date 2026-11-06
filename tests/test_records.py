"""Contrato: confidence da API e probabilidade da decisão ficam em campos separados.

O exemplo publicado de Choice tem confidence 0,7 e P(calm) 0,8. Não podem ser fundidos.
"""

from jev_study.records import declared_confidence, declared_probability, nearest_score_level
from jev_study.schema import ScoreAnswer, parse_response
from tests.payloads import BOX_RESPONSE, CHOICE_RESPONSE, NOUL_RESPONSE, SCORE_RESPONSE


def test_published_examples_keep_confidence_and_probability_distinct():
    choice = parse_response(CHOICE_RESPONSE).answers["tone"]
    score = parse_response(SCORE_RESPONSE).answers["urgency"]
    noul = parse_response(NOUL_RESPONSE).answers["billing"]
    box = parse_response(BOX_RESPONSE).answers["red_rectangle"]
    assert declared_confidence(choice) == 0.7
    assert declared_probability(choice) == 0.8
    assert declared_confidence(score) == 0.55
    assert declared_probability(score) == 0.8
    assert declared_confidence(noul) is None
    assert declared_probability(noul) == 0.98
    assert declared_confidence(box) is None
    assert declared_probability(box) is None


def test_score_probability_uses_half_up_nearest_level():
    answer = ScoreAnswer(
        type="score",
        score=1.5,
        confidence=0.4,
        legend={"0": "a", "1": "b", "2": "c"},
        probabilities={"0": 0.25, "1": 0.25, "2": 0.5},
    )
    assert nearest_score_level(0.4) == 0
    assert nearest_score_level(0.5) == 1
    assert nearest_score_level(1.5) == 2
    assert nearest_score_level(1.7) == 2
    assert declared_probability(answer) == 0.5
