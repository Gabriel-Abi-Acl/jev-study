"""Contrato: o JSON publicado entra no schema; o JSON quebrado não entra."""

import pytest
from pydantic import ValidationError

from jev_study.errors import SchemaError
from jev_study.schema import (
    ChoiceQuestion,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    dump_request,
    parse_response,
)
from tests.payloads import (
    BOX_RESPONSE,
    CHARGED_TWICE,
    CHOICE_RESPONSE,
    NOUL_RESPONSE,
    SCORE_RESPONSE,
)


def test_published_responses_parse():
    noul = parse_response(NOUL_RESPONSE)
    choice = parse_response(CHOICE_RESPONSE)
    score = parse_response(SCORE_RESPONSE)
    box = parse_response(BOX_RESPONSE)
    assert noul.answers["billing"].noul == 0.98
    assert choice.answers["tone"].choice == "calm"
    assert choice.answers["tone"].confidence == 0.7
    assert score.answers["urgency"].score == 1.7
    assert box.answers["red_rectangle"].confidence is None
    assert box.answers["red_rectangle"].box.left == 0.19


def test_choice_request_matches_published_shape():
    request = SystemOneRequest(
        model="jev-latest",
        state=CHARGED_TWICE,
        questions={
            "tone": ChoiceQuestion(
                instructions="What is the tone of this message?",
                criteria={
                    "angry": "An upset or hostile message",
                    "calm": "A neutral or polite message",
                    "excited": "An enthusiastic or eager message",
                },
            )
        },
    )
    assert dump_request(request) == {
        "model": "jev-latest",
        "state": CHARGED_TWICE,
        "questions": {
            "tone": {
                "type": "choice",
                "instructions": "What is the tone of this message?",
                "criteria": {
                    "angry": "An upset or hostile message",
                    "calm": "A neutral or polite message",
                    "excited": "An enthusiastic or eager message",
                },
            }
        },
    }


def test_score_request_criteria_is_a_list_and_noul_criteria_use_boolean_keys():
    score = SystemOneRequest(
        model="jev-latest",
        state=CHARGED_TWICE,
        questions={
            "urgency": ScoreQuestion(
                instructions="How urgent is this message?",
                criteria=[
                    "Can wait",
                    "Needs attention this week",
                    "Needs attention today",
                ],
            )
        },
    )
    noul = SystemOneRequest(
        model="jev-latest",
        state=CHARGED_TWICE,
        questions={
            "billing": NoulQuestion(
                instructions="Is this message about billing?",
                criteria=NoulCriteria(
                    true_case="Unsolicited advertising.",
                    false_case="A legitimate conversation.",
                ),
            )
        },
    )
    assert dump_request(score)["questions"]["urgency"]["criteria"] == [
        "Can wait",
        "Needs attention this week",
        "Needs attention today",
    ]
    assert dump_request(noul)["questions"]["billing"]["criteria"] == {
        "true": "Unsolicited advertising.",
        "false": "A legitimate conversation.",
    }
    bare = SystemOneRequest(
        model="jev-latest",
        state=CHARGED_TWICE,
        questions={"billing": NoulQuestion(instructions="Is this message about billing?")},
    )
    assert "criteria" not in dump_request(bare)["questions"]["billing"]


def test_request_roundtrip_keeps_aliases():
    request = SystemOneRequest(
        model="jev-1.14.0",
        state={"subject": "Cobrança", "body": "Quero o estorno."},
        questions={"billing": NoulQuestion(instructions="Pede estorno?")},
    )
    assert SystemOneRequest.model_validate(dump_request(request)) == request


def test_invalid_answers_fail_schema():
    with pytest.raises(SchemaError):
        parse_response({"model": "jev-latest", "answers": {"tone": {"type": "choice"}}})
    with pytest.raises(SchemaError):
        parse_response(
            {
                "model": "jev-latest",
                "answers": {
                    "tone": {
                        "type": "choice",
                        "choice": "calm",
                        "confidence": 0.7,
                        "probabilities": {"angry": 1.2},
                    }
                },
            }
        )
    with pytest.raises(SchemaError):
        parse_response(
            {
                "model": "jev-latest",
                "answers": {
                    "tone": {
                        "type": "choice",
                        "choice": "missing",
                        "confidence": 0.7,
                        "probabilities": {"calm": 1.0},
                    }
                },
            }
        )
    with pytest.raises(SchemaError):
        parse_response(
            {
                "model": "jev-latest",
                "answers": {
                    "urgency": {
                        "type": "score",
                        "score": 9,
                        "confidence": 0.5,
                        "legend": {"0": "a", "1": "b"},
                        "probabilities": {"0": 0.5, "1": 0.5},
                    }
                },
            }
        )
    with pytest.raises(SchemaError):
        parse_response(
            {
                "model": "jev-preview",
                "answers": {
                    "damage": {
                        "type": "box",
                        "box": {"top": 0.8, "bottom": 0.2, "left": 0.1, "right": 0.4},
                        "confidence": None,
                    }
                },
            }
        )
    with pytest.raises(SchemaError):
        parse_response(
            {
                "model": "jev-latest",
                "answers": {"billing": {"type": "noul", "noul": 1.4}},
            }
        )


def test_box_accepts_null_confidence_and_rejects_a_missing_one():
    parsed = parse_response(BOX_RESPONSE)
    assert parsed.answers["red_rectangle"].confidence is None
    payload = {
        "model": "jev-preview",
        "answers": {
            "red_rectangle": {
                "type": "box",
                "box": {"top": 0.2, "bottom": 0.66, "left": 0.19, "right": 0.68},
            }
        },
    }
    with pytest.raises(SchemaError):
        parse_response(payload)


def test_empty_questions_are_rejected():
    with pytest.raises(ValidationError):
        SystemOneRequest(model="jev-1.14.0", state="oi", questions={})
