"""Contrato do gabarito v1: schema, contagem e pedido compatível com os adapters.

Não mede acerto, calibração, latência nem custo.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from jev_study.backends.fake import echo_labels
from jev_study.backends.jev import JevHttpBackend
from jev_study.backends.llm import build_llm_messages
from jev_study.dataset import DatasetError, load_eval_decisions
from jev_study.dataset.eval_set import default_eval_path
from jev_study.runner import run_tasks
from jev_study.schema import SystemOneRequest, dump_request

_ROUTE = {
    "billing": "Cobrança, fatura, estorno ou disputa de pagamento.",
    "technical": "Defeito, indisponibilidade ou bug do produto.",
    "account": "Login, perfil ou dado cadastral, sem defeito do produto.",
}


def _choice_payload(task_id: str = "choice_ok", **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "id": task_id,
        "category": "choice",
        "state": "Quero o estorno da mensalidade que entrou duas vezes.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Qual equipe deve cuidar deste chamado?",
                "criteria": dict(_ROUTE),
            }
        },
        "expected": {"team": {"choice": "billing"}},
        "notes": "O pedido é estorno.",
        "expectation": "clear",
    }
    payload.update(overrides)
    return payload


def _score_payload(task_id: str = "score_ok") -> dict:
    return {
        "schema_version": 1,
        "id": task_id,
        "category": "score",
        "state": "O relatório precisa ficar pronto até sexta-feira desta semana.",
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": "Qual é a urgência deste pedido?",
                "criteria": [
                    "Pode esperar",
                    "Precisa de atenção nesta semana",
                    "Precisa de atenção hoje",
                ],
            }
        },
        "expected": {"urgency": {"score_level": 1}},
        "notes": "Prazo dentro da semana.",
        "expectation": "clear",
    }


def _noul_payload(task_id: str = "noul_ok") -> dict:
    return {
        "schema_version": 1,
        "id": task_id,
        "category": "noul",
        "state": "Peço o estorno da cobrança repetida.",
        "questions": {
            "refund_requested": {
                "type": "noul",
                "instructions": "Esta mensagem pede estorno ou reembolso?",
                "criteria": {
                    "true": "A pessoa pede devolução de valor.",
                    "false": "A pessoa não pede dinheiro de volta.",
                },
            }
        },
        "expected": {"refund_requested": {"noul": True}},
        "notes": "O verbo pede estorno.",
        "expectation": "clear",
    }


def _write(directory: Path, payloads: list[dict]) -> Path:
    path = directory / "decisions.jsonl"
    lines = [json.dumps(payload, ensure_ascii=False) for payload in payloads]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fixture(category: str) -> dict:
    if category == "choice":
        return _choice_payload()
    if category == "score":
        return _score_payload()
    return _noul_payload()


def test_default_file_is_the_versioned_jsonl():
    path = default_eval_path()
    assert path.parts[-3:] == ("data", "eval", "decisions.v1.jsonl")


def test_v1_balances_types_and_marks_low_confidence_as_design():
    rows = load_eval_decisions()
    assert 40 <= len(rows) <= 50
    counts = Counter(row.category for row in rows)
    assert set(counts) == {"choice", "score", "noul"}
    assert len(set(counts.values())) == 1
    low = Counter(row.category for row in rows if row.expectation == "low_confidence")
    assert set(low) == {"choice", "score", "noul"}
    assert all(1 <= low[category] <= 3 for category in counts)
    assert all(row.schema_version == 1 for row in rows)
    assert all(row.notes.strip() for row in rows)
    assert all(len(row.notes) >= 40 for row in rows if row.expectation == "low_confidence")
    assert all(task.source == "eval-v1" and not task.skip for task in _tasks(rows))

    noul_labels = [
        label.noul for row in rows if row.category == "noul" for label in row.expected.values()
    ]
    assert set(noul_labels) == {True, False}
    assert min(noul_labels.count(True), noul_labels.count(False)) >= 5
    levels = {
        label.score_level
        for row in rows
        if row.category == "score"
        for label in row.expected.values()
    }
    assert {0, 1, 2} <= levels
    choices = {
        label.choice for row in rows if row.category == "choice" for label in row.expected.values()
    }
    assert len(choices) >= 3


def test_v1_file_has_one_question_per_row_and_no_model_output():
    raw = default_eval_path().read_text(encoding="utf-8")
    assert not raw.startswith("\ufeff")
    assert raw.endswith("\n")
    folded = raw.casefold()
    for forbidden in ("outsmart", "base64", "data:image", "latency_ms", "estimated_cost"):
        assert forbidden not in folded
    assert "@" not in raw
    label_key = {"choice": "choice", "score": "score_level", "noul": "noul"}
    ids: list[str] = []
    for line in raw.splitlines():
        item = json.loads(line)
        assert set(item) == {
            "schema_version",
            "id",
            "category",
            "state",
            "questions",
            "expected",
            "notes",
            "expectation",
        }
        assert isinstance(item["state"], str) and item["state"].strip()
        assert len(item["questions"]) == 1
        question = next(iter(item["questions"].values()))
        assert question["type"] == item["category"]
        assert question["type"] != "box"
        assert list(next(iter(item["expected"].values()))) == [label_key[item["category"]]]
        ids.append(item["id"])
    assert len(ids) == len(set(ids))
    assert all(item_id.isascii() and item_id.replace("_", "").isalnum() for item_id in ids)


def test_known_labels_keep_the_preferred_reading():
    rows = {row.id: row for row in load_eval_decisions()}
    billing = rows["choice_route_duplicate_refund"]
    assert billing.expected["team"].choice == "billing"
    assert billing.expectation == "clear"
    ambiguous_login = rows["choice_login_after_reset_low"]
    assert ambiguous_login.expected["team"].choice == "account"
    assert ambiguous_login.expectation == "low_confidence"
    assert rows["choice_tone_thanks_but_unacceptable_low"].expected["tone"].choice == "angry"
    assert rows["score_urgency_payroll_today"].expected["urgency"].score_level == 2
    assert rows["score_urgency_no_rush"].expected["urgency"].score_level == 0
    week_or_next = rows["score_urgency_this_week_or_next_low"]
    assert week_or_next.expected["urgency"].score_level == 1
    assert week_or_next.expectation == "low_confidence"
    irritation = rows["score_frustration_irritation_passed_low"]
    assert irritation.expected["frustration"].score_level == 0
    assert rows["noul_refund_explicit_yes"].expected["refund_requested"].noul is True
    assert rows["noul_refund_export_how_to_no"].expected["refund_requested"].noul is False
    conditional = rows["noul_refund_conditional_low"]
    assert conditional.expected["refund_requested"].noul is True
    assert conditional.expectation == "low_confidence"
    inside_ticket = rows["noul_spam_offer_inside_ticket_low"]
    assert inside_ticket.expected["unsolicited_ad"].noul is True
    assert inside_ticket.expectation == "low_confidence"


@pytest.mark.parametrize("category", ["choice", "score", "noul"])
def test_loader_accepts_one_fixture_per_question_type(tmp_path, category):
    rows = load_eval_decisions(_write(tmp_path, [_fixture(category)]))
    assert len(rows) == 1
    assert rows[0].category == category
    question = next(iter(rows[0].questions.values()))
    assert question.type == category
    task = rows[0].as_decision_task()
    assert task.expected.keys() == rows[0].questions.keys()


def test_clear_row_may_omit_notes_and_two_questions_of_one_type(tmp_path):
    payload = _choice_payload()
    del payload["notes"]
    payload["questions"]["priority"] = {
        "type": "choice",
        "instructions": "Qual fila recebe o chamado?",
        "criteria": {"soon": "Ainda hoje.", "later": "Pode esperar."},
    }
    payload["expected"]["priority"] = {"choice": "soon"}
    row = load_eval_decisions(_write(tmp_path, [payload]))[0]
    assert row.notes == ""
    assert set(row.questions) == {"team", "priority"}
    assert row.as_decision_task().expected["priority"].choice == "soon"


def test_dict_state_round_trips_into_the_request(tmp_path):
    payload = _choice_payload()
    payload["state"] = {"channel": "email", "body": "Quero o estorno da mensalidade."}
    row = load_eval_decisions(_write(tmp_path, [payload]))[0]
    request = SystemOneRequest(model="jev-1.14.0", state=row.state, questions=row.questions)
    assert dump_request(request)["state"] == payload["state"]


def test_adapters_receive_questions_without_the_label():
    picked = []
    for category in ("choice", "score", "noul"):
        picked.append(
            next(
                row
                for row in load_eval_decisions()
                if row.category == category and row.expectation == "clear"
            )
        )
    for row in picked:
        request = SystemOneRequest(
            model="jev-1.14.0",
            state=row.state,
            questions=row.questions,
        )
        body = dump_request(request)
        assert set(body) == {"model", "state", "questions"}
        again = SystemOneRequest.model_validate(body)
        assert again.questions.keys() == row.questions.keys()
        assert again.state == row.state
        messages = build_llm_messages(request)
        user = json.loads(messages[1]["content"])
        assert user == body
        assert row.id not in messages[0]["content"]
        assert "expected" not in messages[1]["content"]
        _assert_jev_posts_the_same_body(request, body)


def test_label_echo_accepts_every_v1_task_without_usage():
    tasks = [row.as_decision_task() for row in load_eval_decisions()]
    records = run_tasks(echo_labels(tasks), tasks)
    assert len(records) == len(tasks)
    assert {record.question_type for record in records} == {"choice", "score", "noul"}
    assert all(record.error is None and record.skipped is False for record in records)
    assert all(record.estimated_cost_usd is None for record in records)
    assert all(record.input_tokens is None for record in records)
    by_id = {record.task_id: record for record in records}
    assert by_id["choice_route_duplicate_refund"].decision["choice"] == "billing"
    assert by_id["score_urgency_by_friday"].decision["score"] == 1
    assert by_id["noul_refund_export_how_to_no"].decision["noul"] == 0


def test_missing_and_empty_files_are_rejected(tmp_path):
    with pytest.raises(DatasetError, match="arquivo ausente"):
        load_eval_decisions(tmp_path / "ausente.jsonl")
    empty = tmp_path / "vazio.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="arquivo vazio"):
        load_eval_decisions(empty)


@pytest.mark.parametrize(
    ("payloads", "match"),
    [
        (["{"], "JSON inválido"),
        (["[]"], "não é um objeto"),
        ([_choice_payload(schema_version=2)], "schema_version"),
        ([_choice_payload(declared_confidence=0.2)], "declared_confidence"),
        ([_choice_payload(category="score")], "categoria difere"),
        ([_choice_payload(expected={"team": {"choice": "legal"}})], "incompatível"),
        ([_score_payload() | {"expected": {"urgency": {"score_level": 9}}}], "score_level"),
        ([_noul_payload() | {"expected": {"refund_requested": {"choice": "billing"}}}], "gabarito"),
        ([_choice_payload(expectation="low_confidence", notes="  ")], "low_confidence exige nota"),
        ([_choice_payload(state="   ")], "state vazio"),
        (
            [
                {
                    **_choice_payload(),
                    "questions": {"damage": {"type": "box", "instructions": "Onde está o risco?"}},
                }
            ],
            "categoria difere",
        ),
    ],
)
def test_loader_rejects_malformed_rows(tmp_path, payloads, match):
    lines = []
    for payload in payloads:
        if isinstance(payload, str):
            lines.append(payload)
        else:
            lines.append(json.dumps(payload, ensure_ascii=False))
    path = tmp_path / "ruim.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=match):
        load_eval_decisions(path)


def test_loader_rejects_blank_line_duplicate_id_and_repeated_material(tmp_path):
    first = _choice_payload("um")
    path = tmp_path / "ruim.jsonl"
    path.write_text(json.dumps(first, ensure_ascii=False) + "\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="linha 2 vazia"):
        load_eval_decisions(path)

    path.write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False)
            for payload in (first, _choice_payload("um", state="Outro texto de cadastro."))
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="id duplicado"):
        load_eval_decisions(path)

    path.write_text(
        "\n".join(
            json.dumps(_choice_payload(task_id), ensure_ascii=False) for task_id in ("um", "dois")
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="estado e perguntas repetidos"):
        load_eval_decisions(path)

    path.write_text("\ufeff" + json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="BOM"):
        load_eval_decisions(path)


def _tasks(rows):
    return [row.as_decision_task() for row in rows]


def _assert_jev_posts_the_same_body(request: SystemOneRequest, body: dict) -> None:
    captured: dict = {}
    question_name, question = next(iter(request.questions.items()))

    def transport(url, headers, raw, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json.loads(raw.decode("utf-8"))
        answer = _schema_answer(question)
        payload = {
            "model": "jev-1.14.0",
            "answers": {question_name: answer},
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        return 200, json.dumps(payload).encode("utf-8")

    backend = JevHttpBackend(
        api_key="test-key",
        model="jev-1.14.0",
        transport=transport,
    )
    response = backend.decide(request)
    assert captured["url"].endswith("/v1/systemone")
    assert captured["body"] == body
    assert "Authorization" in captured["headers"]
    assert "TypeSafe-Danger-Zone-Images" not in captured["headers"]
    assert response.answers[question_name].type == question.type


def _schema_answer(question) -> dict:
    """Resposta mínima só para o cliente HTTP parsear. Não é gabarito nem medição."""
    if question.type == "choice":
        keys = list(question.criteria)
        chosen = keys[0]
        return {
            "type": "choice",
            "choice": chosen,
            "confidence": 0.5,
            "probabilities": {key: (1.0 if key == chosen else 0.0) for key in keys},
        }
    if question.type == "score":
        legend = {str(index): text for index, text in enumerate(question.criteria)}
        return {
            "type": "score",
            "score": 0,
            "confidence": 0.5,
            "legend": legend,
            "probabilities": {key: (1.0 if key == "0" else 0.0) for key in legend},
        }
    return {"type": "noul", "noul": 0.5}
