"""Contrato do placeholder: 10 a 15 itens, quatro primitivas, Box sem imagem e sem rótulo."""

import json

from jev_study.dataset import load_synthetic_tasks


def test_placeholder_covers_primitives_and_skips_only_the_image_task():
    tasks = load_synthetic_tasks()
    assert 10 <= len(tasks) <= 15
    types = {question.type for task in tasks for question in task.questions.values()}
    assert types == {"choice", "score", "noul", "box"}
    skipped = [task for task in tasks if task.skip]
    assert len(skipped) == 1
    assert {question.type for question in skipped[0].questions.values()} == {"box"}
    assert skipped[0].expected == {}
    assert skipped[0].skip_reason is not None
    assert "imagem" in skipped[0].skip_reason.casefold()
    runnable = [task for task in tasks if not task.skip]
    assert runnable
    assert all(task.note.strip() for task in runnable)
    assert all(isinstance(task.state, str) and task.state.strip() for task in tasks)
    blob = json.dumps([task.model_dump(mode="json") for task in tasks])
    assert "base64" not in blob
    assert "data_base64" not in blob


def test_routing_rubric_and_known_labels():
    tasks = {task.id: task for task in load_synthetic_tasks()}
    route = tasks["route_duplicate_charge"]
    question = route.questions["team"]
    assert set(question.criteria) == {"billing", "technical", "account"}
    assert route.expected["team"].choice == "billing"
    assert tasks["route_app_crash"].expected["team"].choice == "technical"
    assert tasks["route_login_email"].expected["team"].choice == "account"
    assert tasks["urgency_payroll_today"].expected["urgency"].score_level == 2
    assert tasks["urgency_by_friday"].expected["urgency"].score_level == 1
    assert tasks["urgency_copy_when_possible"].expected["urgency"].score_level == 0
    assert tasks["refund_requested_yes"].expected["refund_requested"].noul is True
    assert tasks["refund_requested_no"].expected["refund_requested"].noul is False
    assert tasks["legitimate_boleto_not_spam"].expected["unsolicited_ad"].noul is False
